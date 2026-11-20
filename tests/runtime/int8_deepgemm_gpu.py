#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_info(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def persist(out, result):
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(out)


class Mode:
    def __init__(self, name):
        self.name = name

    def is_normal(self):
        return self.name == "normal"

    def enable_normal(self):
        return self.name in ("normal", "auto")

    def enable_low_latency(self):
        return self.name in ("low_latency", "auto")


@contextmanager
def mode_context(mode, asm):
    name = "sglang.srt.layers.moe.utils"
    missing = object()
    original = sys.modules.get(name, missing)
    module = types.ModuleType(name)
    module.get_deepep_mode = lambda: mode
    sys.modules[name] = module
    key = "SGLANG_INT8_DEEPGEMM_ASM"
    old = os.environ.get(key)
    os.environ[key] = "1" if asm else "0"
    try:
        yield
    finally:
        if original is missing:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def tensor_info(tensor):
    raw = tensor.detach().contiguous().cpu().numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": list(tensor.stride()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Real gedf118 packing/GEMM/Graph contract probe; no pack or GEMM mocks."
    )
    parser.add_argument("--plugin-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--experts", type=int, default=3)
    args = parser.parse_args()
    if args.experts < 3:
        parser.error("--experts must be at least 3")
    out = args.out if args.out.suffix == ".json" else args.out / "gpu-int8-contract.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "running",
        "started_unix": time.time(),
        "argv": sys.argv,
        "method": {
            "isolation": "Only sglang.get_deepep_mode and its environment flag are supplied. Candidate weight preparation, all vendor packers, and all GEMMs execute real code.",
            "oracle": "CPU int32 matrix multiplication of original logical INT8 tensors; float32 dyadic activation/weight scales; final BF16 rounding.",
            "numerical_acceptance": "Exact BF16 equality on valid rows. No tolerance is widened for ASM. On failure, absolute error and BF16-ULP diagnostics are saved.",
            "graph": "Changed activation, activation scales, masked_m or m_indices replay at fixed allocated capacity 256.",
            "empty_rows": "Per-expert counts include 0/1/7/63; all-empty replay checks launch completion only because no output rows are defined.",
            "scope": "Synthetic model-shaped GEMM contract, not a full MoE forward, model service, dataset score, or performance measurement.",
        },
        "pack_comparisons": [],
        "cases": [],
    }
    persist(out, result)
    try:
        import torch
        import deepgemm
        import deepgemm.group_gemm_weight_pack as packing

        if not torch.cuda.is_available() or not torch.version.hip:
            raise RuntimeError("This probe requires a real HIP device; CPU-only execution cannot pass.")
        torch.cuda.set_device(args.device)
        properties = torch.cuda.get_device_properties(args.device)
        arch = getattr(properties, "gcnArchName", properties.name).split(":")[0]
        if arch != "gfx936":
            raise RuntimeError("The pinned gedf118 comparison is defined for gfx936, got " + arch)
        helper = args.plugin_dir / "hcu_sglang/layers/quantization/hcu_deepgemm_w8a8_utils.py"
        spec = importlib.util.spec_from_file_location("candidate_int8_weight_contract", helper)
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
        vendor_root = Path(deepgemm.__file__).resolve().parent
        sources = [vendor_root / name for name in (
            "__init__.py", "group_gemm_weight_pack.py",
            "m_group_gemm_nt_contiguous.py", "m_group_gemm_nt_masked.py"
        )]
        sources.extend(sorted(vendor_root.glob("op*.so")))
        result["runtime"] = {
            "python": sys.version,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "deepgemm_distribution": importlib.metadata.version("deepgemm"),
            "device": args.device,
            "arch": arch,
            "device_name": properties.name,
            "plugin_helper": file_info(helper),
            "vendor_files": [file_info(path) for path in sources],
        }
        expected_native = "40378fafacb43b69b1eacc20759b9b4fd78963d01e34adee24ea8785b4380f38"
        actual_native = [v["sha256"] for v in result["runtime"]["vendor_files"] if v["path"].endswith(".so")]
        if actual_native != [expected_native]:
            raise RuntimeError("Native library differs from pinned gedf118 evidence: " + repr(actual_native))
        persist(out, result)
        experts, capacity = args.experts, 256
        counts_first = [0, 1, 7] + [0] * (experts - 3)
        counts_second = [63, 0, 1] + [0] * (experts - 3)
        counts_empty = [0] * experts
        models = {
            "glm53": [("gate_up", 4096, 4096), ("down", 4096, 2048)],
            "hy3": [("gate_up", 3072, 4096), ("down", 4096, 1536)],
        }
        reference_cache = {}
        generator = torch.Generator(device="cpu").manual_seed(20261120)

        def valid_mask(counts):
            return torch.arange(capacity).unsqueeze(0) < torch.tensor(counts).unsqueeze(1)

        def indices_for(counts):
            indices = torch.full((experts, capacity), -1, dtype=torch.int32)
            for expert, count in enumerate(counts):
                indices[expert, :count] = expert
            return indices.flatten()

        def reference(data, phase, counts):
            key = (data["key"], phase, tuple(counts))
            if key not in reference_cache:
                a_cpu = data["a_cpu"][phase]
                sa_cpu = data["sa_cpu"][phase]
                pieces = []
                for expert, count in enumerate(counts):
                    if not count:
                        continue
                    accumulated = torch.mm(
                        a_cpu[expert, :count].to(torch.int32),
                        data["weight_cpu"][expert].to(torch.int32).transpose(0, 1),
                    )
                    value = accumulated.to(torch.float32) * sa_cpu[expert, :count]
                    value = value * data["sb_cpu"][expert].transpose(0, 1)
                    pieces.append(value.to(torch.bfloat16))
                reference_cache[key] = torch.cat(pieces) if pieces else torch.empty((0, data["n"]), dtype=torch.bfloat16)
            return reference_cache[key]

        def check(output, data, phase, counts, where):
            observed = output.reshape(experts, capacity, data["n"]).detach().cpu()[valid_mask(counts)]
            expected = reference(data, phase, counts)
            stats = {"phase": where, "counts": counts, "valid_elements": observed.numel()}
            if not expected.numel():
                stats["status"] = "launch_completed_no_valid_rows"
                return stats
            equal = torch.equal(observed, expected)
            delta = (observed.float() - expected.float()).abs()
            stats.update({"exact_bf16_equal": equal, "max_abs_error": delta.max().item(), "mismatches": (observed != expected).sum().item()})
            if not equal:
                toward = torch.full_like(expected, float("inf"))
                spacing = (torch.nextafter(expected, toward).float() - expected.float()).abs()
                stats["max_expected_bf16_ulp_error"] = (delta / spacing.clamp_min(torch.finfo(torch.float32).tiny)).max().item()
                nonzero = (observed != expected).nonzero()[:8]
                stats["first_mismatches"] = [{"row_col": p.tolist(), "actual": observed[tuple(p)].item(), "expected": expected[tuple(p)].item()} for p in nonzero]
            return stats

        for model, matrices in models.items():
            data_by_name = {}
            for name, n, k in matrices:
                weight_cpu = torch.randint(-7, 8, (experts, n, k), dtype=torch.int8, generator=generator)
                a_first = torch.randint(-7, 8, (experts, capacity, k), dtype=torch.int8, generator=generator)
                a_second = a_first + 1
                sa_first = torch.pow(2.0, -5 + torch.arange(experts * capacity).reshape(experts, capacity, 1) % 3)
                sa_second = sa_first * 2
                sb_cpu = torch.pow(2.0, -5 + torch.arange(experts * n).reshape(experts, n, 1) % 3)
                data = {"key": model + "/" + name, "name": name, "n": n, "k": k, "weight_cpu": weight_cpu, "weight_gpu": weight_cpu.cuda(), "a_cpu": [a_first, a_second], "sa_cpu": [sa_first, sa_second], "sb_cpu": sb_cpu, "sb_gpu": sb_cpu.cuda()}
                data_by_name[name] = data
                contiguous = deepgemm.marlin_i8_contiguous_weight(data["weight_gpu"].clone(), shuffle_unique=0, low_memory=True)
                masked = deepgemm.marlin_i8_masked_weight(data["weight_gpu"].clone(), shuffle_unique=0, low_memory=True)
                pair32 = packing.pack_i8_weight_pair32(data["weight_gpu"].clone())
                comparison = {"model": model, "matrix": name, "logical_shape": list(weight_cpu.shape), "contiguous": tensor_info(contiguous), "masked": tensor_info(masked), "pair32": tensor_info(pair32), "masked_pair32_exact": torch.equal(masked, pair32), "contiguous_pair32_bytes_equal": torch.equal(contiguous.flatten(), pair32.flatten())}
                result["pack_comparisons"].append(comparison)
                persist(out, result)
                if not comparison["masked_pair32_exact"] or comparison["contiguous_pair32_bytes_equal"] or contiguous.dim() != 3 or masked.dim() != 7:
                    raise AssertionError("Pinned gfx936/gedf118 layout contract differs from expected: " + repr(comparison))
                del contiguous, masked, pair32

            for mode_name in ("normal", "low_latency", "auto"):
                for asm in (False, True):
                    mode = Mode(mode_name)
                    layer = torch.nn.Module()
                    layer.register_buffer("w13_weight", data_by_name["gate_up"]["weight_gpu"].clone())
                    layer.register_buffer("w2_weight", data_by_name["down"]["weight_gpu"].clone())
                    with mode_context(mode, asm):
                        candidate.prepare_w8a8_int8_deepgemm_weights(layer)
                        candidate.prepare_w8a8_int8_deepgemm_weights(layer)
                    expected_paths = []
                    if mode.enable_normal():
                        expected_paths.append("contiguous")
                    if mode.enable_low_latency():
                        expected_paths.append("masked")
                    for name, data in data_by_name.items():
                        prefix = "w13" if name == "gate_up" else "w2"
                        for path in expected_paths:
                            suffix = "_masked" if path == "masked" else ""
                            packed = getattr(layer, prefix + "_weight_deepgemm" + suffix)
                            if path == "contiguous":
                                oracle_pack = deepgemm.marlin_i8_contiguous_weight(data["weight_gpu"].clone(), shuffle_unique=0, low_memory=True)
                            elif asm:
                                oracle_pack = packing.pack_i8_weight_gfx936_asm(data["weight_gpu"].clone())
                            else:
                                oracle_pack = deepgemm.marlin_i8_masked_weight(data["weight_gpu"].clone(), shuffle_unique=0, low_memory=True)
                            item = {"model": model, "matrix": name, "mode": mode_name, "asm": asm, "path": path, "packed_shape": list(packed.shape), "pack_exact": torch.equal(packed, oracle_pack), "checks": [], "status": "running"}
                            result["cases"].append(item)
                            persist(out, result)
                            if not item["pack_exact"]:
                                raise AssertionError("Candidate pack disagrees with the unmodified real vendor pack: " + repr(item))
                            del oracle_pack
                            a = data["a_cpu"][0].cuda()
                            sa = data["sa_cpu"][0].cuda()
                            sb = data["sb_gpu"]
                            counts = torch.tensor(counts_first, dtype=torch.int32, device="cuda")
                            indices = indices_for(counts_first).cuda()
                            output = torch.full((experts, capacity, data["n"]), float("nan"), dtype=torch.bfloat16, device="cuda")

                            def run():
                                if path == "contiguous":
                                    return candidate.run_w8a8_int8_contiguous_gemm(layer, (a.reshape(experts * capacity, data["k"]), sa.reshape(experts * capacity, 1)), (packed, sb), output.reshape(experts * capacity, data["n"]), indices)
                                if asm:
                                    return deepgemm.m_grouped_w8a8_gemm_nt_masked((a, sa), (packed, sb), output, counts, capacity)
                                return deepgemm.m_grouped_i8_gemm_nt_masked((a, sa), (packed, sb), output, counts, capacity, shuffle_unique=0, config={"MODE": 1001})

                            run()
                            torch.cuda.synchronize()
                            item["checks"].append(check(output, data, 0, counts_first, "eager"))
                            persist(out, result)
                            if not item["checks"][-1]["exact_bf16_equal"]:
                                raise AssertionError("Eager numerical mismatch; see saved diagnostics")
                            capture_stream = torch.cuda.Stream()
                            capture_stream.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(capture_stream):
                                run()
                            torch.cuda.current_stream().wait_stream(capture_stream)
                            torch.cuda.synchronize()
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph, stream=capture_stream):
                                run()
                            graph.replay()
                            torch.cuda.synchronize()
                            item["checks"].append(check(output, data, 0, counts_first, "graph_unchanged"))
                            a.copy_(data["a_cpu"][1])
                            sa.copy_(data["sa_cpu"][1])
                            counts.copy_(torch.tensor(counts_second, dtype=torch.int32))
                            indices.copy_(indices_for(counts_second))
                            output.fill_(float("nan"))
                            graph.replay()
                            torch.cuda.synchronize()
                            item["checks"].append(check(output, data, 1, counts_second, "graph_changed_input_and_counts"))
                            persist(out, result)
                            if any(not c.get("exact_bf16_equal", False) for c in item["checks"]):
                                raise AssertionError("Graph numerical mismatch; see saved diagnostics")
                            counts.zero_()
                            indices.fill_(-1)
                            graph.replay()
                            torch.cuda.synchronize()
                            item["checks"].append(check(output, data, 1, counts_empty, "graph_all_experts_empty"))
                            item["status"] = "passed"
                            persist(out, result)
                            print(json.dumps({k: item[k] for k in ("model", "matrix", "mode", "asm", "path", "status")}), flush=True)
                            del graph, capture_stream, a, sa, counts, indices, output
                    del layer
                    torch.cuda.empty_cache()
            del data_by_name
            torch.cuda.empty_cache()
        result["status"] = "passed"
        result["summary"] = {"cases_passed": len(result["cases"]), "pack_comparisons": len(result["pack_comparisons"]), "models": list(models), "experts": experts, "capacity": capacity, "counts_first": counts_first, "counts_changed": counts_second}
    except BaseException as exc:
        result["status"] = "failed"
        result["exception"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        persist(out, result)
        raise
    finally:
        result["finished_unix"] = time.time()
        persist(out, result)
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
