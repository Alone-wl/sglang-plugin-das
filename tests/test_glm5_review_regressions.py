from __future__ import annotations

import ast
from pathlib import Path
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
EMPTY_BATCH = ROOT / "hcu_sglang/layers/attention/dsa_empty_batch.py"
LAYER_SPLIT = ROOT / "hcu_sglang/mem_cache/glm5_layer_split.py"
DISAGG_STATE = ROOT / "hcu_sglang/disagg/glm5_next.py"


def load_tree(path: Path):
    text = path.read_text(encoding="utf-8")
    return text, ast.parse(text)


def function_named(tree, name):
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def registrations(tree, function_name):
    function = function_named(tree, function_name)
    return {
        node.args[0].value
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "register"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }


class Glm5ReviewRegressionTests(unittest.TestCase):
    def test_empty_verify_guard_covers_native_backend_and_uses_its_metadata_type(self):
        _, tree = load_tree(EMPTY_BATCH)
        targets = registrations(tree, "register_hooks")
        self.assertIn(
            "hcu_sglang.layers.attention.glm5_next.dsa_backend.NativeSparseAttnBackend.init_forward_metadata",
            targets,
        )
        handler = function_named(tree, "around_init_forward_metadata")
        source = ast.unparse(handler)
        self.assertIn("import_module(backend.__class__.__module__)", source)
        self.assertIn("batch.batch_size != 0", source)
        self.assertIn("batch.forward_mode.is_target_verify()", source)

    def test_layer_split_pages_hook_covers_native_backend(self):
        _, tree = load_tree(LAYER_SPLIT)
        targets = registrations(tree, "register_hooks")
        self.assertIn(
            "hcu_sglang.layers.attention.glm5_next.dsa_backend.NativeSparseAttnBackend.init_forward_metadata",
            targets,
        )

    def test_pd_state_preserves_index_ownership_and_full_tail_identity(self):
        class Hybrid:
            def __init__(self, inner):
                self.full_kv_pool = inner
                self._hcu_glm5_cache = True
                self.use_dsa = True

        class Pool:
            def __init__(self, start, count, index_ids, base):
                self.start_layer, self.layer_num = start, count
                self.index_ids = index_ids
                self.kpool_use_compress = True
                self.index_ptrs = [base + i for i in index_ids]
                self.tail_ptrs = [base + 100 + i for i in range(count * 2)]

            def get_state_layer_ids(self):
                return list(self.index_ids)

            def get_state_buf_infos(self):
                return self.index_ptrs, [32] * len(self.index_ptrs), [8] * len(self.index_ptrs)

            def get_compress_tail_buf_infos(self):
                items = [8 if self.start_layer + i % self.layer_num in self.index_ids else 0
                         for i in range(self.layer_num * 2)]
                return self.tail_ptrs, [32] * len(self.tail_ptrs), items

        state_type = types.SimpleNamespace(DSA="index", DSA_TAIL="tail")
        namespace = {"StateType": state_type, "HybridLinearKVPool": Hybrid, "is_hcu": lambda: True}
        for path, names in (
            (ROOT / "hcu_sglang/disagg/utils.py", {"append_state_component"}),
            (DISAGG_STATE, {"append_hcu_dsa_state"}),
            (LAYER_SPLIT, {"configure_indexer_layer_ids"}),
        ):
            _, tree = load_tree(path)
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
            unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions], type_ignores=[])
            exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), namespace)
        native = types.ModuleType("sglang.srt.disaggregation.utils")
        native.append_state_component = namespace["append_state_component"]
        memory = types.ModuleType("sglang.srt.mem_cache.memory_pool")
        memory.HybridLinearKVPool = Hybrid
        protocol = types.ModuleType("hcu_sglang.disagg.protocol")
        protocol.StateType = state_type
        modules = {native.__name__: native, memory.__name__: memory, protocol.__name__: protocol}
        fields = ("state_types", "state_data_ptrs", "state_data_lens", "state_item_lens",
                  "state_data_formats", "state_dim_per_tensor", "state_conv_shard_groups",
                  "state_slice_outer_counts", "state_layer_ids")

        for owned in ([4, 6], [4], []):
            with self.subTest(owned=owned), patch.dict("sys.modules", modules):
                target, draft = Pool(4, 3, list(owned), 1000), Pool(13, 1, [13], 2000)
                wrapper = Hybrid(target)
                args = types.SimpleNamespace(**{name: [] for name in fields})
                namespace["append_hcu_dsa_state"](args, wrapper, draft, 45)
                self.assertEqual(args.state_types, ["index", "tail"])
                self.assertEqual(list(zip(args.state_data_ptrs[0], args.state_layer_ids[0])),
                                 [(1000 + i, i) for i in owned] + [(2013, 45)])
                self.assertEqual(args.state_data_ptrs[1], target.tail_ptrs + draft.tail_ptrs)
                self.assertEqual(args.state_layer_ids[1], [4, 5, 6, 4, 5, 6, 45, 45])
                self.assertEqual(len(args.state_item_lens[1]), 8)
                for field in ("state_data_ptrs", "state_data_lens", "state_item_lens", "state_layer_ids"):
                    getattr(args, field)[0].reverse()
                before = list(zip(args.state_data_ptrs[0], args.state_layer_ids[0]))
                namespace["configure_indexer_layer_ids"](args, wrapper)
                self.assertEqual(list(zip(args.state_data_ptrs[0], args.state_layer_ids[0])), before)
                args.state_layer_ids[0].pop()
                with self.assertRaisesRegex(ValueError, "buffer count"):
                    namespace["configure_indexer_layer_ids"](args, wrapper)

        with patch.dict("sys.modules", modules):
            args = types.SimpleNamespace(**{name: [] for name in fields})
            with self.assertRaisesRegex(ValueError, "total_kv_layers"):
                namespace["append_hcu_dsa_state"](args, Hybrid(Pool(4, 3, [4], 1000)), Pool(13, 1, [13], 2000), None)
            broken = Pool(4, 3, [4], 1000)
            broken.tail_ptrs.pop()
            with self.assertRaisesRegex(ValueError, "tail buffers"):
                namespace["append_hcu_dsa_state"](args, Hybrid(broken), None, 45)


if __name__ == "__main__":
    unittest.main()
