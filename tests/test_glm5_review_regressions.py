from __future__ import annotations

import ast
from pathlib import Path
import unittest

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

    def test_indexer_registration_preserves_pointer_ordered_layer_ids(self):
        _, tree = load_tree(LAYER_SPLIT)
        handler = function_named(tree, "configure_indexer_layer_ids")
        assignments = [
            node
            for node in ast.walk(handler)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Attribute)
                and target.value.attr == "state_layer_ids"
                for target in node.targets
            )
        ]
        self.assertEqual(assignments, [])
        source = ast.unparse(handler)
        self.assertIn("layer_ids = kv_args.state_layer_ids[component]", source)
        self.assertIn("len(kv_args.state_data_ptrs[component])", source)

    def test_pd_registration_pairs_index_k_pointers_with_indexer_layer_ids(self):
        _, tree = load_tree(DISAGG_STATE)
        helper = function_named(tree, "append_hcu_dsa_state")
        source = ast.unparse(helper)
        self.assertIn("target.get_state_layer_ids()", source)
        self.assertIn("draft.get_state_layer_ids()", source)
        self.assertIn("pool.get_state_buf_infos()", source)


if __name__ == "__main__":
    unittest.main()
