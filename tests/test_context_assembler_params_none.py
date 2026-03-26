"""Tests for the params=None fix in context_assembler."""
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers to construct a minimal ContextAssembler without real disk I/O
# ---------------------------------------------------------------------------

def _make_assembler():
    """Return a ContextAssembler instance with no real filesystem dependencies."""
    from backend.rag.context_assembler import ContextAssembler

    assembler = ContextAssembler.__new__(ContextAssembler)
    assembler.data_dir = MagicMock()
    assembler.repo_name = "test_repo"
    assembler._enriched = None
    assembler.max_tokens = 4000
    return assembler


# ---------------------------------------------------------------------------
# _render_function_detail
# ---------------------------------------------------------------------------

class TestRenderFunctionDetailParamsNone:
    """Acceptance criteria for _render_function_detail with params=None."""

    def test_params_none_does_not_raise(self):
        """params=None must not raise TypeError or any exception."""
        assembler = _make_assembler()
        node = {
            "name": "my_func",
            "file": "mymodule.py",
            "params": None,
        }
        # Should not raise
        result = assembler._render_function_detail(node)
        assert isinstance(result, str)

    def test_params_none_same_as_empty_list(self):
        """Output for params=None must equal output for params=[]."""
        assembler = _make_assembler()
        node_none = {
            "name": "my_func",
            "file": "mymodule.py",
            "params": None,
        }
        node_empty = {
            "name": "my_func",
            "file": "mymodule.py",
            "params": [],
        }
        assert assembler._render_function_detail(node_none) == assembler._render_function_detail(node_empty)

    def test_params_key_absent_does_not_raise(self):
        """A node with no 'params' key at all must still work correctly."""
        assembler = _make_assembler()
        node = {
            "name": "no_params_key",
            "file": "mymodule.py",
        }
        result = assembler._render_function_detail(node)
        assert isinstance(result, str)
        assert "no_params_key()" in result

    def test_params_normal_list_still_rendered(self):
        """Normal params list must still be rendered correctly."""
        assembler = _make_assembler()
        node = {
            "name": "greet",
            "file": "hello.py",
            "params": ["name", "greeting"],
        }
        result = assembler._render_function_detail(node)
        assert "name, greeting" in result

    def test_params_none_produces_empty_param_string(self):
        """params=None should result in an empty parameter string in the signature."""
        assembler = _make_assembler()
        node = {
            "name": "func",
            "file": "mod.py",
            "params": None,
        }
        result = assembler._render_function_detail(node)
        # The signature should look like func() with nothing between parens
        assert "func()" in result


# ---------------------------------------------------------------------------
# _render_related
# ---------------------------------------------------------------------------

class TestRenderRelatedParamsNone:
    """_render_related must also tolerate params=None."""

    def test_render_related_params_none_does_not_raise(self):
        assembler = _make_assembler()
        enriched = {
            "node1": {
                "type": "function",
                "name": "helper",
                "file": "utils.py",
                "params": None,
            }
        }
        # Should not raise
        result = assembler._render_related(["node1"], enriched, [])
        assert isinstance(result, str)

    def test_render_related_params_none_same_as_empty(self):
        assembler = _make_assembler()
        enriched_none = {
            "node1": {
                "type": "function",
                "name": "helper",
                "file": "utils.py",
                "params": None,
            }
        }
        enriched_empty = {
            "node1": {
                "type": "function",
                "name": "helper",
                "file": "utils.py",
                "params": [],
            }
        }
        result_none = assembler._render_related(["node1"], enriched_none, [])
        result_empty = assembler._render_related(["node1"], enriched_empty, [])
        assert result_none == result_empty


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------

class TestPipelineWithParamsNone:
    """The full context assembly pipeline must succeed when params=None is present."""

    def test_pipeline_completes_with_params_none(self):
        """assemble() must return a valid string (not crash) when nodes have params=None."""
        from backend.rag.context_assembler import ContextAssembler

        assembler = ContextAssembler.__new__(ContextAssembler)
        assembler.data_dir = MagicMock()
        assembler.repo_name = "test_repo"
        assembler.max_tokens = 4000

        # Seed the enriched cache directly so no disk I/O happens
        assembler._enriched = {
            "mod.py::bad_func": {
                "type": "function",
                "name": "bad_func",
                "file": "mod.py",
                "params": None,   # <-- the problematic value
                "docstring": "A function whose params failed to parse.",
                "complexity": 2,
            },
            "mod.py::good_func": {
                "type": "function",
                "name": "good_func",
                "file": "mod.py",
                "params": ["x", "y"],
                "docstring": "A normal function.",
                "complexity": 1,
            },
        }

        # Build a minimal retrieval result that references these nodes
        retrieval_result = {
            "primary": [
                {
                    "node_id": "mod.py::bad_func",
                    "score": 0.95,
                    "node": assembler._enriched["mod.py::bad_func"],
                },
                {
                    "node_id": "mod.py::good_func",
                    "score": 0.80,
                    "node": assembler._enriched["mod.py::good_func"],
                },
            ],
            "related": [],
            "edges": [],
        }

        # _render_primary and _render_related are the paths that crash;
        # call them directly to verify the pipeline sections work.
        primary_text = assembler._render_primary(retrieval_result["primary"])
        assert isinstance(primary_text, str)
        assert "bad_func" in primary_text

        related_text = assembler._render_related(
            ["mod.py::bad_func", "mod.py::good_func"],
            assembler._enriched,
            [],
        )
        assert isinstance(related_text, str)

    def test_multiple_none_params_nodes_pipeline(self):
        """Pipeline must handle multiple nodes all having params=None."""
        from backend.rag.context_assembler import ContextAssembler

        assembler = ContextAssembler.__new__(ContextAssembler)
        assembler.data_dir = MagicMock()
        assembler.repo_name = "test_repo"
        assembler.max_tokens = 4000
        assembler._enriched = {
            f"mod.py::func_{i}": {
                "type": "function",
                "name": f"func_{i}",
                "file": "mod.py",
                "params": None,
            }
            for i in range(5)
        }

        primary_nodes = [
            {"node_id": nid, "score": 0.9, "node": node}
            for nid, node in assembler._enriched.items()
        ]

        # Must not raise
        result = assembler._render_primary(primary_nodes)
        assert isinstance(result, str)
        for i in range(5):
            assert f"func_{i}" in result
