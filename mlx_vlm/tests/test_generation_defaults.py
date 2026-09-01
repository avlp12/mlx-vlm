"""The CLI and the library must agree on every default they both define.

`mlx_vlm.generate` (the CLI) and `generate_step` (the library) used to declare
their own copies of the same constants, so one side could be changed without
the other. These tests compare the two by introspection rather than by
restating the values, so they keep working when a default is deliberately
changed and still fail when the two sides drift apart.
"""

import inspect
import sys
from unittest.mock import patch

from mlx_vlm.generate import ar, common, dispatch
from mlx_vlm.generate.ar import generate_step
from mlx_vlm.generate.dispatch import parse_arguments


def _cli_defaults():
    with patch.object(sys, "argv", ["mlx_vlm.generate"]):
        return vars(parse_arguments())


def _library_defaults():
    return {
        name: parameter.default
        for name, parameter in inspect.signature(generate_step).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def test_cli_and_library_agree_on_shared_defaults():
    cli, library = _cli_defaults(), _library_defaults()
    shared = sorted(set(cli) & set(library))

    # Guard against this test quietly comparing nothing: a rename on either
    # side would otherwise empty the intersection and still pass.
    assert len(shared) >= 10, f"expected many shared defaults, found {shared}"

    # A CLI default of None means "not given on the command line", and the
    # value is resolved further down (--draft-kind is inferred from the model
    # type, for instance). Any other disagreement is drift.
    mismatched = {
        name: {"cli": cli[name], "library": library[name]}
        for name in shared
        if cli[name] is not None and cli[name] != library[name]
    }
    assert not mismatched, (
        f"CLI and library defaults disagree: {mismatched}. Both sides should "
        "read the constant from mlx_vlm.generate.common."
    )


def test_shared_defaults_are_defined_once():
    """Modules that expose a default re-export `common`'s, never their own copy."""
    constants = [name for name in dir(common) if name.startswith("DEFAULT_")]
    assert constants, "no DEFAULT_* constants found in mlx_vlm.generate.common"

    for module in (ar, dispatch):
        for name in constants:
            if hasattr(module, name):
                assert getattr(module, name) is getattr(common, name), (
                    f"{module.__name__}.{name} is not the object defined in "
                    "mlx_vlm.generate.common"
                )


# --------------------------------------------------------------------------
# The speculative GPU loop must run inside wired_limit.
#
# _run_speculative owns the GPU thread for its whole life but never constructed
# a BatchGenerator, which is what wires the model (generate/ar.py:2345). So the
# speculative path ran UNWIRED and the model paged on every forward. An unwired
# forward costs about 1690 ms whatever the block width -- against 34 ms at width
# 1 and 59 ms at width 5 wired -- and the server's verify measured 1688.8 ms.
# That is why the cost looked uniform across all 45 layers and independent of
# width, and why twelve component-level hypotheses all came back at 1.00x.
#
# Live effect of wiring it: 1.64 -> 45.28 tok/s, with acceptance and round counts
# unchanged. These tests pin the invariant so the path cannot silently unwire.
# --------------------------------------------------------------------------
def test_speculative_loop_is_dispatched_inside_wired_limit():
    import ast
    import inspect
    import textwrap

    from mlx_vlm.server import generation as gen

    # getsource on a method keeps the class indentation; ast needs it dedented
    tree = ast.parse(textwrap.dedent(inspect.getsource(gen.ResponseGenerator._run_impl)))

    def _calls(node):
        return {
            n.func.attr
            for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }

    guarded = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        names = {
            item.context_expr.func.id
            for item in node.items
            if isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Name)
        }
        if "wired_limit" in names and "_run_speculative" in _calls(node):
            guarded = True
            break

    assert guarded, (
        "_run_speculative() must be called inside a `with wired_limit(...)` block. "
        "Without it the speculative path runs unwired and every forward costs "
        "~1690 ms instead of ~59 ms (measured 1.64 vs 45.28 tok/s end to end)."
    )


def test_speculative_dispatch_is_the_only_unguarded_path():
    """If _run_speculative ever gains a second call site, it must be guarded too."""
    import ast
    import inspect

    from mlx_vlm.server import generation as gen

    src = inspect.getsource(gen)
    tree = ast.parse(src)
    sites = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_run_speculative"
    ]
    assert len(sites) == 1, (
        f"expected exactly one _run_speculative() call site, found {len(sites)}. "
        "Every site must sit inside a wired_limit block; update this test and "
        "the guard together."
    )
