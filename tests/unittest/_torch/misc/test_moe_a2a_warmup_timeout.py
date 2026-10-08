import unittest
from unittest import mock

from tensorrt_llm._torch.pyexecutor import model_engine
from tensorrt_llm._torch.pyexecutor.engine.runners import common
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine


class _WarmupScopeStub:
    """Minimal object that reuses the engine's warmup scope.

    Building a real PyTorchModelEngine needs a model and a device; the scope
    itself only touches _warmup_depth, the MoE all-to-all budget selector, and
    moe_load_balancer (skipped when it is None), so a stub exercises the real
    code path without either.

    The stub borrows the methods without inheriting, so it has to declare
    moe_load_balancer and _warmup_depth itself.
    """

    warmup_scope = PyTorchModelEngine.warmup_scope
    _set_warmup_effects = PyTorchModelEngine._set_warmup_effects
    moe_load_balancer = None

    def __init__(self):
        self._warmup_depth = 0


class TestMoeA2AWarmupBudget(unittest.TestCase):
    """The MoE all-to-all completion-flag budget must track the warmup phase.

    The kernel-side deadline is only safe if it is raised for warmup *and*
    lowered again afterwards; a budget that latches on would leave the hang
    watchdog permanently relaxed in steady state. See nvbugs/6482566.
    """

    def test_set_warmup_forwards_value_to_op(self):
        with mock.patch.object(
            model_engine.torch.ops.trtllm, "moe_a2a_set_warmup", create=True
        ) as op:
            model_engine.set_moe_a2a_warmup(True)
            model_engine.set_moe_a2a_warmup(False)
        self.assertEqual([c.args[0] for c in op.call_args_list], [True, False])

    def test_missing_op_is_tolerated(self):
        """An older C++ build without the op must not break startup."""
        with mock.patch.object(
            model_engine.torch.ops.trtllm,
            "moe_a2a_set_warmup",
            create=True,
            side_effect=AttributeError("no such op"),
        ):
            model_engine.set_moe_a2a_warmup(True)  # must not raise

    def test_capture_context_selects_steady_state_then_restores(self):
        """CUDA graphs bake the budget in at capture time.

        Capture runs inside the warmup window, so the context manager must hand
        the kernel the steady-state budget and restore warmup afterwards.
        """
        seen = []
        with mock.patch.object(common, "set_moe_a2a_warmup", side_effect=seen.append):
            with common.moe_a2a_steady_state_budget_for_capture():
                self.assertEqual(seen, [False])
            self.assertEqual(seen, [False, True])

    def test_warmup_scope_switches_budget_both_ways(self):
        """Regression: the budget must not latch on after warmup.

        PyExecutor's warmup enters the engine's scope, and engine.warmup()
        enters it again inside. Only the outermost scope may switch the
        budget, and leaving it, even by an exception, must restore the
        steady-state budget for the whole serving lifetime.
        """
        stub = _WarmupScopeStub()
        seen = []
        with mock.patch.object(model_engine, "set_moe_a2a_warmup", side_effect=seen.append):
            with stub.warmup_scope():
                with stub.warmup_scope():
                    self.assertEqual(seen, [True])
                self.assertEqual(seen, [True])
            with self.assertRaises(RuntimeError):
                with stub.warmup_scope():
                    raise RuntimeError("warmup failure")

        self.assertEqual(seen, [True, False, True, False])
        self.assertEqual(stub._warmup_depth, 0)


if __name__ == "__main__":
    unittest.main()
