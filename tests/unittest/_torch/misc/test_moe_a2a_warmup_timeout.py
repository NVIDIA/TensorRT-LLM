import unittest
from unittest import mock

from tensorrt_llm._torch.pyexecutor import model_engine
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine


class _WarmupFlagStub:
    """Minimal object that reuses the engine's is_warmup property.

    Building a real PyTorchModelEngine needs a model and a device; the property
    itself only touches _is_warmup, the MoE all-to-all budget selector, and
    moe_load_balancer_iter_info (a no-op without a balancer), so a stub exercises
    the real code path without either.
    """

    is_warmup = PyTorchModelEngine.is_warmup
    moe_load_balancer_iter_info = PyTorchModelEngine.moe_load_balancer_iter_info


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
            model_engine._set_moe_a2a_warmup(True)
            model_engine._set_moe_a2a_warmup(False)
        self.assertEqual([c.args[0] for c in op.call_args_list], [True, False])

    def test_missing_op_is_tolerated(self):
        """An older C++ build without the op must not break startup."""
        with mock.patch.object(
            model_engine.torch.ops.trtllm,
            "moe_a2a_set_warmup",
            create=True,
            side_effect=AttributeError("no such op"),
        ):
            model_engine._set_moe_a2a_warmup(True)  # must not raise

    def test_capture_context_selects_steady_state_then_restores(self):
        """CUDA graphs bake the budget in at capture time.

        Capture runs inside the warmup window, so the context manager must hand
        the kernel the steady-state budget and restore warmup afterwards.
        """
        seen = []
        with mock.patch.object(model_engine, "_set_moe_a2a_warmup", side_effect=seen.append):
            with model_engine._moe_a2a_steady_state_budget_for_capture():
                self.assertEqual(seen, [False])
            self.assertEqual(seen, [False, True])

    def test_is_warmup_setter_switches_budget_both_ways(self):
        """Regression: the budget must not latch on after warmup.

        PyExecutor sets is_warmup=True before calling warmup() and False after,
        both through this setter. Selecting the budget anywhere else (e.g. only
        in set_warmup_flag) leaves the relaxed warmup budget in force for the
        whole serving lifetime.
        """
        stub = _WarmupFlagStub()
        seen = []
        with mock.patch.object(model_engine, "_set_moe_a2a_warmup", side_effect=seen.append):
            stub.is_warmup = True
            stub.is_warmup = False

        self.assertEqual(seen, [True, False])
        self.assertFalse(stub.is_warmup)


if __name__ == "__main__":
    unittest.main()
