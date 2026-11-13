# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import traceback

__all__ = ["DelayedAssert"]


class DelayedAssert:
    """Collect assertions and report them together."""

    def __init__(self, store_stack: bool = False) -> None:
        self.assertions: list[tuple[bool, str, list[str]]] = []
        self.store_stack = store_stack

    def add(self, result: bool, msg: str) -> None:
        self.assertions.append((bool(result), str(msg), traceback.format_stack()))

    def get_msg(self) -> str:
        ret = ["Some assertions failed:"]
        for result, msg, stack in self.assertions:
            ret.append(
                "\n".join(
                    [
                        f"Assert result: {result}",
                        msg,
                        "".join(stack) if self.store_stack else "",
                    ]
                )
            )
        return "Some assertions failed:\n" + "\n-----------------------------------------\n".join(
            ret
        )

    def clear(self) -> None:
        self.assertions.clear()

    def assert_all(self) -> None:
        assert all(ret[0] for ret in self.assertions), self.get_msg()
        self.clear()
