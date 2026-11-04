# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write station.log, a synthetic service log of 3,000 lines.

The content comes from a fixed seed, so every run writes the same file.
"""

LINES = 3000
NODES = [f"node-{number:02d}" for number in range(1, 13)]
BUSY_NODE = "node-07"
OUTPUT = "station.log"


class Sequence:
    """A small linear congruential generator: the same numbers on every Python version."""

    def __init__(self, seed):
        self.state = seed

    def below(self, bound):
        """Return the next number in [0, bound)."""
        self.state = (self.state * 1103515245 + 12345) % 2**31
        return (self.state >> 8) % bound


def routine(seq):
    """Return the message of an ordinary INFO line."""
    kind = seq.below(5)
    if kind == 0:
        return f"heartbeat load={seq.below(90) / 100 + 0.05:.2f} queue={seq.below(12)}"
    if kind == 1:
        return f"job-{seq.below(9000) + 1000} done in {seq.below(900) + 40} ms"
    if kind == 2:
        return f"cache sweep, {seq.below(40) + 1} entries"
    if kind == 3:
        return f"sync with {NODES[seq.below(len(NODES))]} ok"
    return f"metrics flushed, {seq.below(500) + 20} records"


def warning(seq):
    """Return the message of a WARN line."""
    kind = seq.below(3)
    if kind == 0:
        return f"temperature {76 + seq.below(12)} C, limit 75"
    if kind == 1:
        return f"queue depth {101 + seq.below(80)}, limit 100"
    return f"disk /data at {91 + seq.below(8)} percent"


def error(seq):
    """Return the message of an ERROR line."""
    kind = seq.below(3)
    dropped = 1 + seq.below(40)
    if kind == 0:
        return f"job-{seq.below(9000) + 1000} failed after 3 retries, dropped={dropped}"
    if kind == 1:
        return f"lost contact with {NODES[seq.below(len(NODES))]}, dropped={dropped}"
    return f"disk /data full, writes rejected, dropped={dropped}"


def generate(seed=20260302):
    """Return the lines of the log."""
    seq = Sequence(seed)
    seconds = 0
    recoveries = {}
    lines = []
    for index in range(LINES):
        seconds += 2 + seq.below(13)
        clock = f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
        if index == 0:
            lines.append(f"{clock} node-01 INFO station log started, date 2026-03-02")
            continue
        if index in recoveries:
            lines.append(f"{clock} {recoveries.pop(index)} INFO recovered, backlog cleared")
            continue
        node = NODES[seq.below(len(NODES))]
        roll = seq.below(1000)
        if roll < 10 and seq.below(3) == 0:
            node = BUSY_NODE
        if roll < 4:
            lines.append(f"{clock} {node} ERROR {error(seq)}")
            later = index + 20 + seq.below(60)
            while later in recoveries:
                later += 1
            recoveries[later] = node
        elif roll < 10:
            lines.append(f"{clock} {node} WARN {warning(seq)}")
        else:
            lines.append(f"{clock} {node} INFO {routine(seq)}")
    return lines


def main():
    lines = generate()
    with open(OUTPUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {OUTPUT}: {len(lines)} lines")


if __name__ == "__main__":
    main()
