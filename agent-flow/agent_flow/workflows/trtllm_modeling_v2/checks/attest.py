# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declared equals observed: a deployment's options against what each rank's engine runs.

Every rank's worker prints one record as its engine comes up (``harness/rankhook``): the
engine's resolved configuration, the class of the model it built, where it imported
``tensorrt_llm`` from, and the host and devices it runs on. A session passes each record
on with the server it came from. :func:`attest` compares every key of that server's
options with the record, for every rank the options ask for: a unique ``(role, instance,
world rank)`` each. A key or a rank with no record reads ``unknown``; an attestation with
a mismatch or an unknown is a failure of the check, never a pass. No option is known here
by name: a new dtype option needs no code.

Before any load a session says which devices each of its servers was given
(:func:`devices_problem`): two servers of one node never share a device.

What this shows is what the engine was configured with. That a cache really reused is
shown by the answers themselves (:func:`reuse_shortfall`): with block reuse on, a session
sends one prompt again and the engine must report cached tokens.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ..spec.deployments import SERVER, ranks_of
from .harness import workload_io

MATCH = "match"
MISMATCH = "mismatch"
UNKNOWN = "unknown"
#: The module path every Modeling V2 target's class lives under.
TARGET_PACKAGE = "tensorrt_llm._torch._experimental.modeling_v2."
#: A prompt long enough to fill several cache blocks, sent again where block reuse is on.
REUSE_PROMPT = " ".join(
    f"Sack {n} of grain was weighed twice at the mill, once on arrival and once before grinding."
    for n in range(1, 25)
)
#: Declared values that leave the choice to the engine (any case): what a rank resolved
#: one to is recorded and shown, and whatever it chose is a match.
ENGINE_CHOOSES = ("default", "auto")


@dataclass(frozen=True)
class Attestation:
    """What the ranks of one server ran, against the options it was given."""

    role: str
    ranks: int
    #: Which server of the session: ``server``, ``ctx0``, ``gen1``.
    instance: str = SERVER
    #: How many option keys were compared on each rank.
    keys: int = 0
    mismatches: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()
    #: What the ranks chose where the options left the choice to the engine.
    resolved: tuple[str, ...] = ()
    #: The model class the ranks built, and the package they imported.
    target: str = ""
    origin: str = ""
    #: Where the ranks ran: their hosts, and the devices they were shown.
    hosts: tuple[str, ...] = ()
    devices: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        return MISMATCH if self.mismatches else UNKNOWN if self.unknown else MATCH

    def failure(self) -> str | None:
        """Why the check does not stand on this engine; ``None`` when declared equals observed."""
        who = "the engine" if self.instance == SERVER else f"the {self.instance} server"
        if self.mismatches:
            return (
                f"{who} did not run what its options declare: {'; '.join(self.mismatches[:6])}"
                + (f" (and {len(self.mismatches) - 6} more)" if len(self.mismatches) > 6 else "")
            )
        if self.unknown:
            return (
                f"what {who} ran is unknown, and unknown is not a pass: "
                f"{'; '.join(self.unknown[:6])}"
                + (f" (and {len(self.unknown) - 6} more)" if len(self.unknown) > 6 else "")
            )
        return None

    def record(self) -> dict[str, Any]:
        return {**asdict(self), "status": self.status}


def leaves(options: Mapping[str, Any], prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Every key of ``options`` that states a value, by its dotted path."""
    for key, value in options.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            yield from leaves(value, f"{path}.")
        else:
            yield path, value


def _same(declared: Any, observed: Any) -> bool:
    if isinstance(declared, Mapping):
        # An empty block asks for the block with its defaults; a stated one, inside a list,
        # for each of its keys.
        return isinstance(observed, Mapping) and all(
            key in observed and _same(value, observed[key]) for key, value in declared.items()
        )
    if isinstance(declared, bool) or isinstance(observed, bool) or declared is None:
        return declared is observed or declared == observed
    if isinstance(declared, (int, float)) and isinstance(observed, (int, float)):
        return math.isclose(float(declared), float(observed), rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(declared, (list, tuple)):
        return (
            isinstance(observed, (list, tuple))
            and len(declared) == len(observed)
            and all(_same(a, b) for a, b in zip(declared, observed))
        )

    # Text is compared as written: a path or a revision that differs by case is another one.
    return str(declared).removeprefix("torch.") == str(observed).removeprefix("torch.")


def _observed(resolved: Any, path: str) -> tuple[str, Any]:
    """``(status, value)`` of ``path`` in a rank's resolved configuration."""
    value = resolved
    for key in path.split("."):
        if value is None:
            # The block the key sits in is off on the engine: a different configuration.
            return MISMATCH, None
        if not isinstance(value, Mapping) or key not in value:
            return UNKNOWN, None
        value = value[key]
    return MATCH, value


def rank_records(output: str, instance: str | None = None) -> dict[int, dict[str, Any]]:
    """The record each rank of ``instance`` printed, by rank; a rank that printed twice keeps its last.

    ``instance``: only the records a session passed on from that server (records with no
    label are an aggregated deployment's, ``server``). ``None``: every record.
    """
    found: dict[int, dict[str, Any]] = {}
    for record in workload_io.parse_marked(output, workload_io.RANK_ENGINE_MARKER):
        if instance is not None and str(record.get("instance") or SERVER) != instance:
            continue
        rank = record.get("rank")
        if isinstance(rank, int) and not isinstance(rank, bool):
            found[rank] = record
    return found


def attest(
    output: str,
    options: Mapping[str, Any],
    *,
    role: str = SERVER,
    instance: str = SERVER,
    tree: str | Path | None = None,
    target: bool = True,
    devices: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
) -> Attestation:
    """Compare ``options`` with what every rank of ``instance`` in ``output`` printed.

    ``tree``, when given, is the tree being measured: a rank that imported the package
    from anywhere else ran another tree. ``target``: the model each rank built must be a
    Modeling V2 target (false for a control run on the stock path). ``devices``, when
    given, are the devices the session gave this server: a rank shown others ran elsewhere.
    """
    ranks = ranks_of(options)
    declared = list(leaves(options))
    records = rank_records(output, instance)
    mismatches: list[str] = []
    unknown: list[str] = []
    resolved: dict[str, None] = {}
    targets: set[str] = set()
    origins: set[str] = set()
    hosts: dict[str, None] = {}
    shown: dict[str, None] = {}
    root = Path(tree).resolve() if tree is not None else None
    for rank in range(ranks):
        record = records.get(rank)
        at = f"rank {rank}"
        if record is None or not isinstance(record.get("options"), Mapping):
            why = str((record or {}).get("error") or "it printed no record")
            unknown.append(f"{at}: {why}")
            continue
        for path, value in declared:
            status, seen = _observed(record["options"], path)
            if status == UNKNOWN:
                unknown.append(f"{at}: {path} is not in the engine's configuration")
            elif status == MATCH and isinstance(value, str) and value.lower() in ENGINE_CHOOSES:
                resolved[f"{path}: declared {value}, ran {seen}"] = None
            elif status == MISMATCH or not _same(value, seen):
                mismatches.append(f"{at}: {path} is {seen!r}, declared {value!r}")
        built = str(record.get("target") or "")
        targets.add(built)
        if not built:
            unknown.append(f"{at}: the model class it built")
        elif target and not built.startswith(TARGET_PACKAGE):
            mismatches.append(f"{at}: built {built}, not a Modeling V2 target")
        origin = str(record.get("origin") or "")
        origins.add(origin)
        if not origin:
            unknown.append(f"{at}: where it imported tensorrt_llm from")
        elif root is not None and root not in Path(origin).resolve().parents:
            mismatches.append(
                f"{at}: imported tensorrt_llm from {origin}, not from the tree it measures ({root})"
            )
        if record.get("host"):
            hosts[str(record["host"])] = None
        seen_devices = [str(d) for d in record.get("devices") or ()]
        shown.update(dict.fromkeys(seen_devices))
        given = devices
        if isinstance(devices, Mapping):
            # Across nodes: the devices the session gave this server on the rank's own host.
            here = short_host(record.get("host"))
            given = next((own for host, own in devices.items() if short_host(host) == here), None)
            if given is None:
                mismatches.append(
                    f"{at}: ran on {record.get('host') or 'an unnamed host'}, where the "
                    f"session gave its server no device ({', '.join(devices) or 'none'})"
                )
        if given is not None:
            if not seen_devices:
                unknown.append(f"{at}: the devices it was shown")
            elif seen_devices != [str(d) for d in given]:
                mismatches.append(
                    f"{at}: was shown devices {','.join(seen_devices)}, not the "
                    f"{','.join(str(d) for d in given)} the session gave its server"
                )
    return Attestation(
        role=role,
        ranks=ranks,
        instance=instance,
        keys=len(declared),
        mismatches=tuple(mismatches),
        unknown=tuple(unknown),
        resolved=tuple(resolved),
        target=", ".join(sorted(t for t in targets if t)),
        origin=", ".join(sorted(o for o in origins if o)),
        hosts=tuple(hosts),
        devices=tuple(shown),
    )


def short_host(host: Any) -> str:
    """A host by the name before its first dot: a node's own name and its full name are one."""
    return str(host or "").split(".")[0]


def devices_problem(records: Sequence[Mapping[str, Any]], expected: Mapping[str, int]) -> str:
    """Why the devices a session planned do not stand; ``""`` when they do.

    ``records`` are what the session said before any load: for each server, its devices
    on each host it runs on (one record per server and host). ``expected`` is each
    server's rank count. Every server has one device per rank, and no two servers share
    a device of one host.
    """
    taken: dict[tuple[str, str], str] = {}
    for name, ranks in expected.items():
        mine = [record for record in records if str(record.get("instance")) == name]
        if not mine:
            return f"{name}: the session said nothing of its devices before the load"
        own = [(short_host(r.get("host")), str(d)) for r in mine for d in r.get("devices") or ()]
        if len(own) != ranks or len(set(own)) != len(own):
            given = ",".join(device for _, device in own) or "none"
            return (
                f"{name}: {ranks} rank(s) need {ranks} device(s) of their own, and it was "
                f"given {given}"
            )
        for host, device in own:
            if (host, device) in taken:
                return (
                    f"{name} and {taken[(host, device)]} were both given device {device} "
                    f"of {host or 'one host'}: two servers never share a device"
                )
            taken[(host, device)] = name
    return ""


def reuse_sends(options: Mapping[str, Any]) -> int:
    """How often the prompt is sent: twice, and once more per rank a request may land on."""
    spread = bool(options.get("enable_attention_dp"))
    return 1 + (max(1, int(options.get("tensor_parallel_size") or 1)) if spread else 1)


def reuse_shortfall(cached: Sequence[int | None]) -> str | None:
    """Why the repeated prompt shows no reuse; ``None`` when a repeat reported cached tokens.

    ``cached`` is each send's ``cached_tokens``, in order; the first send has nothing to
    resume. A repeat that reports no count is unknown, which is not a pass.
    """
    repeats = list(cached[1:])
    counts = [c for c in repeats if isinstance(c, int) and not isinstance(c, bool)]
    if any(count > 0 for count in counts):
        return None
    if not repeats or any(count is None for count in repeats):
        return (
            "the repeated prompt came back with no `cached_tokens` count: whether the cache "
            "reused is unknown, and unknown is not a pass"
        )
    return (
        f"the engine reused nothing: one prompt sent {len(cached)} times reported "
        f"{list(cached)} cached tokens"
    )


def describe(records: Sequence[Mapping[str, Any]]) -> str:
    """The attestations of a verdict, for its summary line."""
    said = []
    for record in records:
        if "status" not in record or "keys" not in record:
            continue
        name = record.get("instance") or record.get("role")
        who = "" if name in (None, SERVER) else f"{name}: "
        chosen = "".join(f"; {line}" for line in record.get("resolved") or ())
        where = f"; devices {','.join(record['devices'])}" if record.get("devices") else ""
        said.append(
            f" [engine {who}{record['status']}: {record.get('ranks')} rank(s) x "
            f"{record.get('keys')} option(s); target {record.get('target') or 'unknown'}"
            f"{where}{chosen}]"
        )
    return "".join(said)


__all__ = [
    "ENGINE_CHOOSES",
    "MATCH",
    "MISMATCH",
    "REUSE_PROMPT",
    "TARGET_PACKAGE",
    "UNKNOWN",
    "Attestation",
    "attest",
    "describe",
    "devices_problem",
    "leaves",
    "rank_records",
    "reuse_sends",
    "reuse_shortfall",
]
