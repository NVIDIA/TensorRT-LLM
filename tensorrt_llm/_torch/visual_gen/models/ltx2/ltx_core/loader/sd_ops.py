"""Deterministic state-dict transforms used for projection and loading.
An ordered chain starts with checkpoint keys and produces the model keys used
to bind weights. Between stages, keys name intermediate transformed tensors.
The chain runs on meta tensors to project shapes and dtypes before loading,
then on checkpoint tensors to materialize those same outputs.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Protocol

import torch

ModelKeyGroup = str | tuple[str, ...]  # One key, or a primary key grouped with related keys.


@dataclass(frozen=True, slots=True)
class KeyReplacement:
    """Replace every occurrence of a substring in the current stage's key.
    ``content`` is the substring to find; ``replacement`` is its substitute.
    """

    content: str
    replacement: str


@dataclass(frozen=True, slots=True)
class KeyMatching:
    """Match the current stage's key against three substring conditions.
    ``prefix`` and ``suffix`` must occur at the corresponding ends of the key.
    ``contains`` must occur anywhere in it. An empty field imposes no condition;
    all nonempty conditions must hold for a match.
    """

    prefix: str = ""
    suffix: str = ""
    contains: str = ""


def _matches(key: str, matcher: KeyMatching) -> bool:
    """Return whether a stage key satisfies every condition in ``matcher``."""
    return (
        key.startswith(matcher.prefix)
        and key.endswith(matcher.suffix)
        and (not matcher.contains or matcher.contains in key)
    )


def _replace(key: str, replacement: KeyReplacement) -> str:
    """Apply one substring replacement to the current stage's key."""
    return key.replace(replacement.content, replacement.replacement)


class _KeyValueOperationFn(Protocol):
    """Callable interface shared by concrete and abstract value transforms.
    Functions can implement this interface directly. Callable classes should
    inherit it explicitly, as ``KeyValueOperation`` does.
    """

    __slots__ = ()

    def __call__(self, primary_key: str, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Transform a primary tensor and its auxiliaries into keyed outputs.
        ``primary_key`` names one entry in ``inputs``. Other entries are the
        present auxiliary tensors declared for that primary. Returned keys
        name this stage's outputs, which may become another stage's inputs.
        """
        ...


@dataclass(frozen=True, slots=True)
class KeyValueOperation(_KeyValueOperationFn):
    """A tensor transform and its optional implementation for meta tensors.
    ``operation`` computes output tensors from loaded primary and auxiliary
    tensors. ``abstract_operation`` constructs output meta tensors with the
    same keys, shapes, and dtypes without executing a device kernel. Both take
    the primary key and a mapping of inputs and return a dictionary of
    outputs. When ``abstract_operation`` is absent, projection runs
    ``operation`` on meta tensors; a transform that cannot do that must provide
    the abstract implementation.
    ``auxiliary_key_specs`` declares optional additional inputs. A string names
    an exact stage key. A ``(KeyMatching, KeyReplacement)`` pair matches the
    primary key and derives an auxiliary key by replacement. Candidates absent
    from the stage's inputs are omitted. For example, a weight transform can
    request its sibling scale and emit one dequantized weight.
    ``preserve_output_device`` asks materialization to keep each concrete
    result on the device chosen by the callable. The callable owns any input
    transfers. Direct calls return the callable's results unchanged.
    """

    operation: _KeyValueOperationFn
    abstract_operation: _KeyValueOperationFn | None = None
    auxiliary_key_specs: tuple[str | tuple[KeyMatching, KeyReplacement], ...] = ()
    preserve_output_device: bool = False

    def resolve_auxiliary_keys(self, primary_key: str) -> tuple[str, ...]:
        """Derive candidate auxiliary keys in declaration order, before presence checks."""
        return tuple(
            entry if isinstance(entry, str) else _replace(primary_key, entry[1])
            for entry in self.auxiliary_key_specs
            if isinstance(entry, str) or _matches(primary_key, entry[0])
        )

    def __call__(self, primary_key: str, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the concrete transform on the supplied primary and auxiliary tensors."""
        return self.operation(primary_key, inputs)

    def apply_abstract(self, primary_key: str, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return output meta tensors defining this stage's keys, shapes, and dtypes.
        The caller supplies meta inputs. The explicit abstract implementation,
        or the concrete fallback, runs with ``meta`` as the default device.
        Returning a real tensor raises ``ValueError``. Transform exceptions
        propagate, so an unsupported meta fallback is visible to the author.
        """
        operation = self.abstract_operation or self.operation
        with torch.device("meta"):
            results = operation(primary_key, inputs)

        non_meta = [key for key, value in results.items() if not value.is_meta]
        if non_meta:
            raise ValueError(f"abstract operation emitted non-meta tensors for keys: {non_meta}")
        return results


@dataclass(frozen=True, slots=True)
class SDKeyValueOperation:
    """Associate a primary-key matcher with a tensor transform.
    ``key_matcher`` selects primary keys within this stage's inputs.
    ``kv_operation`` consumes each selected primary and its auxiliary inputs.
    """

    key_matcher: KeyMatching
    kv_operation: KeyValueOperation


@dataclass(frozen=True, slots=True)
class SDOps:
    """Immutable state-dict operations identified completely by ``name``.
    Caches and distributed plans do not inspect operations, closures, captured
    tensors, or other parameters. Authors must therefore use a distinct, stable
    name whenever two instances can produce different results; reusing a name
    declares the instances equivalent for identity purposes.
    ``mapping`` contains key matchers, substring replacements, and matched
    value transforms. A key is kept when any key matcher matches it, or always
    when there are no key matchers. Replacements run in declaration order.
    At most one value transform may match a primary key; chain separate
    ``SDOps`` instances to apply successive value transforms.
    ``allowed_keys`` optionally restricts the names after key replacement. A
    separate final selection stage can restrict model keys while earlier
    stages still read required companion tensors.
    ``excluded_prefixes`` drops named destination families after replacement.
    Use a distinct stage name when the excluded families differ.
    """

    name: str
    mapping: tuple[KeyReplacement | KeyMatching | SDKeyValueOperation, ...] = ()
    allowed_keys: frozenset[str] | None = None
    excluded_prefixes: tuple[str, ...] = ()

    def with_replacement(self, content: str, replacement: str) -> "SDOps":
        """Return a copy with one more substring replacement, preserving its name."""

        new_mapping = (*self.mapping, KeyReplacement(content, replacement))
        return replace(self, mapping=new_mapping)

    def with_matching(self, prefix: str = "", suffix: str = "", contains: str = "") -> "SDOps":
        """Return a copy with another alternative key matcher, preserving its name."""

        new_mapping = (*self.mapping, KeyMatching(prefix, suffix, contains))
        return replace(self, mapping=new_mapping)

    def with_kv_operation(
        self,
        operation: KeyValueOperation | _KeyValueOperationFn,
        key_prefix: str = "",
        key_suffix: str = "",
    ) -> "SDOps":
        """Return a copy matching a value transform by primary-key prefix and suffix.
        A plain callable is wrapped as a ``KeyValueOperation`` with meta fallback.
        The copy preserves the operation name; authors must choose a new name
        when distinguishing different transforms for cache identity.
        """
        key_matcher = KeyMatching(key_prefix, key_suffix)
        nested = operation if isinstance(operation, KeyValueOperation) else KeyValueOperation(operation)
        sd_kv_operation = SDKeyValueOperation(key_matcher, nested)
        new_mapping = (*self.mapping, sd_kv_operation)
        return replace(self, mapping=new_mapping)

    def apply_to_key(self, key: str) -> str | None:
        """Filter a stage input key, rewrite it, and apply the exact-name restriction.
        Return the rewritten key, or ``None`` when it is excluded. This method
        applies no value transforms.
        """
        matchers = [content for content in self.mapping if isinstance(content, KeyMatching)]
        valid = any(_matches(key, matcher) for matcher in matchers)
        if matchers and not valid:
            return None

        for replacement in self.mapping:
            if not isinstance(replacement, KeyReplacement):
                continue
            key = _replace(key, replacement)

        if self.allowed_keys is not None and key not in self.allowed_keys:
            return None
        if key.startswith(self.excluded_prefixes):
            return None

        return key

    def grouped(self, keys: Collection[str]) -> list[ModelKeyGroup]:
        """Return stage keys in order, grouping only value-operation inputs.
        For example, separate weight and bias primaries may both read one gate.
        The gate appears in both input tuples and is not passed through alone.
        An auxiliary key cannot also be an operation primary in this stage.
        A string is a pass-through key; a tuple contains a matching primary
        followed by its present auxiliaries, even when only the primary is
        present. The list retains input-key order, while auxiliaries within a
        tuple follow their operation's declared order. Keys must already be
        filtered and renamed; this method does not apply key replacements.
        """
        operations = {key: self._value_operation(key) for key in keys}
        consumed: set[str] = set()
        groups: dict[str, tuple[str, ...]] = {}
        for key, operation in operations.items():
            if operation is None:
                continue
            extras = tuple(
                dict.fromkeys(extra for extra in operation.resolve_auxiliary_keys(key) if extra in operations)
            )
            for extra in extras:
                if extra == key:
                    raise ValueError(f"{self.name}: {key!r} names itself as an auxiliary input")
                if operations[extra] is not None:
                    raise ValueError(f"{self.name}: {extra!r} is taken by {key!r} but also matched by an operation")
            groups[key] = (key, *extras)
            consumed.update(extras)
        return [groups.get(key, key) for key in operations if key not in consumed]

    def apply_to_key_value(self, primary_key: str, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return concrete outputs for a filtered, renamed primary and its inputs.
        Without a matching transform, return the primary tensor unchanged.
        Key filtering and replacement are performed separately by the caller.
        """
        operation = self._value_operation(primary_key)
        return {primary_key: inputs[primary_key]} if operation is None else operation(primary_key, inputs)

    def apply_to_key_value_abstract(
        self, primary_key: str, inputs: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Return output meta tensors for a filtered, renamed primary and its inputs.
        Without a matching transform, return the primary meta tensor unchanged.
        The caller performs key filtering and replacement before this method.
        """
        operation = self._value_operation(primary_key)
        if operation is None:
            return {primary_key: inputs[primary_key]}
        return operation.apply_abstract(primary_key, inputs)

    def _value_operation(self, key: str) -> KeyValueOperation | None:
        """Find the primary key's value transform, rejecting ambiguous matches."""
        matched = [
            operation
            for operation in self.mapping
            if isinstance(operation, SDKeyValueOperation) and _matches(key, operation.key_matcher)
        ]
        if len(matched) > 1:
            raise ValueError(f"{self.name}: {key!r} is matched by {len(matched)} value operations")
        return matched[0].kv_operation if matched else None


def select_destinations(name: str, keys: Collection[str]) -> SDOps:
    """Create a final stage selecting exact model keys produced by earlier stages.
    ``name`` is an identity prefix. The selected keys determine a stable digest
    appended to it, independent of their supplied order.
    """
    selected = frozenset(keys)
    digest = sha256("\0".join(sorted(selected)).encode()).hexdigest()[:12]
    return SDOps(f"{name}:{digest}", allowed_keys=selected)


# Predefined SDOps instances
LTXV_LORA_COMFY_RENAMING_MAP = (
    SDOps("LTXV_LORA_COMFY_PREFIX_MAP").with_matching().with_replacement("diffusion_model.", "")
)

LTXV_LORA_COMFY_TARGET_MAP = (
    SDOps("LTXV_LORA_COMFY_TARGET_MAP")
    .with_matching()
    .with_replacement("diffusion_model.", "")
    .with_replacement(".lora_A.weight", ".weight")
    .with_replacement(".lora_B.weight", ".weight")
)
