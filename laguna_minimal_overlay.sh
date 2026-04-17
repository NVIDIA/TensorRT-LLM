#!/usr/bin/env bash
# Minimal transformers-4.57-compat overlay for a Laguna checkpoint directory.
#
# Laguna's `configuration_laguna.py` imports two transformers >= 4.58 symbols:
#   - `transformers.configuration_utils.PreTrainedConfig`
#   - `transformers.modeling_rope_utils.RopeParameters`
# Neither exists in transformers 4.57, which is what we have installed.
#
# This overlay symlinks every file in the checkpoint dir, then replaces just
# `configuration_laguna.py` with a copy that is prefixed with a compat shim
# aliasing the missing symbols. No other files are modified — additional
# fixes (config.json rewrites, etc.) are deliberately out of scope.
#
# Usage:
#   laguna_minimal_overlay.sh <src_checkpoint_dir> [overlay_dir]
# Echoes the overlay directory path on stdout.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $(basename "$0") <src_checkpoint_dir> [overlay_dir]" >&2
    exit 1
fi

SRC="$(realpath "$1")"
OVERLAY="${2:-/tmp/laguna-minimal-$(basename "$SRC")}"

if [[ ! -d "$SRC" ]]; then
    echo "ERROR: source dir does not exist: $SRC" >&2
    exit 1
fi
if [[ ! -f "$SRC/configuration_laguna.py" ]]; then
    echo "ERROR: $SRC has no configuration_laguna.py" >&2
    exit 1
fi

rm -rf "$OVERLAY"
mkdir -p "$OVERLAY"

for entry in "$SRC"/*; do
    name="$(basename "$entry")"
    if [[ "$name" == "configuration_laguna.py" ]]; then
        continue
    fi
    ln -s "$entry" "$OVERLAY/$name"
done

SHIM_HEADER='# --- compat shim (added by laguna_minimal_overlay.sh) ---
try:
    from transformers.configuration_utils import PreTrainedConfig  # transformers >= 4.58
except ImportError:
    from transformers.configuration_utils import PretrainedConfig as PreTrainedConfig  # transformers 4.57
try:
    from transformers.modeling_rope_utils import RopeParameters  # transformers >= 4.58
except ImportError:
    RopeParameters = dict  # type: ignore[misc,assignment]
# Neuter `@strict` (huggingface_hub) — newer Laguna configs decorate a non-dataclass
# subclass of PretrainedConfig with it, which raises in our hf-hub version.
def strict(cls=None, **_kwargs):  # type: ignore[no-redef]
    return cls if cls is not None else (lambda c: c)
# Neuter `@auto_docstring` (transformers utils) — only adds docstrings, safe to skip.
def auto_docstring(*_args, **_kwargs):  # type: ignore[no-redef]
    return (lambda c: c) if not _args or not callable(_args[0]) else _args[0]
# --- end compat shim ---
'

# Strip the original imports we just shimmed; everything else is kept verbatim.
{
    printf '%s' "$SHIM_HEADER"
    grep -vE '^from transformers\.configuration_utils import PreTrainedConfig$|^from transformers\.modeling_rope_utils import RopeParameters$|^from huggingface_hub\.dataclasses import strict$|^from transformers\.utils import auto_docstring$' \
        "$SRC/configuration_laguna.py"
} > "$OVERLAY/configuration_laguna.py"

echo "$OVERLAY"
