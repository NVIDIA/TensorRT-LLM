"""Config for every ops tool: one shared file plus a per-project overlay.

Format is TOML, read with the standard library (``tomllib``), so the ops
package adds no dependency.

The split exists because the two halves have different owners:

* **shared, per machine** — allocations, worktree slots, the container image
  and mounts, the dispatch spool root. Several projects run on one machine and
  must see ONE allocation table; a per-project copy would drift, and two
  agents would each think they held the same allocation.
* **per project** — the project root, workspace, log dir, roles and their
  checkouts, notice channels, dashboard options.

Resolution, in order, with the project overlay winning section by section:

1. shared: ``--shared-config``, else ``$AGENT_FLOW_OPS_SHARED``, else
   ``$XDG_CONFIG_HOME/agent-flow/ops.toml`` (default ``~/.config``),
2. project: ``--config``, else ``$AGENT_FLOW_OPS_CONFIG``, else
   ``$AGENT_FLOW_OPS_PROJECT`` (a toml file, or a directory holding
   ``agent-flow-ops.toml``), else ``./agent-flow-ops.toml``.

A single file that contains everything still works: the shared half is
optional, and so is the project half as long as the other one is complete.
Nothing has a built-in site default — a tool that finds no config at all
refuses to run and prints where it looked. Guessing a project directory is how
a tool ends up writing into the wrong project.

The per-project section is ``[project]``. ``[run]`` is the old name and is
still accepted, with a warning.
"""

from __future__ import annotations

import os
import tomllib
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ENV_VAR = "AGENT_FLOW_OPS_CONFIG"
PROJECT_ENV_VAR = "AGENT_FLOW_OPS_PROJECT"
SHARED_ENV_VAR = "AGENT_FLOW_OPS_SHARED"
CWD_NAME = "agent-flow-ops.toml"
USER_REL = "agent-flow/ops.toml"
EXAMPLE = "agent-flow-ops.example.toml"
SHARED_EXAMPLE = "agent-flow-ops.shared.example.toml"
PROJECT_SECTION = "project"
LEGACY_PROJECT_SECTION = "run"
SHARED_SECTIONS = ("allocations", "worktrees", "container", "dispatch")


class OpsConfigError(RuntimeError):
    """Config missing, unreadable, or missing a key a tool needs."""


def _user_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / USER_REL


def _as_config_file(value: str | os.PathLike[str]) -> Path:
    """A project may be named by its toml or by the directory holding it."""
    p = Path(value)
    return p / CWD_NAME if p.is_dir() else p


def candidate_paths(explicit: str | os.PathLike[str] | None = None) -> list[Path]:
    """Every PROJECT config path ``load_config`` would try, in order."""
    if explicit:
        return [_as_config_file(explicit)]
    out = []
    if os.environ.get(ENV_VAR):
        out.append(_as_config_file(os.environ[ENV_VAR]))
    if os.environ.get(PROJECT_ENV_VAR):
        out.append(_as_config_file(os.environ[PROJECT_ENV_VAR]))
    out.append(Path.cwd() / CWD_NAME)
    return out


def shared_candidate_paths(explicit: str | os.PathLike[str] | None = None) -> list[Path]:
    """Every SHARED config path ``load_config`` would try, in order."""
    if explicit:
        return [Path(explicit)]
    out = []
    if os.environ.get(SHARED_ENV_VAR):
        out.append(Path(os.environ[SHARED_ENV_VAR]))
    out.append(_user_config_path())
    return out


def merge(shared: dict, project: dict) -> dict:
    """Overlay the project config on the shared one, one level deep.

    A section present in both is merged key by key (so a project can override
    a single container mount without restating the image); a section present
    in only one is taken whole.
    """
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in shared.items()}
    for section, body in project.items():
        if isinstance(body, dict) and isinstance(out.get(section), dict):
            merged = dict(out[section])
            for k, v in body.items():
                if isinstance(v, dict) and isinstance(merged.get(k), dict):
                    merged[k] = {**merged[k], **v}
                else:
                    merged[k] = v
            out[section] = merged
        else:
            out[section] = body
    return out


@dataclass(frozen=True)
class Allocation:
    """One shared machine allocation (a SLURM job, a reserved host, ...)."""

    key: str
    job_id: str = ""
    job_name: str = ""
    description: str = ""
    container: str = ""
    aliases: tuple[str, ...] = ()


@dataclass
class OpsConfig:
    """Parsed ops config.

    Attribute access for the keys tools need, plus ``raw`` for anything a tool
    wants to read directly.
    """

    path: Path
    raw: dict[str, Any] = field(default_factory=dict)
    shared_path: Path | None = None
    project_path: Path | None = None

    # -- project layout ------------------------------------------------------
    @property
    def project_name(self) -> str:
        return str(self.get(PROJECT_SECTION, "name", default="") or self.run_root.name)

    @property
    def project_root(self) -> Path:
        """Directory holding one project's state: logs, workspace, tables."""
        return Path(self._req(PROJECT_SECTION, "root")).expanduser()

    #: Kept so existing callers and scripts keep working.
    run_root = project_root

    @property
    def workspace(self) -> Path:
        """Agent-visible workspace (task/plan/progress files)."""
        rel = self.get(PROJECT_SECTION, "workspace", default="workspace")
        p = Path(rel).expanduser()
        return p if p.is_absolute() else self.project_root / p

    @property
    def log_dir(self) -> Path:
        rel = self.get(PROJECT_SECTION, "log_dir", default="logs")
        p = Path(rel).expanduser()
        return p if p.is_absolute() else self.project_root / p

    @property
    def archive_root(self) -> Path | None:
        v = self.get(PROJECT_SECTION, "archive_root", default=None)
        return Path(v).expanduser() if v else None

    @property
    def projects_root(self) -> Path | None:
        """Directory that holds every project dir (for ``ops.project list``)."""
        v = self.get("projects", "root", default=None)
        return Path(v).expanduser() if v else None

    # -- allocations ---------------------------------------------------------
    @property
    def allocations(self) -> dict[str, Allocation]:
        out: dict[str, Allocation] = {}
        for key, body in (self.raw.get("allocations") or {}).items():
            body = body or {}
            out[key] = Allocation(
                key=key,
                job_id=str(body.get("job_id", "")),
                job_name=str(body.get("job_name", "")),
                description=str(body.get("description", "")),
                container=str(body.get("container", self.get("container", "name", default=""))),
                aliases=tuple(body.get("aliases", ()) or ()),
            )
        return out

    @property
    def alloc_aliases(self) -> dict[str, str]:
        """Alias -> canonical allocation key."""
        return {a: alloc.key for alloc in self.allocations.values() for a in alloc.aliases}

    # -- container -----------------------------------------------------------
    @property
    def default_allocation(self) -> str:
        """Allocation the container tools use when none is named."""
        declared = list(self.allocations)
        return str(
            self.get("container", "default_allocation", default=declared[0] if declared else "")
        )

    @property
    def container_name(self) -> str:
        return str(self._req("container", "name"))

    @property
    def container_image(self) -> str:
        return str(self.get("container", "image", default=""))

    @property
    def container_mounts(self) -> list[str]:
        return [str(m) for m in self.get("container", "mounts", default=[])]

    @property
    def repo(self) -> Path:
        """Checkout whose code the in-container commands import."""
        return Path(self._req("container", "repo")).expanduser()

    @property
    def container_env(self) -> dict[str, str]:
        return {str(k): str(v) for k, v in (self.get("container", "env", default={}) or {}).items()}

    @property
    def env_prologue(self) -> list[str]:
        """Extra shell lines run inside the container before the command."""
        return [str(line) for line in self.get("container", "env_prologue", default=[])]

    # -- roles ---------------------------------------------------------------
    @property
    def role_checkouts(self) -> dict[str, Path]:
        """Role name -> the checkout that role works from.

        Replaces inferring the role from hard-coded path fragments: a process
        whose cwd is under one of these paths is that role.
        """
        return {
            str(role): Path(p).expanduser()
            for role, p in (self.get("roles", "checkouts", default={}) or {}).items()
        }

    @property
    def roles(self) -> tuple[str, ...]:
        declared = self.get("roles", "names", default=None)
        if declared:
            return tuple(str(r) for r in declared)
        return tuple(self.role_checkouts) or ("coder", "reviewer")

    # -- worktrees -----------------------------------------------------------
    @property
    def worktree_dir(self) -> Path | None:
        v = self.get("worktrees", "dir", default=None)
        return Path(v).expanduser() if v else None

    @property
    def worktree_slots(self) -> list[str]:
        return [str(s) for s in self.get("worktrees", "slots", default=[])]

    # -- dashboard -----------------------------------------------------------
    @property
    def dashboard(self) -> dict[str, Any]:
        return dict(self.raw.get("dashboard") or {})

    # -- generic access ------------------------------------------------------
    def get(self, section: str, key: str, default: Any = None) -> Any:
        return (self.raw.get(section) or {}).get(key, default)

    def _req(self, section: str, key: str) -> Any:
        try:
            return (self.raw[section])[key]
        except KeyError:
            raise OpsConfigError(
                f"{self.path}: missing required key [{section}].{key}. "
                f"See {EXAMPLE} for the full schema."
            ) from None


def _read_toml(path: Path) -> dict:
    try:
        return tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise OpsConfigError(f"{path}: cannot read ops config: {exc}") from None


def _first_existing(paths: list[Path]) -> Path | None:
    return next((p for p in paths if p.is_file()), None)


def _rename_legacy_section(raw: dict, path: Path) -> dict:
    """Accept the old ``[run]`` name for ``[project]``, once, with a warning."""
    if LEGACY_PROJECT_SECTION not in raw:
        return raw
    warnings.warn(
        f"{path}: [{LEGACY_PROJECT_SECTION}] is the old name for [{PROJECT_SECTION}]; rename it.",
        DeprecationWarning,
        stacklevel=3,
    )
    raw = dict(raw)
    legacy = raw.pop(LEGACY_PROJECT_SECTION)
    raw[PROJECT_SECTION] = {**legacy, **(raw.get(PROJECT_SECTION) or {})}
    return raw


def load_config(
    explicit: str | os.PathLike[str] | None = None,
    shared: str | os.PathLike[str] | None = None,
) -> OpsConfig:
    """Load shared + project config.

    Raises ``OpsConfigError`` naming every path tried when neither exists. The
    project overlay wins where the two overlap.
    """
    project_tried = candidate_paths(explicit)
    shared_tried = shared_candidate_paths(shared)
    project_path = _first_existing(project_tried)
    shared_path = _first_existing(shared_tried)
    if project_path is None and shared_path is None:
        listed = "\n  ".join(str(p) for p in shared_tried + project_tried)
        raise OpsConfigError(
            "no agent-flow ops config found. Looked at:\n  "
            + listed
            + f"\n\nCopy {SHARED_EXAMPLE} to the shared path and {EXAMPLE} into the "
            f"project, or pass --config <path> / set {ENV_VAR} or {PROJECT_ENV_VAR}."
        )
    shared_raw = _rename_legacy_section(_read_toml(shared_path), shared_path) if shared_path else {}
    project_raw = (
        _rename_legacy_section(_read_toml(project_path), project_path) if project_path else {}
    )
    return OpsConfig(
        path=(project_path or shared_path).resolve(),
        raw=merge(shared_raw, project_raw),
        shared_path=shared_path.resolve() if shared_path else None,
        project_path=project_path.resolve() if project_path else None,
    )


def add_config_argument(parser) -> None:
    """Add the standard ``--config`` flag to an argparse parser."""
    parser.add_argument(
        "--config",
        default=None,
        help=(
            f"project ops config, or the project dir holding {CWD_NAME} "
            f"(default: ${ENV_VAR}, ${PROJECT_ENV_VAR}, then ./{CWD_NAME})"
        ),
    )
    parser.add_argument(
        "--shared-config",
        default=None,
        help=f"shared machine config (default: ${SHARED_ENV_VAR}, then ~/.config/{USER_REL})",
    )


def config_from_args(args) -> OpsConfig:
    """Load the config named by a parsed ``--config``; exit 2 with the reason."""
    import sys

    try:
        return load_config(getattr(args, "config", None), getattr(args, "shared_config", None))
    except OpsConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
