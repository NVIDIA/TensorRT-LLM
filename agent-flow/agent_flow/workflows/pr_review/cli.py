from __future__ import annotations

import argparse
from pathlib import Path

from .prompts import PromptBundle
from .state import STATE_FILENAME
from .workflow import PrReviewWorkflow


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage cross-model PR/MR review: stage 1 Claude Code reviews "
            "and Codex addresses; stage 2 Codex reviews and Claude Code "
            "addresses. The review conversation stays local — nothing is "
            "posted to the PR/MR, committed, or pushed."
        )
    )
    parser.add_argument(
        "--target",
        help="The GitHub PR or GitLab MR to review — a number or URL. The "
        "sourcing agent detects which platform it is (from the URL or the "
        "repo's remotes) and checks it out with `gh` / `glab`; the matching "
        "authenticated CLI must be on PATH for that agent.",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
        help="Path to the local git checkout to operate in. The PR/MR branch "
        "is checked out here and the coder's edits land in its working tree.",
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Override the base branch/ref the diff is taken against. By "
        "default it is read from the PR/MR metadata.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Workspace directory for shared review files (pr_context.md, "
        "progress.yaml, discussion.md, checkpoint). Defaults to "
        "`workspace/pr-review/<id>` so each PR/MR is isolated.",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=20,
        help="Maximum reviewer/coder rounds per stage before advancing.",
    )
    parser.add_argument(
        "--reviewer-context-reset-interval",
        type=int,
        default=2,
        help="Recycle the reviewer's persistent session every N rounds. Set to 0 to disable.",
    )
    parser.add_argument(
        "--coder-context-reset-interval",
        type=int,
        default=2,
        help="Recycle the coder's persistent session every N rounds. Set to 0 to disable.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe the workspace checkpoint and managed files "
        f"({STATE_FILENAME}, pr_context.md, progress.yaml, discussion.md) and "
        "start fresh. Without this flag the workflow resumes from the "
        "checkpoint when one is present in the workspace.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, prompts: PromptBundle | None = None) -> None:
    """Run the pr-review workflow as a CLI."""
    args = _parse_args(argv)
    with PrReviewWorkflow(
        repo=args.repo,
        target=args.target,
        base=args.base,
        workspace=args.workspace,
        num_rounds=args.num_rounds,
        reviewer_context_reset_interval=args.reviewer_context_reset_interval,
        coder_context_reset_interval=args.coder_context_reset_interval,
        clean=args.clean,
        prompts=prompts,
    ) as workflow:
        workflow.run()


if __name__ == "__main__":
    main()
