
# TensorRT-LLM Contribution Rules

## Issue Tracking

* All enhancement, bugfix, or change requests must begin with the creation of a [TensorRT-LLM Issue Request](https://github.com/nvidia/TensorRT-LLM/issues).
  * The issue request must be reviewed by TensorRT-LLM engineers and approved prior to code review.

## Coding Guidelines

TensorRT-LLM Coding Style can be found [in this document](CODING_GUIDELINES.md).

We use `pre-commit` for automatic code formatting and validation. Install the `pre-commit` package in your local Python environment.

```bash
pip install pre-commit
pre-commit install
```

`pre-commit` will be triggered in every commit.

```bash
git commit -m "fix"

isort....................................................................Passed
CRLF end-lines remover...................................................Passed
yapf.....................................................................Failed
- hook id: yapf
- files were modified by this hook
check for added large files..............................................Passed
check for merge conflicts................................................Passed
check for broken symlinks............................(no files to check)Skipped
detect private key.......................................................Passed
fix end of files.........................................................Passed
check yaml...............................................................Passed
trim trailing whitespace.................................................Passed
check toml...............................................................Passed
mixed line ending........................................................Passed
debug statements (python)................................................Passed
check json...........................................(no files to check)Skipped
autoflake................................................................Passed
clang-format.............................................................Passed
cmake-format.............................................................Passed
codespell................................................................Passed
ruff.....................................................................Passed
ruff-format..............................................................Passed
mdformat.................................................................Passed
```

If any files were modified by this hook, you will need to stage and commit them again.

In addition, please try to keep pull requests (PRs) as concise as possible:
* Avoid committing commented-out code.
* Wherever possible, each PR should address a single concern. If there are several otherwise-unrelated things that should be fixed to reach a desired endpoint, our recommendation is to open several PRs and indicate the dependencies in the description. The more complex the changes are in a single PR, the more time it will take to review those changes.

## Pull Requests

### Developer workflow
Developer workflow for code contributions is as follows:

1. Developers must first [fork](https://help.github.com/en/articles/fork-a-repo) the [upstream](https://github.com/nvidia/TensorRT-LLM) TensorRT-LLM OSS repository.

2. Clone the forked repository and push changes to the personal fork.

  ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_FORK.git TensorRT-LLM
    # Checkout the targeted branch and commit changes
    # Push the commits to a branch on the fork (remote).
    git push -u origin <local-branch>:<remote-branch>
  ```

3. Once the code changes are staged on the fork and ready for review, a [Pull Request](https://help.github.com/en/articles/about-pull-requests) (PR) can be [requested](https://help.github.com/en/articles/creating-a-pull-request) to merge the changes from a branch of the fork into a selected branch of upstream. PRs should typically target the `main` branch.
  * Creation of a PR creation kicks off the code review process.
  * At least one TensorRT-LLM engineer will be assigned for the review. When the PR is under review, the label `Pending Review` will be added to the PR.
  * If changes are requested, then the reviewer will add the label `Changes Requested` to the PR.
  * Once changes are approved, CI will be launched to validate the change. When CI passes, the reviewer will merge the PR.
  * If CI reports any failures, it's up to the requester to fix any CI failures before requesting another review.

### PR Submission Policies

The naming of the merge requests in TensorRT-LLM follows the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/). If the PR includes an API change that might break user code/API usage, consider adding "BREAKING CHANGE" in the title so that reviewers know what to expect. Additionally, if the PR is not related to any bug and task, consider using "chore" or None as the placeholder.

[!IMPORTANT]
For NVIDIA developers, please include the JIRA number or NVBUG ID in the PR title whenever possible. Also, to identify the author of the PR, please ensure that your GitHub account either displays your full name or your NVIDIA account name in the `Name` field of your profile.

Good PR Titles Examples:
* feat: Add support for starcoder-v2 FP8 base + FP16/BF16 LoRA
* BREAKING CHANGE: Set default max batch size to 2048
* chore: Remove version from plugins .so
* None: Stringized enums for better error msgs
* fix https://github.com/NVIDIA/TensorRT-LLM/issues/700: a Memory leak issue in C++ runtime
* [TRTLLM-5516] perf: replicate dummy request for cuda graph padding (**NVIDIAN only**)
* [nvbug/5334370] fix: Fix one model EAGLE3 (**NVIDIAN only**)


This is important for tracking and collecting what has been submitted to which release and makes it easier for others to track the bugs or tasks. It could also be helpful when collecting GitHub publish announcement.

In the PR description, please consider addressing these points:

* Background or motivation for the PR: Why is the change necessary?
* Summarize the changes in one paragraph, if possible.
* If the PR is large, explain why it cannot be broken down into multiple PRs.
* Potential performance or functional impacts of the changes. If there are risks, please inform the reviewers.
* Link to the related PRs.

[!IMPORTANT]
For NVIDIA developers,  please submit feature or bug fixes to the dedicated branch specified in the nvbug
**Keywords** field. For example, if a bug is reported on the release/v0.20 branch, please submit the fix to
`release/v0.20` instead of the main branch.

Meanwhile, please add the "release blocker" label to any PRs that could potentially cause a release delay.


## Tests and Code Review for Protected APIs

Some APIs are committed to be stable; any breaking changes to these APIs require careful design and review.

This repo contains an [API stability testsuite](./tests/api_stability) to protect committed APIs (currently including the core components of LLM API). If your PR brings breaking changes to the protected APIs, the API stability tests will fail, reporting errors like:

```txt
def test_signature(self):
        snapshot = ClassSnapshot.from_inspect(self.TEST_CLASS)
        try:
            snapshot.assert_equal(self.reference)
        except AssertionError as e:
>           raise AssertionError(self.error_msg) from e
E           AssertionError: API stability validation failed. This is probably because you changed LLM's APIs, please ask for reviews from the code owners.

tests/api_stability/test_api_stability.py:241: AssertionError
```

As the error message suggests, please ask for reviews from the code owners of the corresponding APIs.


## Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license. Signing off your commit means you accept the terms of the [Developer Certificate of Origin (DCO)](https://developercertificate.org/).

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```
Welcome to contribute to TensorRT-LLM :)
