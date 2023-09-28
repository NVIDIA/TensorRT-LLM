# CONTRIBUTING

## Workflow

1. Apply to be a developer of the project

Please reach to the maintainers:

- [bhsueh](https://gitlab-master.nvidia.com/bhsueh) (BoYang Hsueh)
- [tali](https://gitlab-master.nvidia.com/tali) (Tao Li)
- [junq](https://gitlab-master.nvidia.com/junq) (Freddy Qi)

2. Clone

```bash
git clone https://gitlab-master.nvidia.com/ftp/tensorrt_llm.git
```

3. Create a local feature branch

```bash
cd tensorrt_llm
git checkout -b feature/xxx
```

4. Commit

We use `pre-commit` to help check the code style.

```bash
pip install pre-commit
pre-commit install
```

`pre-commit` will be triggered in every commit.

```bash
git commit -m "fix"

isort....................................................................Passed
CRLF end-lines remover...............................(no files to check)Skipped
yapf.....................................................................Failed
- hook id: yapf
- files were modified by this hook
check for added large files..............................................Passed
check for merge conflicts................................................Passed
check for broken symlinks............................(no files to check)Skipped
detect private key...................................(no files to check)Skipped
fix end of files.........................................................Passed
check yaml...........................................(no files to check)Skipped
trim trailing whitespace.................................................Passed
autoflake................................................................Passed
clang-format.........................................(no files to check)Skipped
cmake-format.........................................(no files to check)Skipped
```

5. Push

```bash
git push -u origin feature/xxx
```

## Add a new model

[How to add a new model](docs/2023-05-17-how-to-add-a-new-model.md)

## Debug

[How to debug](docs/2023-05-19-how-to-debug.md)
