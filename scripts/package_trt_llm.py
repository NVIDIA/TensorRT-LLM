import argparse
import logging
import platform
import shutil
import subprocess
from collections import namedtuple
from os import PathLike
from pathlib import Path
from typing import Iterable, Tuple


def _clean_files(src_dir: PathLike, extend_files: str) -> None:
    src_dir = Path(src_dir)
    files_to_remove = [
        ".devcontainer",
        "docker/README.md",
        "jenkins",
        "scripts/package_trt_llm.py",
        "scripts/git_replace.py",
        "tests/integration",
        "tests/unittest/trt/model/test_unet.py",
        "tests/microbenchmarks/",
        "tests/README.md",
    ] #yapf: disable

    files_to_remove.extend(extend_files)

    for file in files_to_remove:
        file_path = src_dir / file
        if file_path.is_dir():
            shutil.rmtree(file_path)
            logging.debug(f"Removed directory: {file_path}")
        else:
            file_path.unlink()
            logging.debug(f"Removed file: {file_path}")


def _check_banned_symbols(src_dir: Path, symbols: Iterable[str]) -> None:
    logging.info(f"Checking for banned symbols")

    assert any(
        map(lambda x: platform.system() == x, ("Linux", "Darwin", "Windows")))

    on_windows = platform.system() == "Windows"

    def form_command(search_string: str) -> Tuple[str]:
        if on_windows:
            # Switch exit codes so that 0 is found and 1 is not-found to match linux code-path.
            return (
                "powershell", "-Command",
                f"if (Get-ChildItem -Recurse \"{str(src_dir.absolute())}\" | Select-String \"{search_string}\")",
                "{ exit 0 }", "else", "{ exit 1 }")
        else:
            return ('grep', search_string, '-R', str(src_dir.absolute()))

    exceptions = []
    for search_string in symbols:
        command = form_command(search_string)
        command_log = " ".join(command)
        logging.debug(f"Executing {command_log}")
        keyword_found = subprocess.run(command).returncode
        if keyword_found == 0:
            exceptions.append(
                RuntimeError(
                    f"Search string {search_string} found in path {str(src_dir.absolute())}"
                ))

    if len(exceptions):
        raise Exception(exceptions)


def compress(tgt_pkg_name: Path, src_dir: Path) -> None:
    logging.info(f"Creating compressed package {tgt_pkg_name} from {src_dir}")
    # Create the tar package
    if tgt_pkg_name.suffix == ".zip":
        if platform.system() == "Windows":
            raise NotImplementedError("Windows zip path not implemented.")
        else:
            command = ("(cd", str(src_dir.parent), "&&", "zip", "-r", "-",
                       src_dir.name, ")")
            command = command + (">", str(tgt_pkg_name))
            # command = ('zip', '-r', str(tgt_pkg_name), str(src_dir))
    else:
        command = ('tar', '-C', str(src_dir.parent), '-czvf', str(tgt_pkg_name),
                   src_dir.name)
    command = " ".join(command)
    logging.debug(f"Executing {command}")
    subprocess.run(command, check=True, shell=True)


LibInfo = namedtuple(
    'LibInfo',
    ('name', 'skip_windows', 'is_static', 'path', 'cleanfiles', 'cleantrees'))

LibListConfig = namedtuple('LibListConfig', ('libs', 'cleanfiles'))
_builtin_liblist = {
    "oss": LibListConfig(
        libs=[],
        cleanfiles=[
            '.clangd',
            ".clang-tidy",
        ],
    ),
    "sourceopen": LibListConfig(
        libs=[],
        cleanfiles=[
            "LICENSE",
        ],
    ),
}


def main(
    src_dir: Path,
    liblist: LibListConfig,
    archs: Iterable[str],
    sm_arch_win: str,
    addr: str,
    commit_id: str,
    clean: bool,
    package: str,
):

    if clean:
        _clean_files(src_dir, liblist.cleanfiles)

    if package:
        git_path = src_dir / ".git"
        if git_path.exists():
            shutil.rmtree(git_path)
            logging.debug(f"Removed directory: {git_path}")
        else:
            logging.warning(f"git path not exist, ignored: {git_path}")
    if clean or package:
        _check_banned_symbols(src_dir, symbols=("__LUNOWUD", ))

    if package:
        compress(Path.cwd() / package, Path(src_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Package TRT-LLM.')
    parser.add_argument('src_dir', type=Path, help='Source directory')

    parser.add_argument('--lib_list',
                        dest='liblist',
                        required=True,
                        type=lambda x: _builtin_liblist[x],
                        help='source closed lib list name',
                        metavar="{" + ",".join(_builtin_liblist.keys()) + "}")

    parser.add_argument(
        '--arch',
        action='append',
        dest='archs',
        type=str,
        help='target architecture, can use multi times. required for download',
        choices=[
            'x86_64-windows-msvc', 'x86_64-linux-gnu', 'aarch64-linux-gnu'
        ])

    parser.add_argument(
        '--sm_arch_win',
        type=str,
        default='80-real_86-real_89-real',
        help=
        'sm architecture for windows, required for download. default: %(default)s'
    )

    parser.add_argument(
        '--addr',
        type=str,
        help='artifacts url path. %(default)s',
        default=
        'https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/LLM/main/L0_PostMerge/1379/'
    )

    parser.add_argument(
        '--download',
        type=str,
        dest='commit_id',
        help='download static lib, need specify commit_id',
    )
    parser.add_argument('--package', type=str, help='Target package name')
    parser.add_argument('--clean',
                        action=argparse.BooleanOptionalAction,
                        type=bool,
                        help='clean source file of the libs')

    parser.add_argument('-v',
                        '--verbose',
                        help="verbose",
                        action="store_const",
                        dest="loglevel",
                        const=logging.DEBUG,
                        default=logging.INFO)

    cli = parser.parse_args()
    args = vars(cli)
    print(args)  # Log on Jenkins instance.

    logging.basicConfig(level=cli.loglevel,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    args.pop('loglevel')

    main(**args)
