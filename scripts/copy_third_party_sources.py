"""Copy third-party sources used in the cmake build to a container directory.

The purpose of this script is to simplify the process of producing third party
sources "as used" in the build. We package up all of the sources we use and
stash them in a location in the container so that they are automatically
distributed alongside the build artifacts ensuring that we comply with the
license requirements in an obvious and transparent manner.
"""

import argparse
import logging
import pathlib
import subprocess

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deps-dir",
        type=pathlib.Path,
        required=True,
        help="Path to the third party dependencies directory, e.g. ${CMAKE_BINARY_DIR}/_deps",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Path to the output directory where third party sources will be copied",
    )

    args = parser.parse_args()

    src_dirs = list(sorted(args.deps_dir.glob("*-src")))
    if not src_dirs:
        raise ValueError(f"No source directories found in {args.deps_dir}")

    for src_dir in src_dirs:
        tarball_name = src_dir.name[:-4] + ".tar.gz"
        output_path = args.output_dir / tarball_name
        logger.info(f"Creating tarball {output_path} from {src_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["tar", "-czf", str(output_path), "-C", str(src_dir.parent), src_dir.name],
            check=True,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
