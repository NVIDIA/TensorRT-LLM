import importlib.util
import logging
import os
import re
from dataclasses import dataclass
from itertools import chain, groupby
from pathlib import Path
from typing import Optional

import pygit2


def underline(title: str, character: str = "=") -> str:
    return f"{title}\n{character * len(title)}"


def generate_title(filename: str) -> str:
    with open(filename) as f:
        # fine the first line that contains '###'
        for line in f:
            if '###' in line:
                title = line[3:].strip()
                break
        assert title is not None, f"No title found in {filename}"
    return underline(title)


@dataclass
class DocMeta:
    title: str
    order: int
    section: str
    filename: Path


def extract_meta_info(filename: str) -> Optional[DocMeta]:
    """Extract metadata from file following the pattern ### :[a-zA-Z_]+[0-9]* <value>"""
    metadata_pattern = re.compile(r'^### :([a-zA-Z_]+[0-9]*)\s+(.+)$')

    with open(filename) as f:
        metadata = DocMeta(title="",
                           order=0,
                           section="",
                           filename=Path(filename))

        for line in f:
            line = line.strip()
            match = metadata_pattern.match(line)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                setattr(metadata, key, value)
            elif not line.startswith('###'):
                continue
        if metadata.title == "":
            return None
        return metadata


# NOTE: Update here to keep consistent with the examples
LLMAPI_SECTIONS = ["Basics", "Customization", "Slurm"]


def generate_examples():
    root_dir = Path(__file__).parent.parent.parent.resolve()
    ignore_list = {
        '__init__.py', 'quickstart_example.py', 'quickstart_advanced.py',
        'quickstart_multimodal.py', 'star_attention.py'
    }
    doc_dir = root_dir / "docs/source/examples"

    def collect_script_paths(examples_subdir: str) -> list[Path]:
        """Collect Python and shell script paths from an examples subdirectory."""
        script_dir = root_dir / f"examples/{examples_subdir}"
        script_paths = list(
            chain(script_dir.glob("*.py"), script_dir.glob("*.sh")))
        return [
            path for path in sorted(script_paths)
            if path.name not in ignore_list
        ]

    # Collect source paths for LLMAPI examples
    llmapi_script_paths = collect_script_paths("llm-api")
    llmapi_doc_paths = [
        doc_dir / f"{path.stem}.rst" for path in llmapi_script_paths
    ]
    repo = pygit2.Repository('.')
    commit_hash = str(repo.head.target)
    llmapi_script_base_url = f"https://github.com/NVIDIA/TensorRT-LLM/blob/{commit_hash}/examples/llm-api"

    # Collect source paths for trtllm-serve examples
    serve_script_paths = collect_script_paths("serve")
    serve_doc_paths = [
        doc_dir / f"{path.stem}.rst" for path in serve_script_paths
    ]
    serve_script_base_url = f"https://github.com/NVIDIA/TensorRT-LLM/blob/{commit_hash}/examples/serve"

    def _get_lines_without_metadata(filename: str) -> str:
        """Get line ranges that exclude metadata lines.
        Returns a string like "5-10,15-20" for use in :lines: directive.
        """
        with open(filename) as f:
            metadata_pattern = re.compile(r'^### :([a-zA-Z_]+[0-9]*)\s+(.+)$')
            all_lines = f.readlines()

        # Find line numbers that are NOT metadata (1-indexed)
        content_lines = []
        for line_num, line in enumerate(all_lines, 1):
            line_stripped = line.strip()
            # Include line if it's not empty and not metadata
            if not metadata_pattern.match(line_stripped):
                content_lines.append(line_num)

        if not content_lines:
            return ""  # No content lines found

        # Group consecutive line numbers into ranges
        ranges = []
        start = content_lines[0]
        end = start

        for line_num in content_lines[1:]:
            if line_num == end + 1:
                # Consecutive line, extend current range
                end = line_num
            else:
                # Gap found, close current range and start new one
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = line_num
                end = line_num

        # Add the final range
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")

        return ",".join(ranges)

    # Generate the example docs for each example script
    def write_scripts(base_url: str,
                      example_script_paths: list[Path],
                      doc_paths: list[Path],
                      extra_content="") -> list[DocMeta]:
        metas = []
        for script_path, doc_path in zip(example_script_paths, doc_paths):
            if script_path.name in ignore_list:
                logging.warning(f"Ignoring file: {script_path.name}")
                continue
            script_url = f"{base_url}/{script_path.name}"
            # Determine language based on file extension
            language = "python" if script_path.suffix == ".py" else "bash"

            # Make script_path relative to doc_path and call it include_path
            include_path = '../../..' / script_path.relative_to(root_dir)

            # Extract metadata from the script file
            if meta := extract_meta_info(str(script_path)):
                title = underline(meta.title)
            else:
                logging.warning(
                    f"No metadata found for {script_path.name}, using filename as title"
                )
                title = script_path.stem.replace('_', ' ').title()
                meta = DocMeta(title=title,
                               order=0,
                               section="",
                               filename=script_path)
                title = underline(title)
            metas.append(meta)

            # Get line ranges excluding metadata
            lines_without_metadata = _get_lines_without_metadata(
                str(script_path))

            # Build literalinclude directive
            literalinclude_lines = [f".. literalinclude:: {include_path}"]
            if lines_without_metadata:
                literalinclude_lines.append(
                    f"    :lines: {lines_without_metadata}")
            literalinclude_lines.extend(
                [f"    :language: {language}", f"    :linenos:"])

            content = (f"{title}\n"
                       f"{extra_content}"
                       f"Source {script_url}.\n\n"
                       f"{chr(10).join(literalinclude_lines)}\n")
            with open(doc_path, "w+") as f:
                logging.warning(f"Writing {doc_path}")
                f.write(content)

        return metas

    def write_index(metas: list[DocMeta], doc_template_path: Path,
                    doc_path: Path, example_name: str,
                    section_order: list[str]):
        """Write the index file for the examples.

        Args:
            metas: The metadata for the examples.
            doc_template_path: The path to the template file.
            doc_path: The path to the output file.
            example_name: The name of the examples.
            section_order: The order of sections to display.

        The template file is expected to have the following placeholders:
        - %EXAMPLE_DOCS%: The documentation for the examples.
        - %EXAMPLE_NAME%: The name of the examples.
        """
        with open(doc_template_path) as f:
            template_content = f.read()

        # Sort metadata by section order and example order
        sort_key = lambda x: (section_order.index(x.section)
                              if section_order and x.section in section_order
                              else 0, int(x.order))
        metas.sort(key=sort_key)

        content = []
        for section, group in groupby(metas, key=lambda x: x.section):
            if section_order and section not in section_order:
                raise ValueError(
                    f"Section '{section}' not in section_order {section_order}")

            group_list = list(group)
            content.extend([
                section, "_" * len(section), "", ".. toctree::",
                "   :maxdepth: 2", ""
            ])

            for meta in group_list:
                content.append(f"   {meta.filename.stem}")
            content.append("")

        example_docs = "\n".join(content)

        # Replace placeholders and write to file
        output_content = template_content.replace("%EXAMPLE_DOCS%",
                                                  example_docs).replace(
                                                      "%EXAMPLE_NAME%",
                                                      example_name)
        with open(doc_path, "w") as f:
            f.write(output_content)

    # Generate the toctree for LLMAPI example scripts
    llmapi_metas = write_scripts(llmapi_script_base_url, llmapi_script_paths,
                                 llmapi_doc_paths)
    write_index(metas=llmapi_metas,
                doc_template_path=doc_dir / "llm_examples_index.template.rst_",
                doc_path=doc_dir / "llm_api_examples.rst",
                example_name="LLM Examples",
                section_order=LLMAPI_SECTIONS)

    # Generate the toctree for trtllm-serve example scripts
    serve_extra_content = (
        "Refer to the `trtllm-serve documentation "
        "<https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve.html>`_ "
        "for starting a server.\n\n")
    serve_metas = write_scripts(serve_script_base_url, serve_script_paths,
                                serve_doc_paths, serve_extra_content)
    write_index(metas=serve_metas,
                doc_template_path=doc_dir / "llm_examples_index.template.rst_",
                doc_path=doc_dir / "trtllm_serve_examples.rst",
                example_name="Online Serving Examples",
                section_order=[])


def extract_all_and_eval(file_path):
    ''' Extract the __all__ variable from a Python file.
    This is a trick to make the CI happy even the tensorrt_llm lib is not available.
    NOTE: This requires the __all__ variable to be defined at the end of the file.
    '''
    with open(file_path, 'r') as file:
        content = file.read()

    lines = content.split('\n')
    filtered_line_begin = 0

    for i, line in enumerate(lines):
        if line.startswith("__all__"):
            filtered_line_begin = i
            break

    code_to_eval = '\n'.join(lines[filtered_line_begin:])

    local_vars = {}
    exec(code_to_eval, {}, local_vars)
    return local_vars


def get_pydantic_methods() -> list[str]:
    from pydantic import BaseModel

    class Dummy(BaseModel):
        pass

    methods = set(
        [method for method in dir(Dummy) if not method.startswith('_')])
    methods.discard("__init__")
    return list(methods)


def generate_llmapi():
    root_dir = Path(__file__).parent.parent.parent.resolve()

    # Set up destination paths
    doc_dir = root_dir / "docs/source/llm-api"
    doc_dir.mkdir(exist_ok=True)
    doc_path = doc_dir / "reference.rst"

    llmapi_all_file = root_dir / "tensorrt_llm/llmapi/__init__.py"
    public_classes_names = extract_all_and_eval(llmapi_all_file)['__all__']

    content = underline("API Reference", "-") + "\n\n"
    content += ".. note::\n"
    content += "    Since version 1.0, we have attached a status label to `LLM`, `LlmArgs` and `TorchLlmArgs` Classes.\n\n"
    content += "    1. :tag:`stable` - The item is stable and will keep consistent.\n"
    content += '    2. :tag:`prototype` - The item is a prototype and is subject to change.\n'
    content += '    3. :tag:`beta` - The item is in beta and approaching stability.\n'
    content += '    4. :tag:`deprecated` - The item is deprecated and will be removed in a future release.\n'
    content += "\n"

    for cls_name in public_classes_names:
        cls_name = cls_name.strip()
        options = [
            "    :members:",
            "    :undoc-members:",
            "    :show-inheritance:",
            "    :special-members: __init__",
            "    :member-order: groupwise",
        ]

        options.append("    :inherited-members:")
        if cls_name in ["TorchLlmArgs", "TrtLlmArgs"]:
            # exclude tons of methods from Pydantic
            options.append(
                f"    :exclude-members: {','.join(get_pydantic_methods())}")

        content += f".. autoclass:: tensorrt_llm.llmapi.{cls_name}\n"
        content += "\n".join(options) + "\n\n"
    with open(doc_path, "w+") as f:
        f.write(content)


def update_version():
    version_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     "../../tensorrt_llm/version.py"))
    spec = importlib.util.spec_from_file_location("version_module",
                                                  version_path)
    version_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(version_module)
    version = version_module.__version__
    file_list = [
        "docs/source/quick-start-guide.md",
        "docs/source/commands/trtllm-serve/run-benchmark-with-trtllm-serve.md"
    ]
    for file in file_list:
        file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../" + file))
        with open(file_path, "r") as f:
            content = f.read()
        content = content.replace("x.y.z", version)
        with open(file_path, "w") as f:
            f.write(content)


if __name__ == "__main__":
    import os
    path = os.environ["TEKIT_ROOT"] + "/examples/llm-api/llm_inference.py"
    #print(extract_meta_info(path))
    generate_examples()
