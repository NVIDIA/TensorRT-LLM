import logging
import re
from dataclasses import dataclass
from itertools import chain, groupby
from pathlib import Path
from pprint import pprint
from typing import Optional


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
    ignore_list = {'__init__.py', 'quickstart_example.py'}
    doc_dir = root_dir / "docs/source/examples"

    # Collect source paths for LLMAPI examples
    llmapi_script_dir = root_dir / "examples/llm-api"
    llmapi_script_paths = list(llmapi_script_dir.glob("*.py"))
    llmapi_script_paths += list(llmapi_script_dir.glob("*.sh"))

    llmapi_script_paths = [
        i for i in llmapi_script_paths if i.name not in ignore_list
    ]

    # Determine destination .rst paths for LLMAPI examples
    llmapi_doc_paths = [
        doc_dir / f"{path.stem}.rst" for path in llmapi_script_paths
    ]
    llmapi_script_base_url = "https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llm-api"

    # Collect source paths for trtllm-serve examples
    serve_script_dir = root_dir / "examples/serve"
    serve_script_paths = sorted(
        chain(serve_script_dir.glob("*.py"), serve_script_dir.glob("*.sh")))
    serve_script_paths = [
        i for i in serve_script_paths if i.name not in ignore_list
    ]
    serve_doc_paths = [
        doc_dir / f"{path.stem}.rst" for path in serve_script_paths
    ]
    pprint(serve_script_paths)
    serve_script_base_url = "https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/serve"

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

            # For Python files, use generate_title to extract title from comments
            # For shell scripts, use filename as title
            if meta := extract_meta_info(script_path):
                title = meta.title
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

            content = (f"{title}\n"
                       f"{'='  * len(title)}\n\n"
                       f"{extra_content}"
                       f"Source {script_url}.\n\n"
                       f".. literalinclude:: {include_path}\n"
                       f"    :language: {language}\n"
                       "    :linenos:\n")
            with open(doc_path, "w+") as f:
                logging.warning(f"Writing {doc_path}")
                f.write(content)

        return metas

    def write_index(metas: list[DocMeta], doc_template_path: Path,
                    doc_path: Path, example_name: str,
                    section_order: list[str]):
        '''
        Write the index file for the examples.

        Args:
            metas: The metadata for the examples.
            doc_template_path: The path to the template file.
            doc_path: The path to the output file.
            example_name: The name of the examples.

        The template file is expected to have the following placeholders:
        - %EXAMPLE_DOCS%: The documentation for the examples.
        - %EXAMPLE_NAME%: The name of the examples.
        '''
        with open(doc_template_path) as f:
            examples_index = f.read()

        metas.sort(key=lambda x: (section_order.index(x.section)
                                  if section_order else 0, int(x.order)))
        pprint(metas)

        content = []
        for section, group in groupby(metas, key=lambda x: x.section):
            if section_order:
                assert section in section_order, f"Section {section} not in {section_order}, please add it with proper order"
            group = list(group)
            content.append(section)
            content.append("_" * len(section))
            content.append("")
            # settings
            content.append('.. toctree::')
            content.append('   :maxdepth: 2')
            content.append('')

            for meta in group:
                content.append(f'   {meta.filename.stem}')

            content.append('')

        example_docs = "\n".join(content)
        with open(doc_path, "w+") as f:
            f.write(examples_index.replace(r"%EXAMPLE_DOCS%", example_docs)\
                    .replace(r"%EXAMPLE_NAME%", example_name))

    # Generate the toctree for LLMAPI example scripts
    metas = write_scripts(llmapi_script_base_url, llmapi_script_paths,
                          llmapi_doc_paths)
    write_index(metas=metas,
                doc_template_path=doc_dir / "llm_examples_index.template.rst_",
                doc_path=doc_dir / "llm_api_examples.rst",
                example_name="LLM Examples",
                section_order=LLMAPI_SECTIONS)
    # Generate the toctree for trtllm-serve example scripts
    trtllm_serve_content = "Refer to the `trtllm-serve documentation <https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve.html>`_ for starting a server.\n\n"
    metas = write_scripts(serve_script_base_url, serve_script_paths,
                          serve_doc_paths, trtllm_serve_content)
    write_index(metas=metas,
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


def generate_llmapi():
    root_dir = Path(__file__).parent.parent.parent.resolve()

    # Destination paths
    doc_dir = root_dir / "docs/source/llm-api"
    doc_dir.mkdir(exist_ok=True)
    doc_path = doc_dir / "reference.rst"

    llmapi_all_file = root_dir / "tensorrt_llm/llmapi/__init__.py"
    public_classes_names = extract_all_and_eval(llmapi_all_file)['__all__']

    content = underline("API Reference", "-") + "\n\n"
    for cls_name in public_classes_names:
        cls_name = cls_name.strip()
        options = [
            "    :members:", "    :undoc-members:", "    :show-inheritance:"
        ]

        if cls_name != 'LLM':  # Conditionally add :special-members: __init__
            options.append("    :special-members: __init__")

        if cls_name in ['TrtLLM', 'TorchLLM', 'LLM']:
            options.append("    :inherited-members:")

        content += f".. autoclass:: tensorrt_llm.llmapi.{cls_name}\n"
        content += "\n".join(options) + "\n\n"

    with open(doc_path, "w+") as f:
        f.write(content)


if __name__ == "__main__":
    import os
    path = os.environ["TEKIT_ROOT"] + "/examples/llm-api/llm_inference.py"
    #print(extract_meta_info(path))
    generate_examples()
