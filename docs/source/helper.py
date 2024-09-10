import logging
import math
from pathlib import Path


def underline(title: str, character: str = "=") -> str:
    return f"{title}\n{character * len(title)}"


def generate_title(filename: str) -> str:
    # Turn filename into a title
    title = filename.replace("_", " ").title()
    # Underline title

    title = ' '.join([
        word.upper() if word.lower() == 'llm' else word
        for word in title.split()
    ])
    title = underline(title)
    return title


def generate_examples():
    root_dir = Path(__file__).parent.parent.parent.resolve()

    # Source paths
    script_dir = root_dir / "examples/llm-api"
    script_paths = sorted(
        script_dir.glob("*.py"),
        # The autoPP example should be at the end since it is a preview example
        key=lambda x: math.inf if 'llm_auto_parallel' in x.stem else 0)

    ignore_list = {'__init__.py', 'quickstart_example.py'}
    script_paths = [i for i in script_paths if i.name not in ignore_list]
    # Destination paths
    doc_dir = root_dir / "docs/source/llm-api-examples"
    doc_paths = [doc_dir / f"{path.stem}.rst" for path in script_paths]

    # Generate the example docs for each example script
    for script_path, doc_path in zip(script_paths, doc_paths):
        if script_path.name in ignore_list:
            logging.warning(f"Ignoring file: {script_path.name}")
            continue
        script_url = f"https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llm-api/{script_path.name}"

        # Make script_path relative to doc_path and call it include_path
        include_path = '../../..' / script_path.relative_to(root_dir)
        content = (f"{generate_title(doc_path.stem)}\n\n"
                   f"Source {script_url}.\n\n"
                   f".. literalinclude:: {include_path}\n"
                   "    :language: python\n"
                   "    :linenos:\n")
        with open(doc_path, "w+") as f:
            f.write(content)

    # Generate the toctree for the example scripts
    with open(doc_dir / "llm_examples_index.template.rst_") as f:
        examples_index = f.read()
    with open(doc_dir / "llm_api_examples.rst", "w+") as f:
        example_docs = "\n   ".join(path.stem for path in script_paths)

        f.write(examples_index.replace(r"%EXAMPLE_DOCS%", example_docs))


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
    doc_path = doc_dir / "index.rst"

    hlapi_all_file = root_dir / "tensorrt_llm/hlapi/__init__.py"
    public_classes_names = extract_all_and_eval(hlapi_all_file)['__all__']

    content = underline("API Reference", "-") + "\n\n"
    for cls_name in public_classes_names:
        cls_name = cls_name.strip()
        content += (f".. autoclass:: tensorrt_llm.hlapi.{cls_name}\n"
                    "    :members:\n"
                    "    :undoc-members:\n"
                    "    :special-members: __init__\n"
                    "    :show-inheritance:\n")

    with open(doc_path, "w+") as f:
        f.write(content)
