# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import subprocess
import sys

import pygit2

sys.path.insert(0, os.path.abspath('.'))

project = 'tensorrt_llm'
copyright = '2024, NVidia'
author = 'NVidia'
branch_name = pygit2.Repository('.').head.shorthand
html_show_sphinx = False
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = ['performance/performance-tuning-guide/introduction.md']

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser',  # for markdown support
    "breathe",
    'sphinx.ext.todo',
    'sphinxarg.ext',
    'sphinx_click',
    'sphinx_copybutton'
]

myst_url_schemes = {
    "http":
    None,
    "https":
    None,
    "source":
    "https://github.com/NVIDIA/TensorRT-LLM/tree/" + branch_name + "/{{path}}",
}

myst_heading_anchors = 4

myst_enable_extensions = [
    "deflist",
]

autosummary_generate = True
copybutton_exclude = '.linenos, .gp, .go'
copybutton_prompt_text = ">>> |$ |# "
copybutton_line_continuation_character = "\\"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# ------------------------  C++ Doc related  --------------------------
# Breathe configuration
breathe_default_project = "TensorRT-LLM"
breathe_projects = {"TensorRT-LLM": "../cpp_docs/xml"}

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

CPP_INCLUDE_DIR = os.path.join(SCRIPT_DIR, '../../cpp/include/tensorrt_llm')
CPP_GEN_DIR = os.path.join(SCRIPT_DIR, '_cpp_gen')
print('CPP_INCLUDE_DIR', CPP_INCLUDE_DIR)
print('CPP_GEN_DIR', CPP_GEN_DIR)


def setup(app):
    from helper import generate_examples, generate_llmapi

    generate_examples()
    generate_llmapi()


def gen_cpp_doc(ofile_name: str, header_dir: str, summary: str):
    cpp_header_files = [
        file for file in os.listdir(header_dir) if file.endswith('.h')
    ]

    with open(ofile_name, 'w') as ofile:
        ofile.write(summary + "\n")
        for header in cpp_header_files:
            ofile.write(f"{header}\n")
            ofile.write("_" * len(header) + "\n\n")

            ofile.write(f".. doxygenfile:: {header}\n")
            ofile.write("   :project: TensorRT-LLM\n\n")


runtime_summary = f"""
Runtime
==========

.. Here are files in the cpp/include/runtime
.. We manually add subsection to enable detailed description in the future
.. It is also doable to automatically generate this file and list all the modules in the conf.py
    """.strip()

subprocess.run(['mkdir', '-p', CPP_GEN_DIR])
gen_cpp_doc(CPP_GEN_DIR + '/runtime.rst', CPP_INCLUDE_DIR + '/runtime',
            runtime_summary)

executor_summary = f"""
Executor
==========

.. Here are files in the cpp/include/executor
.. We manually add subsection to enable detailed description in the future
.. It is also doable to automatically generate this file and list all the modules in the conf.py
    """.strip()

subprocess.run(['mkdir', '-p', CPP_GEN_DIR])
gen_cpp_doc(CPP_GEN_DIR + '/executor.rst', CPP_INCLUDE_DIR + '/executor',
            executor_summary)
