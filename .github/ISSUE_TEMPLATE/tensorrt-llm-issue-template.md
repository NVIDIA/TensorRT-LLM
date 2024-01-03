---
name: TensorRT-LLM Issue Template
about: 'As shown in the title '
title: ''
labels: ''
assignees: ''

---

name: "Bug Report"
description: Submit a bug report to help us improve TensorRT-LLM
labels: [ "bug" ]
body:
  - type: textarea
    id: system-info
    attributes:
      label: System Info
      description: Please share your system info with us.
      placeholder: |
        - CPU architecture (e.g., x86_64, aarch64)
        - CPU/Host memory size (if known)
        - GPU properties
          - GPU name (e.g., NVIDIA H100, NVIDIA A100, NVIDIA L40S)
          - GPU memory size (if known)
          - Clock frequencies used (if applicable)
        - Libraries
          - TensorRT-LLM branch or tag (e.g., main, v0.7.1)
          - TensorRT-LLM commit (if known)
          - Versions of TensorRT, AMMO, CUDA, cuBLAS, etc. used
          - Container used (if running TensorRT-LLM in a container)
        - NVIDIA driver version
        - OS (Ubuntu 22.04, CentOS 7, Windows 10)
        - Any other information that may be useful in reproducing the bug
    validations:
      required: true

  - type: textarea
    id: who-can-help
    attributes:
      label: Who can help?
      description: |
        Your issue will be replied to more quickly if you can figure out the right person to tag with @
        Here is a rough guide of **who to tag**.
        
        All issues are read by one of the core maintainers, so if you don't know who to tag, just leave this blank and
        a core maintainer will ping the right person.
        
        Please tag fewer than 3 people.
        
        Quantization: @Tracin 

        Documentation: @juney-nvidia 

        Feature request: @ncomly-nvidia

        Others: @byshiue
        
      placeholder: "@Username ..."

  - type: checkboxes
    id: information-scripts-examples
    attributes:
      label: Information
      description: 'The problem arises when using:'
      options:
        - label: "The official example scripts"
        - label: "My own modified scripts"

  - type: checkboxes
    id: information-tasks
    attributes:
      label: Tasks
      description: "The tasks I am working on are:"
      options:
        - label: "An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)"
        - label: "My own task or dataset (give details below)"

  - type: textarea
    id: reproduction
    validations:
      required: true
    attributes:
      label: Reproduction
      description: |
        Please provide a code sample that reproduces the problem you ran into. It can be a Colab link or just a code snippet.
        If you have code snippets, error messages, stack traces please provide them here as well.
        Important! Use code tags to correctly format your code. See https://help.github.com/en/github/writing-on-github/creating-and-highlighting-code-blocks#syntax-highlighting
        Do not use screenshots, as they are hard to read and (more importantly) don't allow others to copy-and-paste your code.
        It is best if we can reproduce your issue by only copy-and-paste your scripts and codes.

      placeholder: |
        Steps to reproduce the behavior:

          1.          
          2.
          3.

  - type: textarea
    id: expected-behavior
    validations:
      required: true
    attributes:
      label: Expected behavior
      description: "Provide a brief summary of the expected behavior of the software. Provide output files or examples if possible."

  - type: textarea
    id: actual-behavior
    validations:
      required: true
    attributes:
      label: actual behavior
      description: "Describe the actual behavior of the software and how it deviates from the expected behavior. Provide output files or examples if possible."

  - type: textarea
    id: additioanl-notes
    validations:
      required: true
    attributes:
      label: additional notes
      description: "Provide any additional context here you think might be useful for the TensorRT-LLM team to help debug this issue (such as experiments done, potential things to investigate)."
