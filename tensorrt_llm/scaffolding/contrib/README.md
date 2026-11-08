# Contribute Guide

Community adopt a flexible approach for contributors as most of workers are extending Controller/Task/Worker.

Contributors can create a new project to develop various Controller/Task/Worker. And a project can import Controller/Task/Worker from other projects. This approach can reduce the difficulty and threshold of code merge, and also help the community discover generic and important requirements.

Community will continue to reorganize the projects and move some generic units to core directory.

### How to create a new project?

Just create a new directory in `tensorrt_llm/scaffolding/contrib/` and add your code there.

### How to make your code include Controller/Task/Worker can be reused by other projects?

Just add your Controller/Task/Worker to the `__init__.py` file of `tensorrt_llm/scaffolding/contrib/`.

### How to show examples of your project?

Just add your example to the `examples/scaffolding/contrib/` directory.

In summary, the part of the code you want to be imported by other users or projects should be put on `tensorrt_llm/scaffolding/contrib/` directory and added to the `__init__.py` file. The code to run the project and show the results should be put on `examples/scaffolding/contrib/` directory.
