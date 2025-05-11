# Contrib Examples

We create this directory to store the community contributed projects.

Contributors can develop inference time compute methods with various Controller/Task/Worker.

We will continue to move some generic works on this directory back to the main code.

### How to create a new project?

Just create a new directory in `tensorrt_llm/scaffolding/contrib/` and add your code there.

### How to make your code include Controller/Task/Worker can be reused by other projects?

Just add your Controller/Task/Worker to the `__init__.py` file of `tensorrt_llm/scaffolding/contrib/`.

### How to show examples of your project?

Just add your example to the `examples/scaffolding/contrib/` directory.

In summary, the part of the code you want to be imported by other users or projects should be put on `tensorrt_llm/scaffolding/contrib/` directory and added to the `__init__.py` file. The code to run the project and show the results should be put on `examples/scaffolding/contrib/` directory.
