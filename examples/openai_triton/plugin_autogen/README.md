# Integrating Triton Kernel with TensorRT Plugin Generator

In the previous [OpenAI Triton Plugin in TensorRT-LLM](../../openai_triton/README.md) tutorial, it is demonstrated how to integrate a Triton kernel by manually writing a TensorRT plugin in C++ as well as a Python wrapper. In the latest TensorRT-LLM, we now have an end-to-end tool called PluginGen that simplifies this process. All you need to do is providing a plugin configuration.

In this example, we will introduce the usage of the PluginGen tool and demonstrate the integration of the [Fused Attention](../openai_triton/fmha_triton.py) kernel.


To use the feature, you need a Triton version posterior to the [d0c35b3](https://github.com/openai/triton/commit/d0c35b3b7d6badf0c0d56a821dddab7ace73b4de) commit
and this example has been tested on the [d4644d6](https://github.com/openai/triton/tree/d4644d6cb3ae674e1f15932cac1f28104795744f) commit.

## Introduction to the PluginGen Toolkit

The PluginGen script can be found at `tensorrt_llm/tools/triton_integration/plugin_gen.py`. Its usage is as follows:

```sh
usage: plugin_gen.py [-h] --workspace WORKSPACE --kernel_config KERNEL_CONFIG [--tensorrt_llm_include_path TENSORRT_LLM_INCLUDE_PATH]
```

There are three command-line arguments:

1. `workspace`: This is the root directory to hold the temporary generation files. PluginGen should not alter anything outside of the workspace,
2. `kernel_config`: This is a Python file that holds a variable called `KERNELS` of type `List[KernelMetaData]`. PluginGen can process one or more kernels at a time,
3. `tensorrt_llm_include_path`: This is the path to the TensorRT LLM include directory. It is used to include the TensorRT LLM header files in the generated plugin.

You can refer to [./kernel_config.py](./kernel_config.py) for an example of `KernelMetaData` for the Fused Attention kernel. It contains several fields:

1. `ios` (short for "input and outputs"): This holds all the metadata of the inputs and outputs of the Triton kernel, including the data type, shape, and the name of the tensor. There are several kinds of arguments:
   - `InputArg`: A common variable for this kernel.
   - `OutputArg`: An output of the kernel.
   - `ParamArg`: A special input that is a constant; it will be mapped to a PluginField in the generated plugin.
   - `DimSizeArg`: A special input that is an expression of the input tensors' shape size; it requires an inference rule to compute the value.
2. `shape_infer_rules`: This field contains two types of rules:
   a) Rules for deducing the shape of the output tensors from the input tensors. The syntax is like `input0[dim_names], input1[dim_names] -> output0[dim_names]`.
   b) Rules for inferring `DimSizeArg`. The syntax is like `input0[dim_names]: some_dim_expression -> arg_name`.

The user should provide the kernel configurations as well as the Triton kernel script, and the PluginGen toolkit will handle the following steps:

1. Trigger the Triton AOT tool to obtain the necessary C files.
2. Generate the C++ code for a TensorRT plugin.
3. Generate the CMAKE code for compiling all the C/C++ files.
4. Perform the compilation and generate `libtriton_plugins.so`.
5. Generate a `functional.py` containing a Python wrapper for this plugin.

After the generation, you should have `libtriton_plugins.so` and `functional.py` in the workspace. You can use them to integrate the Triton kernel by simply using the corresponding Python methods in the generated `functional.py` during the model-building stage, just like other layers located in the TensorRT LLM built-in `functional.py`.

## End-to-End Example for FHMA Kernel Integration

In this section, we will demonstrate the integration of the Fused Attention kernel. The steps are as follows:

### Pre-Stage: Install Triton with a Specific Version

In case the Triton AOT tool's update breaks compatibility, we recommend installing a specific version of Triton. The commit we tested is [d4644d6](https://github.com/openai/triton/tree/d4644d6cb3ae674e1f15932cac1f28104795744f).

Install Triton with the following commands:

```sh
git clone https://github.com/openai/triton
cd triton/python/
pip install cmake && pip install .
cd -
```

### Step 1: Prepare the Configuration for FHMA

To instruct the PluginGen toolkit on how to generate the plugin, please provide a Python file containing the metadata of the kernels. You can refer to [./kernel_config.py](./kernel_config.py) for an example of preparing `KernelMetaData` for the Fused Attention kernel.

### Step 2: Run the PluginGen Tool and Generate the Plugin

```sh
python3 {GIT_ROOT_DIR}/tensorrt_llm/tools/plugin_gen/plugin_gen.py --workspace ./tmp --kernel_config ./kernel_config.py
```

PluginGen will generate all the necessary files within the `./tmp` directory. The final output will be located in the `./tmp/output` directory, where you should ideally find two files:

```
-rw-r--r-- 1 1001 1001    2163 Sep 21 17:13 functional.py
-rwxr-xr-x 1 1001 1001 3748464 Sep 21 17:13 libtriton_plugins.so
```

### Post-Stage: Use the Plugin

To use the plugin in a TensorRT LLM model, please refer to the generated `output/functional.py`. It should contain Python wrappers for all the plugins. To use the plugins, first import `functional.py` and then use the corresponding Python methods to build the model.

For an example of using the Fused Attention plugin in a model, please refer to [build_engine.py](./build_engine.py) for building the TensorRT engine and [run_engine.py](./run_engine.py) for running the engine in the runtime.

To run the example, you can use the following commands:

```sh
# copy the triton script to the current directory
cp ../manual_plugin/fmha_triton.py .

# build the TensorRT engine
python3 build_engine.py

# run the engine
python3 run_engine.py
```
