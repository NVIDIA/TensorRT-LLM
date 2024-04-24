(windows)=

# Installing on Windows

```{note}
The Windows release of TensorRT-LLM is currently in beta. We recommend using the `rel` branch for the most stable experience.
```

**Prerequisites**

1. Clone this repository using [Git for Windows](https://git-scm.com/download/win).

2. Install the dependencies one of two ways:

    1. Run the provided PowerShell script; `setup_env.ps1`, which installs Python, CUDA 12.2, and Microsoft MPI automatically with default settings. Run PowerShell as Administrator to use the script.

    ```bash
    ./setup_env.ps1 [-skipCUDA] [-skipPython] [-skipMPI]
    ```

    2. Install the dependencies one at a time.

        1. Install [Python 3.10](https://www.python.org/downloads/windows/).

            1. Select **Add python.exe to PATH** at the start of the installation. The installation may only add the `python` command, but not the `python3` command.
            2. Navigate to the installation path `%USERPROFILE%\AppData\Local\Programs\Python\Python310` (`AppData` is a hidden folder) and copy `python.exe` to `python3.exe`.

    3. Install [CUDA 12.2 Toolkit](https://developer.nvidia.com/cuda-12-2-2-download-archive?target_os=Windows&target_arch=x86_64). Use the Express Installation option. Installation may require a restart.

    4. Download and install [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=57467). You will be prompted to choose between an `exe`, which installs the MPI executable, and an `msi`, which installs the MPI SDK. Download and install both.

3. Download and unzip [cuDNN](https://developer.nvidia.com/cudnn).

    1. Move the folder to a location you can reference later, such as `%USERPROFILE%\inference\cuDNN`.
    2. Add the libraries and binaries for cuDNN to your system's `Path` environment variable.

        1. Click the Windows button and search for *environment variables*.
        2. Click **Edit the system environment variables** > **Environment Variables**.
        3. In the new window under *System variables*, click **Path** > **Edit**. Add **New** lines for the `bin` and `lib` directories of cuDNN. Your `Path` should include lines like this:

        ```bash
        %USERPROFILE%\inference\cuDNN\bin
        %SERPROFILE%\inference\cuDNN\lib
          ```

        4. Click **OK** on all the open dialog windows.
        5. Close and re-open any existing PowerShell or Git Bash windows so they pick up the new `Path`.


**Steps**

1. Install TensorRT-LLM.

  ```bash
  pip install tensorrt_llm --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu121
  ```

  Run the following command to verify that your TensorRT-LLM installation is working properly.

  ```bash
  python -c "import tensorrt_llm; print(tensorrt_llm._utils.trt_version())"
  ```

2. Build the model.
3. Deploy the model.
