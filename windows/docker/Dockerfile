# https://learn.microsoft.com/en-us/visualstudio/install/build-tools-container?view=vs-2022

# Use the Windows Server Core 2019 image.
FROM mcr.microsoft.com/windows/servercore:ltsc2019

# Restore the default Windows shell for correct batch processing.
# (Used for VS Build Tools installation)
SHELL ["cmd", "/S", "/C"]

# -----------------------------------------------------------------------------

# Install CUDA 12.2

RUN powershell -Command \
    $ErrorActionPreference = 'Stop'; \
    Invoke-WebRequest -Uri https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_537.13_windows.exe \
    -OutFile "cuda_installer.exe"; \
    Start-Process cuda_installer.exe -Wait -ArgumentList '-s'; \
    Remove-Item cuda_installer.exe -Force

# -----------------------------------------------------------------------------

# Install Python 3.10.11

# Download and install Python
RUN powershell -Command \
    $ErrorActionPreference = 'Stop'; \
    Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe -OutFile python-3.10.11.exe ; \
    Start-Process python-3.10.11.exe -Wait -ArgumentList '/quiet InstallAllUsers=1 PrependPath=1' ; \
    Remove-Item python-3.10.11.exe -Force

# Add python3 command
RUN powershell -Command \
    cp "\"C:\\\\Program Files\\\\Python310\\\\python.exe\" \"C:\\\\Program Files\\\\Python310\\\\python3.exe\""

# -----------------------------------------------------------------------------

# Install Microsoft MPI

# The latest version is 10.1.3, but it requires you to get a temporary download
# link.
# https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi-release-notes
# We use 10.1.1 which has a release on the GitHub page
RUN powershell -Command \
    $ErrorActionPreference = 'Stop'; \
    Invoke-WebRequest -Uri https://github.com/microsoft/Microsoft-MPI/releases/download/v10.1.1/msmpisetup.exe \
    -OutFile "msmpisetup.exe"; \
    Start-Process .\msmpisetup.exe -Wait ; \
    Remove-Item msmpisetup.exe -Force

# Add MPI binaries to Path
RUN setx Path "%Path%;C:\Program Files\Microsoft MPI\Bin"

# Download the MSMPI SDK
RUN powershell -Command \
    $ErrorActionPreference = 'Stop'; \
    Invoke-WebRequest -Uri https://github.com/microsoft/Microsoft-MPI/releases/download/v10.1.1/msmpisdk.msi \
    -OutFile "msmpisdk.msi"; \
    Start-Process msiexec.exe -Wait -ArgumentList '/I msmpisdk.msi /quiet'; \
    Remove-Item msmpisdk.msi -Force

# -----------------------------------------------------------------------------

# Install CMake

RUN powershell -Command \
    $ErrorActionPreference = 'Stop'; \
    Invoke-WebRequest -Uri https://github.com/Kitware/CMake/releases/download/v3.27.7/cmake-3.27.7-windows-x86_64.msi \
    -OutFile "cmake.msi"; \
    Start-Process msiexec.exe -Wait -ArgumentList '/I cmake.msi /quiet'; \
    Remove-Item cmake.msi -Force

# Add CMake binaries to Path
RUN setx Path "%Path%;C:\Program Files\CMake\bin"

# -----------------------------------------------------------------------------

# Install VS Build Tools

RUN \
    # Download the Build Tools bootstrapper.
    curl -SL --output vs_buildtools.exe https://aka.ms/vs/17/release/vs_buildtools.exe \
    \
    # Install Build Tools with the Microsoft.VisualStudio.Workload.AzureBuildTools workload, excluding workloads and components with known issues.
    && (start /w vs_buildtools.exe --quiet --wait --norestart --nocache \
        --installPath "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools" \
        --includeRecommended \
        --add Microsoft.VisualStudio.Workload.MSBuildTools \
        --add Microsoft.VisualStudio.Workload.VCTools \
        --remove Microsoft.VisualStudio.Component.Windows10SDK.10240 \
        --remove Microsoft.VisualStudio.Component.Windows10SDK.10586 \
        --remove Microsoft.VisualStudio.Component.Windows10SDK.14393 \
        --remove Microsoft.VisualStudio.Component.Windows81SDK \
        || IF "%ERRORLEVEL%"=="3010" EXIT 0) \
    \
    # Cleanup
    && del /q vs_buildtools.exe

# -----------------------------------------------------------------------------

# Install Vim (can delete this but it's nice to have)

RUN powershell -Command \
    $ErrorActionPreference = 'Stop'; \
    Invoke-WebRequest -Uri https://ftp.nluug.nl/pub/vim/pc/gvim90.exe \
    -OutFile "install_vim.exe"; \
    Start-Process install_vim.exe -Wait -ArgumentList '/S'; \
    Remove-Item install_vim.exe -Force

# Add Vim binaries to Path
RUN setx Path "%Path%;C:\Program Files (x86)\Vim\vim90"

# -----------------------------------------------------------------------------

# Install Chocolatey
# Chocolatey is a package manager for Windows
# I probably could've used it to install some of the above, but I didn't...

# If you try to install Chocolatey 2.0.0, it fails on .NET Framework 4.8 installation
# https://stackoverflow.com/a/76470753
ENV chocolateyVersion=1.4.0

# https://docs.chocolatey.org/en-us/choco/setup#install-with-cmd.exe
RUN powershell -Command \
    $ErrorActionPreference = 'Stop'; \
    powershell.exe -NoProfile -InputFormat None -ExecutionPolicy Bypass \
    -Command "[System.Net.ServicePointManager]::SecurityProtocol = 3072; \
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))" && \
    SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"

# -----------------------------------------------------------------------------

# Install Git via Chocolatey
RUN powershell -Command \
    choco install git -y

# -----------------------------------------------------------------------------

# Install CUDA 11.8 NVTX

#RUN powershell -Command \
#    $ErrorActionPreference = 'Stop'; \
#    Invoke-WebRequest -Uri https://developer.download.nvidia.com/compute/cuda/11.8.0/network_installers/cuda_11.8.0_windows_network.exe \
#    -OutFile "cuda_11_installer.exe"; \
#    Start-Process cuda_11_installer.exe -Wait -ArgumentList '-s nvtx_11.8'; \
#    Remove-Item cuda_11_installer.exe -Force

# The above command-line installation method installs NVTX headers at
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include\nvtx3\
# CMake can't find this location for some reason.
# Instead, we just copy the older NvToolsExt version to where CMake expects.
# This assumes NvToolsExt was installed on the host machine using the
# CUDA 11.8 GUI installer and copied to the build context
COPY ["NvToolsExt", "C:\\\\Program Files\\\\NVIDIA Corporation\\\\NvToolsExt"]

# -----------------------------------------------------------------------------

# Create a working directory
WORKDIR "C:\\\\workspace"

# -----------------------------------------------------------------------------

# Download and unzip TensorrRT 9.2.0.5 for TensorRT-LLM
RUN powershell -Command \
    $ErrorActionPreference = 'Stop'; \
    Invoke-WebRequest -Uri https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/9.2.0/tensorrt-9.2.0.5.windows10.x86_64.cuda-12.2.llm.beta.zip \
    -OutFile TensorRT-9.2.0.5.zip; \
    Expand-Archive .\TensorRT-9.2.0.5.zip -DestinationPath .; \
    Remove-Item TensorRT-9.2.0.5.zip -Force

# Add TensorRT libs to Path
RUN setx Path "%Path%;C:\workspace\TensorRT-9.2.0.5\lib"

# Install TensorRT Python wheel
RUN powershell -Command \
    $ErrorActionPreference = 'Stop'; \
    pip install TensorRT-9.2.0.5\python\tensorrt-9.2.0.post12.dev5-cp310-none-win_amd64.whl

# -----------------------------------------------------------------------------

# Copy cuDNN into the working directory
# This assumes cuDNN exists on the host machine in the build context
COPY ["cuDNN", "cuDNN"]

# Add cuDNN libs and bin to Path.
RUN setx Path "%Path%;C:\workspace\cuDNN\lib;C:\workspace\cuDNN\bin;"

# -----------------------------------------------------------------------------

# Define the entry point for the docker container.
# This entry point launches the 64-bit PowerShell developer shell.
# We need to launch with amd64 arch otherwise Powershell defaults to x86 32-bit build commands which don't jive with CUDA
ENTRYPOINT ["C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\Common7\\Tools\\VsDevCmd.bat", "-arch=amd64", "&&", "powershell.exe", "-NoLogo", "-ExecutionPolicy", "Bypass"]
