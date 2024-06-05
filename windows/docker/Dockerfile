# https://learn.microsoft.com/en-us/visualstudio/install/build-tools-container?view=vs-2022

# Use the Windows Server Core 2019 image.
FROM mcr.microsoft.com/windows/servercore:ltsc2019 AS devel

SHELL ["powershell", "-Command", "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue';"]

# -----------------------------------------------------------------------------
# Create a working directory

WORKDIR "C:\\\\workspace"

# -----------------------------------------------------------------------------
# Install runtime dependencies

COPY setup_env.ps1 C:\\workspace\\setup_env.ps1
# TRT is installed along with build-time dependencies
RUN C:\workspace\setup_env.ps1 -skipTRT -skipCUDNN
RUN Remove-Item "C:\workspace\setup_env.ps1" -Force
# If enabling CUDNN, CUDNN paths are populated in the env variable CUDNN, add it to PATH
# RUN [Environment]::SetEnvironmentVariable('Path', $Env:Path + ';' + $Env:CUDNN, [EnvironmentVariableTarget]::Machine)

# -----------------------------------------------------------------------------
# Install build-time dependencies

COPY setup_build_env.ps1 C:\\workspace\\setup_build_env.ps1
# TRT is installed in workspace
RUN C:\workspace\setup_build_env.ps1 -TRTPath 'C:\\workspace'
RUN Remove-Item "C:\workspace\setup_build_env.ps1" -Force

# Add binaries to Path
RUN [Environment]::SetEnvironmentVariable('Path', $Env:Path + ';C:\Program Files\CMake\bin', [EnvironmentVariableTarget]::Machine)

# -----------------------------------------------------------------------------

# Install Vim (can delete this but it's nice to have)
# and add binaries to Path

RUN Invoke-WebRequest -Uri https://ftp.nluug.nl/pub/vim/pc/gvim90.exe \
    -OutFile "install_vim.exe"; \
    Start-Process install_vim.exe -Wait -ArgumentList '/S'; \
    Remove-Item install_vim.exe -Force ; \
    [Environment]::SetEnvironmentVariable('Path', $Env:Path + ';C:\Program Files (x86)\Vim\vim90', [EnvironmentVariableTarget]::Machine)
# -----------------------------------------------------------------------------

# Install Chocolatey
# Chocolatey is a package manager for Windows

# If you try to install Chocolatey 2.0.0, it fails on .NET Framework 4.8 installation
# https://stackoverflow.com/a/76470753
ENV chocolateyVersion=1.4.0

# https://docs.chocolatey.org/en-us/choco/setup#install-with-powershell.exe
RUN Set-ExecutionPolicy Bypass -Scope Process -Force; \
 [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; \
 iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# -----------------------------------------------------------------------------

# Install Git via Chocolatey
RUN choco install git -y

# -----------------------------------------------------------------------------
# Install CUDA 11.8 NVTX
RUN Invoke-WebRequest -Uri https://developer.download.nvidia.com/compute/cuda/11.8.0/network_installers/cuda_11.8.0_windows_network.exe \
    -OutFile cuda_11.8.0_windows_network.exe; \
    Invoke-WebRequest -Uri https://7-zip.org/a/7zr.exe \
    -OutFile 7zr.exe

RUN .\7zr.exe e -i!'nsight_nvtx\nsight_nvtx\NVIDIA NVTX Installer.x86_64.Release.v1.21018621.Win64.msi' cuda_11.8.0_windows_network.exe ;

RUN cmd.exe /S /C "msiexec.exe /i 'NVIDIA NVTX Installer.x86_64.Release.v1.21018621.Win64.msi' /norestart /quiet"

RUN Remove-Item 'NVIDIA NVTX Installer.x86_64.Release.v1.21018621.Win64.msi' -Force ; \
    Remove-Item 7zr.exe -Force ; \
    Remove-Item cuda_11.8.0_windows_network.exe -Force

# -----------------------------------------------------------------------------

# Define the entry point for the docker container.
# This entry point launches the 64-bit PowerShell developer shell.
# We need to launch with amd64 arch otherwise Powershell defaults to x86 32-bit build commands which don't jive with CUDA
ENTRYPOINT ["C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\Common7\\Tools\\VsDevCmd.bat", "-arch=amd64", "&&", "powershell.exe", "-NoLogo", "-ExecutionPolicy", "Bypass"]

# -----------------------------------------------------------------------------
# COPY requirements-windows.txt C:\\workspace\\requirements-windows.txt
# COPY requirements-dev-windows.txt C:\\workspace\\requirements-dev-windows.txt
# RUN python3 -m pip --no-cache-dir install -r C:\workspace\requirements-dev-windows.txt
# RUN Remove-Item "C:\workspace\requirements-windows.txt" -Force
# RUN Remove-Item "C:\workspace\requirements-dev-windows.txt" -Force
