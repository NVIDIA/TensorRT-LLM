# Command-line --options that let you skip unnecessary installations
param (
    [switch]$skipCUDA,
    [switch]$skipPython,
    [switch]$skipMPI
)

# Set the error action preference to 'Stop' for the entire script.
# Respond to non-terminating errors by stopping execution and displaying an error message.
$ErrorActionPreference = 'Stop'

# Install CUDA 12.2
if (-not ($skipCUDA)) {
    Write-Output "Downloading CUDA 12.2 - this will take a while"
    Invoke-WebRequest -Uri 'https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_537.13_windows.exe' -OutFile 'cuda_installer.exe'
    Write-Output "Installing CUDA 12.2 silently - this will take a while"
    Start-Process -Wait -FilePath 'cuda_installer.exe' -ArgumentList '-s'
    Write-Output "Removing CUDA installer"
    Remove-Item -Path 'cuda_installer.exe' -Force
    Write-Output "Done CUDA installation at 'C:\Program Files\NVIDIA Corporation' and 'C:\Program Files\NVIDIA GPU Computing Toolkit'"
} else {
    Write-Output "Skipping CUDA installation"
}

# Install Python 3.10.11
if (-not ($skipPython)) {
    Write-Output "Downloading Python installer"
    Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe' -OutFile 'python-3.10.11.exe'
    Write-Output "Installing Python 3.10 silently and adding to system Path for all users"
    Start-Process -Wait -FilePath 'python-3.10.11.exe' -ArgumentList '/quiet InstallAllUsers=1 PrependPath=1'
    Write-Output "Removing Python installer"
    Remove-Item -Path 'python-3.10.11.exe' -Force
    Write-Output "Creating python3 alias executable"
    Copy-Item -Path 'C:\Program Files\Python310\python.exe' -Destination 'C:\Program Files\Python310\python3.exe'
    Write-Output "Done Python installation at 'C:\Program Files\Python310'"
} else {
    Write-Output "Skipping Python installation"
}

# Install Microsoft MPI
if (-not ($skipMPI)) {
    Write-Output "Downloading Microsoft MPI installer"
    Invoke-WebRequest -Uri 'https://github.com/microsoft/Microsoft-MPI/releases/download/v10.1.1/msmpisetup.exe' -OutFile 'msmpisetup.exe'
    Write-Output "Installing Microsoft MPI"
    Start-Process -Wait -FilePath '.\msmpisetup.exe' -ArgumentList '-unattend'
    Write-Output "Removing MPI installer"
    Remove-Item -Path 'msmpisetup.exe' -Force
    Write-Output "Adding MPI to system Path"
    [Environment]::SetEnvironmentVariable('Path', "$env:Path;C:\Program Files\Microsoft MPI\Bin", [EnvironmentVariableTarget]::Machine)
    Write-Output "Downloading Microsoft MPI SDK installer"
    Invoke-WebRequest -Uri 'https://github.com/microsoft/Microsoft-MPI/releases/download/v10.1.1/msmpisdk.msi' -OutFile 'msmpisdk.msi'
    Write-Output "Installing Microsoft MPI SDK"
    Start-Process -Wait -FilePath 'msiexec.exe' -ArgumentList '/I msmpisdk.msi /quiet'
    Write-Output "Removing MPI SDK installer"
    Remove-Item -Path 'msmpisdk.msi' -Force
    Write-Output "Done MPI installation at 'C:\Program Files\Microsoft MPI' and 'C:\Program Files (x86)\Microsoft SDKs\MPI'"
} else {
    Write-Output "Skipping MPI installation"
}

# TODO Automate cuDNN installation
