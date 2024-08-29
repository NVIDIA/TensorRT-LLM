# Command-line --options that let you skip unnecessary installations
param (
    [switch]$skipCUDA,
    [switch]$skipPython,
    [switch]$skipMPI = $true,
    [switch]$skipCUDNN = $true,
    [string]$cudaVersion, #CUDA version defaults to $defaultCudaVersion, specify otherwise
    [switch]$skipTRT = $true
)

# Default CUDA version if not specified by user.
$defaultCudaVersion = "12.5.1"

# Set the error action preference to 'Stop' for the entire script.
# Respond to non-terminating errors by stopping execution and displaying an error message.
$ErrorActionPreference = 'Stop'

#The order of data is:
# CUDA Status: 1 present or skipped (preceded by version), 0 not present (preceded by message)
# Python: 0 not present, 1 present at 3.10
# Microsoft MPI: 0 not present, 1 present
# Microsoft MPI in EnvPath: 0 not present, 1 present

New-Item -Path "$($env:LOCALAPPDATA)\trt_env_outlog.txt" -Force

# Install CUDA
if (-not $skipCUDA){
    if(-not ($cudaVersion)){
        $cudaVersion = $defaultCudaVersion
    }
    $cudaVer = "NVIDIA CUDA Toolkit " + $cudaVersion

    if (-not (Get-Package -Name $cudaVer -EA Ignore)) {
        Write-Output "Downloading $cudaVer - this will take a while"
        $ProgressPreference = 'SilentlyContinue'
        if ($cudaVersion -eq "12.2"){
            $cudaUri = 'https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_537.13_windows.exe'
        } elseif ($cudaVersion -eq "12.3"){
            $cudaUri = 'https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_546.12_windows.exe'
        } elseif ($cudaVersion -eq "12.4"){
            $cudaUri = 'https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_551.61_windows.exe'
        } elseif ($cudaVersion -eq "12.4.1"){
            $cudaUri = 'https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_551.78_windows.exe'
        } elseif ($cudaVersion -eq "12.5.1"){
            $cudaUri = 'https://developer.download.nvidia.com/compute/cuda/12.5.1/local_installers/cuda_12.5.1_555.85_windows.exe'
        } else {
            $cudaUri = Read-Host "Please go to https://developer.nvidia.com/cuda-downloads and input the url of the CUDA version you wish to use"
        }
        Invoke-WebRequest -Uri $cudaUri -OutFile 'cuda_installer.exe'

        Write-Output "Installing $cudaVer silently - this will take a while"
        Start-Process -Wait -FilePath 'cuda_installer.exe' -ArgumentList '-s'
        $ProgressPreference = 'Continue'
        Write-Output "Removing CUDA installer"
        Remove-Item -Path 'cuda_installer.exe' -Force
        Write-Output "Done $cudaVer installation at 'C:\Program Files\NVIDIA Corporation' and 'C:\Program Files\NVIDIA GPU Computing Toolkit'"
        Add-Content -Path $env:LOCALAPPDATA\trt_env_outlog.txt -Value "0"
        Add-Content -Path $env:LOCALAPPDATA\trt_env_outlog.txt -Value $cudaVer
    } else {
        Write-Output "CUDA Installation already exists"
        Add-Content -Path $env:LOCALAPPDATA\trt_env_outlog.txt -Value "1"
        Add-Content -Path $env:LOCALAPPDATA\trt_env_outlog.txt -Value "CUDA present"
    }
} else {
        Write-Output "Skipping CUDA installation"
        Add-Content -Path $env:LOCALAPPDATA\trt_env_outlog.txt -Value "1"
        Add-Content -Path $env:LOCALAPPDATA\trt_env_outlog.txt -Value "CUDA present"
}

#Install Python 3.10
if(-not $skipPython){
    if (-not(Test-Path -Path 'C:\Program Files\Python310\python3.exe')) {
        Write-Output "Downloading Python installer"
        Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe' -OutFile 'python-3.10.11.exe'
        Write-Output "Installing Python 3.10 silently and adding to system Path for all users"
        Start-Process -Wait -FilePath 'python-3.10.11.exe' -ArgumentList '/quiet InstallAllUsers=1 PrependPath=1'
        Write-Output "Removing Python installer"
        Remove-Item -Path 'python-3.10.11.exe' -Force
        Write-Output "Creating python3 alias executable"
        Copy-Item -Path 'C:\Program Files\Python310\python.exe' -Destination 'C:\Program Files\Python310\python3.exe'
        Write-Output "Done Python installation at 'C:\Program Files\Python310'"
        [Environment]::SetEnvironmentVariable('Path', "C:\Program Files\Python310;C:\Program Files\Python310\Scripts;$env:Path", [EnvironmentVariableTarget]::Machine)
        Add-Content -Path $env:LOCALAPPDATA\trt_env_outlog.txt -Value "0"
    } else {
        Write-Output "Python installation already exists"
        Add-Content -Path $env:LOCALAPPDATA\trt_env_outlog.txt -Value "1"
    }
} else {
        Write-Output "Skipping Python installation"
        Add-Content -Path $env:LOCALAPPDATA\trt_env_outlog.txt -Value "1"
}

# Install Microsoft MPI
if (-not ($skipMPI)) {
    if (-not (Test-Path -Path 'C:\Program Files\Microsoft MPI\Bin')) {
        Write-Output "Downloading Microsoft MPI not detected"
        Add-Content -Path $env:LOCALAPPDATA\trt_env_outlog.txt -Value "0"
        # The latest version is 10.1.3, but it requires you to get a temporary download
        # link.
        # https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi-release-notes
        # We use 10.1.1 which has a release on the GitHub page
        Write-Output "Downloading Microsoft MPI installer"
        Invoke-WebRequest -Uri 'https://github.com/microsoft/Microsoft-MPI/releases/download/v10.1.1/msmpisetup.exe' -OutFile 'msmpisetup.exe'
        Write-Output "Installing Microsoft MPI"
        Start-Process -Wait -FilePath '.\msmpisetup.exe' -ArgumentList '-unattend'
        Write-Output "Removing MPI installer"
        Remove-Item -Path 'msmpisetup.exe' -Force
        Write-Output "Adding MPI to system Path"
        Add-Content -Path $env:LOCALAPPDATA\trt_env_outlog.txt -Value "0"
        [Environment]::SetEnvironmentVariable('Path', "$env:Path;C:\Program Files\Microsoft MPI\Bin", [EnvironmentVariableTarget]::Machine)
        Write-Output "Downloading Microsoft MPI SDK installer"
        Invoke-WebRequest -Uri 'https://github.com/microsoft/Microsoft-MPI/releases/download/v10.1.1/msmpisdk.msi' -OutFile 'msmpisdk.msi'
        Write-Output "Installing Microsoft MPI SDK"
        Start-Process -Wait -FilePath 'msiexec.exe' -ArgumentList '/I msmpisdk.msi /quiet'
        Write-Output "Removing MPI SDK installer"
        Remove-Item -Path 'msmpisdk.msi' -Force
        Write-Output "Done MPI installation at 'C:\Program Files\Microsoft MPI' and 'C:\Program Files (x86)\Microsoft SDKs\MPI'"
    } else {
        Write-Output "Microsoft MPI found"
        Add-Content -Path $env:LOCALAPPDATA\trt_env_outlog.txt -Value "1"

        #Check if its part of PATH:
        $pathContent = [Environment]::GetEnvironmentVariable('path', 'Machine')
        $myPath = "C:\Program Files\Microsoft MPI\Bin"

        if ($pathContent -split ';'  -contains  $myPath)
        {
            Write-Output "MPI exists in PATH"
            Add-Content -Path $env:LOCALAPPDATA\trt_env_outlog.txt -Value "1"
        } else {
            Write-Output "MPI does not exist in PATH, adding..."
            Add-Content -Path $env:LOCALAPPDATA\trt_env_outlog.txt -Value "0"
            [Environment]::SetEnvironmentVariable('Path', "$env:Path;C:\Program Files\Microsoft MPI\Bin", [EnvironmentVariableTarget]::Machine)
        }
    }
} else {
    Write-Output "Skipping MPI installation"
    Add-Content -Path $env:LOCALAPPDATA\trt_env_outlog.txt -Value "1"
    Add-Content -Path $env:LOCALAPPDATA\trt_env_outlog.txt -Value "1"
}

if(-not $skipCUDNN){
    $CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\' + $cudaVersion + '\lib\x64\cudnn.lib'
    if(-not(Test-Path -Path $CUDA_PATH)){
        Write-Output "Installing CUDNN"
        Add-Content -Path $env:LOCALAPPDATA\trt_env_outlog.txt -Value "0"
        New-Item -Path $env:LOCALAPPDATA\CUDNN -ItemType Directory -Force
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri 'https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.1.0.70_cuda12-archive.zip' -OutFile $env:LOCALAPPDATA\CUDNN\cudnn.zip
        Expand-Archive -Path $env:LOCALAPPDATA\CUDNN\cudnn.zip -DestinationPath $env:LOCALAPPDATA\CUDNN\cudnn_unzip

        New-Item -Path ".\" -Name "CUDNN" -ItemType "directory"
        $binPath = Join-Path $PWD \CUDNN\bin
        $includePath = Join-Path $PWD \CUDNN\include
        $libPath = Join-Path $PWD \CUDNN\lib\x64
        New-Item -Path $binPath -ItemType Directory
        New-Item -Path $includePath -ItemType Directory
        New-Item -Path $libPath -ItemType Directory
        Copy-Item -Path "$env:LOCALAPPDATA\CUDNN\cudnn_unzip\cudnn-windows-x86_64-9.1.0.70_cuda12-archive\bin\*" -Destination $binPath
        Copy-Item -Path "$env:LOCALAPPDATA\CUDNN\cudnn_unzip\cudnn-windows-x86_64-9.1.0.70_cuda12-archive\include\*" -Destination $includePath
        Copy-Item -Path "$env:LOCALAPPDATA\CUDNN\cudnn_unzip\cudnn-windows-x86_64-9.1.0.70_cuda12-archive\lib\x64\*" -Destination $libPath

        [Environment]::SetEnvironmentVariable("CUDNN", "$PWD;$binPath;$includePath;$libPath", [EnvironmentVariableTarget]::Machine)

        Write-Output "Cleaning up CUDNN files"
        Remove-Item -Path $env:LOCALAPPDATA\CUDNN\cudnn.zip -Force
        Remove-Item -Recurse -Force $env:LOCALAPPDATA\CUDNN\cudnn_unzip
    } else {
        Write-Output "CUDNN already present"
        Add-Content -Path $env:LOCALAPPDATA\trt_env_outlog.txt -Value "1"
    }
} else {
    Write-Output "Skipping CUDNN installation"
    Add-Content -Path $env:LOCALAPPDATA\trt_env_outlog.txt -Value "1"
}

# Install TensorRT
if (-not ($skipTRT)) {
    $TRT_BASE = Join-Path $PWD \TensorRT
    if (-not (Test-Path -Path $TRT_BASE)) {
        Write-Output "Grabbing TensorRT..."
        $ProgressPreference = 'SilentlyContinue'
        New-Item -Path .\TensorRT -ItemType Directory
        Invoke-WebRequest -Uri 'https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.3.0/zip/TensorRT-10.3.0.26.Windows.win10.cuda-12.5.zip' -OutFile .\TensorRT\trt.zip
        Expand-Archive -Path .\TensorRT\trt.zip -DestinationPath .\TensorRT\
        Remove-Item -Path .\TensorRT\trt.zip -Force
        $trtPath = Join-Path $TRT_BASE TensorRT-10.3.0.26
        Write-Output "TensorRT installed at ${trtPath}"

        $trtSubPaths = @{
            "bin" = Join-Path $trtPath bin
            "include" = Join-Path $trtPath include
            "lib" = Join-Path $trtPath lib
        }

        foreach ($key in $trtSubPaths.Keys) {
            $subPath = $trtSubPaths[$key]
            if (-not (Test-Path -Path $subPath)) {
              Write-Error "TensorRT ${key} path ${subPath} does not exist!"
            }
        }
        $TRTEnvVar = $trtSubPaths.Values -join ";"

        [Environment]::SetEnvironmentVariable("TRT", "$TRTEnvVar", [EnvironmentVariableTarget]::Machine)
        Add-Content -Path $env:LOCALAPPDATA\trt_env_outlog.txt -Value "0"
    } else {
        Write-Output "TensorRT already present"
        Add-Content -Path $env:LOCALAPPDATA\trt_env_outlog.txt -Value "1"
    }
} else {
    Write-Output "Skipping TRT installation"
    Add-Content -Path $env:LOCALAPPDATA\trt_env_outlog.txt -Value "1"
}


return $env:Path
