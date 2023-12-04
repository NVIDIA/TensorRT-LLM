# Command-line --options that let you skip unnecessary installations
param (
    [switch]$skipCMake,
    [switch]$skipVSBuildTools,
    [switch]$skipTRT,
    [string]$TRTPath
)

# Set the error action preference to 'Stop' for the entire script.
# Respond to non-terminating errors by stopping execution and displaying an error message.
$ErrorActionPreference = 'Stop'

# Check for valid usage
if ((-not $skipTRT) -and (-not $TRTPath)) {
    Write-Output "Please provide a path for TensorRT installation using -TRTPath"
    Write-Output "Specify the containing folder, not the TensorRT folder itself, which will be created"
    exit 1
}

# Install CMake
if (-not $skipCMake) {
    Write-Output "Downloading CMake installer"
    Invoke-WebRequest -Uri 'https://github.com/Kitware/CMake/releases/download/v3.27.7/cmake-3.27.7-windows-x86_64.msi' -OutFile 'cmake.msi'
    Write-Output "Installing CMake 3.27.7 silently"
    Start-Process -Wait -FilePath 'msiexec.exe' -ArgumentList '/I cmake.msi /quiet'
    Write-Output "Removing CMake installer"
    Remove-Item -Path 'cmake.msi' -Force
    Write-Output "Adding CMake to system Path"
    [Environment]::SetEnvironmentVariable('Path', "$env:Path;C:\Program Files\CMake\bin", [EnvironmentVariableTarget]::Machine)
    Write-Output "Done CMake installation at 'C:\Program Files\CMake'"
} else {
    Write-Output "Skipping CMake installation"
}

# Install VS Build Tools
if (-not $skipVSBuildTools) {
    Write-Output "Downloading Visual Studio Build Tools installer - this will take a while"
    Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vs_buildtools.exe' -OutFile 'vs_buildtools.exe'
    Write-Output "Installing Visual Studio Build Tools silently - this will take a while"
    Start-Process -Wait -FilePath '.\vs_buildtools.exe' -ArgumentList '--quiet --wait --norestart --nocache --installPath "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools" --includeRecommended --add Microsoft.VisualStudio.Workload.MSBuildTools --add Microsoft.VisualStudio.Workload.VCTools --remove Microsoft.VisualStudio.Component.Windows10SDK.10240 --remove Microsoft.VisualStudio.Component.Windows10SDK.10586 --remove Microsoft.VisualStudio.Component.Windows10SDK.14393 --remove Microsoft.VisualStudio.Component.Windows81SDK' -PassThru
    Write-Output "Removing Visual Studio Build Tools installer"
    Remove-Item -Path 'vs_buildtools.exe' -Force
    Write-Output "Done Visual Studio Build Tools installation at 'C:\ProgramFiles(x86)\Microsoft Visual Studio\2022\BuildTools'"
} else {
    Write-Output "Skipping Visual Studio Build Tools installation"
}

# Install TensorRT 9.2.0.5 for TensorRT-LLM
if (-not $skipTRT) {
    Write-Output "Downloading TensorRT"
    Invoke-WebRequest -Uri 'https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/9.2.0/tensorrt-9.2.0.5.windows10.x86_64.cuda-12.2.llm.beta.zip' -OutFile 'TensorRT-9.2.0.5.zip'
    Write-Output "Extracting TensorRT"
    # Get path
    $absolutePath = Resolve-Path $TRTPath
    Expand-Archive -Path '.\TensorRT-9.2.0.5.zip' -DestinationPath $absolutePath
    Write-Output "Removing TensorRT zip"
    Remove-Item -Path 'TensorRT-9.2.0.5.zip' -Force
    Write-Output "Adding TensorRT to system Path"
    [Environment]::SetEnvironmentVariable('Path', "$env:Path;$absolutePath\TensorRT-9.2.0.5\lib", [EnvironmentVariableTarget]::Machine)
    Write-Output "Installing TensorRT Python wheel"
    pip install $absolutePath\TensorRT-9.2.0.5\python\tensorrt-9.2.0.post12.dev5-cp310-none-win_amd64.whl
    Write-Output "Done TensorRT installation at '$absolutePath\TensorRT-9.2.0.5'"
} else {
    Write-Output "Skipping TensorRT installation"
}
