param(
    [Parameter(Mandatory=$true)]
    [string]$cudnnVersion,  # cuDNN version must be specified

    [Parameter(Mandatory=$true)]
    [string]$cudnnPackagePath  # Path to the unzipped cuDNN files must be specified
)

# Root directory for cuDNN
$cudnnRoot = "C:\Program Files\NVIDIA\CUDNN\$cudnnVersion"

# Create cuDNN directories if they don't exist
New-Item -ItemType Directory -Force -Path "$cudnnRoot\bin"
New-Item -ItemType Directory -Force -Path "$cudnnRoot\include"
New-Item -ItemType Directory -Force -Path "$cudnnRoot\lib"

# Copy cuDNN files
Copy-Item -Path "$cudnnPackagePath\bin\cudnn*.dll" -Destination "$cudnnRoot\bin" -Force
Copy-Item -Path "$cudnnPackagePath\include\cudnn*.h" -Destination "$cudnnRoot\include" -Force
Copy-Item -Path "$cudnnPackagePath\lib\cudnn*.lib" -Destination "$cudnnRoot\lib" -Force

# Add cuDNN to the system PATH
$envPath = [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::Machine)
if (-not $envPath.Contains("$cudnnRoot\bin")) {
    [System.Environment]::SetEnvironmentVariable("Path", $envPath + ";$cudnnRoot\bin", [System.EnvironmentVariableTarget]::Machine)
}

# Output message
Write-Output "cuDNN setup complete. Please restart your computer for the changes to take effect."
