Write-Host $(Get-WmiObject Win32_VideoController -Filter "Name LIKE 'Nvidia%'").InstalledDisplayDrivers

$installedDriverPath = ((Get-WmiObject Win32_VideoController -Filter "Name LIKE 'Nvidia%'").InstalledDisplayDrivers -split ',')[0]
Write-Host $installedDriverPath

$installedDriverHostPath = $installedDriverPath.Replace('\Windows\', '\HostWindows\')
Write-Host $installedDriverHostPath

$installedDriverHostDirectory = (Get-Item $installedDriverHostPath).Directory.FullName
Write-Host $installedDriverHostDirectory

cmd /c dir $installedDriverHostDirectory

Copy-Item -Path $installedDriverHostDirectory\nvcuda_loader64.dll -Destination C:\Windows\System32\nvcuda.dll
Copy-Item -Path $installedDriverHostDirectory\nvml_loader.dll -Destination C:\Windows\System32\nvml.dll
Copy-Item -Path C:\HostWindows\System32\nvidia-smi.exe -Destination C:\Windows\System32\nvidia-smi.exe

nvidia-smi.exe
