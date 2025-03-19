param (
    [string]$stagingPath,
    [string]$packageName
)

if (Test-Path $packageName) {
    Remove-Item $packageName -Force -Recurse
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
[IO.Compression.ZipFile]::CreateFromDirectory($stagingPath, $packageName)
