param (
    [string]$cloneRoot,
    [string]$stagingPath
)

#
# Make paths absolute so that we can safely use Push-Location,
# allowing for more concise commands.
#
$cloneRoot = [string](Resolve-Path $cloneRoot).Path

$srcStagingPath = "$stagingPath\TensorRT-LLM\src"

#
# Create |stagingPath|
#
if (Test-Path -Path $stagingPath -PathType Container) {
    Write-Host Remove $stagingPath
    Remove-Item -Force -Path $stagingPath -Recurse
}
if (-not (Test-Path -Path $srcStagingPath -PathType Container)) {
    Write-Host Create $srcStagingPath
    New-Item -ItemType Directory -Name $srcStagingPath | Out-Null
}

Write-Host Push $srcStagingPath
Push-Location $srcStagingPath
    #
    # Copy |cloneRoot| into |srcStagingPath|
    #
    $nonSourceFolders = (
        'build'
    )
    Write-Host Copy "$cloneRoot\*"
    Copy-Item -Exclude $nonSourceFolders -Path $cloneRoot\* -Recurse

    #
    # Remove .git files and directories
    #
    Write-Host 'Remove .git (multiple)'
    Get-ChildItem -Filter .git -Recurse | Remove-Item -Force -Recurse

    #
    # Remove cpp\build
    #
    Write-Host 'Remove cpp\build'
    Remove-Item -Force -Path cpp\build -Recurse

    #
    # Remove tensorrt_llm\libs
    #
    Write-Host 'Remove tensorrt_llm\libs'
    Remove-Item -Force -Path tensorrt_llm\libs -Recurse

Write-Host Pop
Pop-Location
