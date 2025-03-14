param (
    [string]$cloneRoot,
    [string]$stagingPath
)

#
# Make paths absolute so that we can safely use Push-Location,
# allowing for more concise commands.
#
$cloneRoot = [string](Resolve-Path $cloneRoot).Path
$stagingPath = [string](Resolve-Path $stagingPath).Path

$srcBuildDir = "$cloneRoot\cpp\build\tensorrt_llm"

$libsDir = (Get-Item $stagingPath).Parent.FullName

function Push-Required-Location([string]$location) {
    if (-not (Test-Path -Path $location -PathType Container)) {
        New-Item -ItemType Directory -Path $location | Out-Null
    }
    Push-Location $location
}

function Copy-File([string]$srcPath) {
    Write-Host Copy $srcPath
    Copy-Item -Path $srcPath
}

function Copy-To-Libs-Dir([string]$srcPath) {
    Push-Required-Location $libsDir
    Copy-Item -Path $srcPath
    Pop-Location
}

Push-Required-Location "$stagingPath\TensorRT-LLM"
    #
    # whl
    #
    Copy-File "$cloneRoot\build\tensorrt_llm-*.whl"

    Push-Required-Location 'benchmarks\cpp'
        #
        # benchmarks and dependencies
        #
        $srcBenchmarkDir = "$cloneRoot\cpp\build\benchmarks"
        Copy-File "$srcBenchmarkDir\bertBenchmark.exe"
        Copy-File "$srcBenchmarkDir\gptSessionBenchmark.exe"
        Copy-File "$srcBenchmarkDir\gptManagerBenchmark.exe"
        ('dll', 'lib') | ForEach-Object {
            (
                "$srcBuildDir\tensorrt_llm.$_",
                "$srcBuildDir\plugins\nvinfer_plugin_tensorrt_llm.$_",
                "$srcBuildDir\kernels\decoderMaskedMultiheadAttention\decoderXQAImplJIT\nvrtcWrapper\tensorrt_llm_nvrtc_wrapper.$_"
            ) | ForEach-Object {
                Copy-File $_
            }
        }
    Pop-Location

    #
    # batch manager
    #
    Copy-To-Libs-Dir `
        "$srcBuildDir\batch_manager\tensorrt_llm_batch_manager_static.lib"

    #
    # executor
    #
    Copy-To-Libs-Dir `
        "$srcBuildDir\executor\tensorrt_llm_executor_static.lib"
    #

    #
    # internal_cutlass_kernels
    #
    Copy-To-Libs-Dir `
        "$srcBuildDir\kernels\internal_cutlass_kernels\tensorrt_llm_internal_cutlass_kernels_static.lib"

    #
    # nvrtc wrapper
    #
    ('dll', 'lib') | ForEach-Object {
        Copy-To-Libs-Dir `
            "$srcBuildDir\kernels\decoderMaskedMultiheadAttention\decoderXQAImplJIT\nvrtcWrapper\tensorrt_llm_nvrtc_wrapper.$_"
    }

Pop-Location
