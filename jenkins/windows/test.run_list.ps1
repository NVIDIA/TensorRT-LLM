param (
    [string]$config = 'SingleDevice',
    [string]$llmSrc = 'C:\tekit',
    [string]$gpu = 'RTX4090',
    [string]$testList = 'l0_windows_premerge',
    [string]$trtVersion4 = '10.8.0.43'
)

$trtVersion3 = $trtVersion4.split('.')[0..2] -join '.'

$trtIncludePath = "C:\workspace\TensorRT-${trtVersion4}\include"
$trtLibPath = "C:\workspace\TensorRT-${trtVersion4}\lib"
$trtPythonPackage = "C:\workspace\TensorRT-${trtVersion4}\python\tensorrt-${trtVersion3}-cp310-none-win_amd64.whl"
$turtleBinPath = (Resolve-Path ".\turtle${config}\bin\trt_test")

$testPythonExe = 'C:\Program Files\Python310\python.exe'
$testPython3Exe = 'C:\Program Files\Python310\python3.exe'

$outputDir = "${gpu}-l0_windows_premerge-${config}"

# Workaround for:
#   RTX4090/test_unittests.py::test_unittests[attention-gpt-no-cache] FAILED [100%]
#   ___________________ test_unittests[attention-gpt-no-cache] ____________________
#   test_unittests.py:124: in test_unittests
#       import pandas as pd
#   E   ModuleNotFoundError: No module named 'pandas'
'pandas' | Out-File -Append -Encoding utf8 -FilePath ".\turtle${config}\requirements.txt"

# Workaround for:
#   RTX4090/test_unittests.py::test_unittests[attention-gpt-no-cache] FAILED [100%]
#   ---------------------------- Captured stderr call -----------------------------
#   ERROR: usage: __main__.py [options] [file_or_dir] [file_or_dir] [...]
#   __main__.py: error: unrecognized arguments: --timeout=1000
#     inifile: C:\tekit\tests\pytest.ini
#     rootdir: C:\tekit\tests
pip install pytest-timeout

# Fix for:
#   UnicodeEncodeError: 'charmap' codec can't encode character '\ufffd' in position 26: character maps to <undefined>
$env:PYTHONIOENCODING = 'utf-8'

$turtleCmdLineParams = @(
    "python",
    ${turtleBinPath},
    "--test-prefix ${gpu}",
    "-D ${llmSrc}\tests\llm-test-defs\turtle\defs",
    "-d 'L:'",
    "-f ${llmSrc}\tests\llm-test-defs\turtle\test_lists\bloom\${testList}.txt",
    "--waives-file ${llmSrc}\tests\llm-test-defs\turtle\test_lists\waivesWindows.txt",
    "--trt-py3-package ${trtPythonPackage}",
    "-I ${trtIncludePath}",
    "-L ${trtLibPath}",
    "--test-timeout",
    "3600",
    "--test-python3-exe '${testPythonExe}'",
    "--junit-xml",
    "--output-dir ${outputDir}"
)
Write-Host $turtleCmdLineParams
Write-Host

Remove-Item -ErrorAction SilentlyContinue -Force -Path ${outputDir} -Recurse

Write-Host ($turtleCmdLineParams -join ' ')
Write-Host

$env:WIN10_PY3_EXE = $testPython3Exe

Invoke-Expression ($turtleCmdLineParams -join ' ')
exit $LASTEXITCODE
