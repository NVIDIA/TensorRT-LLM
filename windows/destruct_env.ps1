$defaultEnv = @()
foreach($line in Get-Content -Path $env:LOCALAPPDATA\trt_env_outlog.txt) {
    $defaultEnv += $line
}

#The order of data is:
# CUDA Status: 1 present (preceded by version), 0 not present (preceded by message)
# Python: 0 not present, 1 present at 3.10
# Microsoft MPI: 0 not present, 1 present
# Microsoft MPI in EnvPath: 0 not present, 1 present

#0 = CUDA
#2 = CUDA Version
#3 = Python
#4 = MPI Presence
#5 = CUDNN
#6 = TRT

if ($defaultEnv[0].Equals("0")) {
    Write-Output "Removing CUDA"
    Uninstall-Package -Name $defaultEnv[1]
}

if ($defaultEnv[2].Equals("0")) {
    Write-Output "Removing Python 3.10.11"
    Uninstall-Package -Name "Python 3.10.11 (64-bit)"
}

if ($defaultEnv[3].Equals("0")) {
    #MPI must be removed
    Write-Output "Removing MPI"
    uninstall-package -ProviderName msi -Name "Microsoft MPI SDK (10.1.12498.16)"
    uninstall-package -ProviderName msi -Name "Microsoft MPI (10.1.12498.16)"
}

if ($defaultEnv[4].Equals("0")) {
    #MPI must be removed from PATH
    $path = [Environment]::GetEnvironmentVariable('path', 'Machine')
    $path = ($path.Split(';') | Where-Object { $_ -ne 'C:\Program Files\Microsoft MPI\Bin' }) -join ';'
    [System.Environment]::SetEnvironmentVariable("path", $path,'Machine')
}

if ($defaultEnv[5].Equals("0")) {
    Write-Output "Removing CUDNN"
    [Environment]::SetEnvironmentVariable('CUDNN', '', [EnvironmentVariableTarget]::Machine)
}

if ($defaultEnv[6].Equals("0")) {
    Write-Output "Removing TRT"
    [Environment]::SetEnvironmentVariable('TRT', '', [EnvironmentVariableTarget]::Machine)
}
