$runnerName = (hostname.exe).Trim()
.\actions-runner\config.cmd --unattended --replace --url https://github.com/${env:RUNNER_REPO} --pat $env:RUNNER_PAT --runnergroup $env:RUNNER_GROUP --labels $env:RUNNER_LABELS --work $env:RUNNER_WORKDIR --name $runnerName
.\actions-runner\run.cmd