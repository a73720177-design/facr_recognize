$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\recognition.py" @args
exit $LASTEXITCODE
