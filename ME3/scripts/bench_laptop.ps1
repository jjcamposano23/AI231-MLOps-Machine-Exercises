# Laptop side of King's fd-benchmark (Iris runs on the Pi). Run from ME3:
#   powershell -ExecutionPolicy Bypass -File scripts\bench_laptop.ps1 -Student 2025xxxxx [-Rounds 1]
# First time it installs fdbench and downloads the datasets (~30 MB). The firewall step needs an ADMIN PowerShell once:
#   powershell -ExecutionPolicy Bypass -File fd-benchmark\scripts\open_firewall_windows.ps1
param([Parameter(Mandatory = $true)][string]$Student, [int]$Rounds = 1)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\fd-benchmark")

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1
}
if (-not (Test-Path "datasets\manifest.json")) {
    .\.venv\Scripts\python tools\fetch_datasets.py
}

$models = "ASR=Deepgram nova-3 via asr-mcp,LLM=openai/gpt-4.1-mini via OpenRouter,TTS=Deepgram aura-2-pandora via tts-mcp,VAD=Silero v5 + Smart Turn v3,AEC=WebRTC APM"
Write-Host "Now start Iris on the Pi with:  ssh jjcamposano@vcm.local 'bash ~/ME3/scripts/bench_pi.sh'"
.\.venv\Scripts\python -m fdbench run --student $Student --rounds $Rounds --connect direct --where remote `
    --judge-provider none --yes --recordings k --cooling "heatsink" --models $models
