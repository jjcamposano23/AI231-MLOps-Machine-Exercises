# One-time: let this laptop SSH into the Pi with a key (you type the Pi password once).
# Usage: powershell -ExecutionPolicy Bypass -File scripts\install_ssh_key.ps1 [user@host]
param([string]$Target = "jjcamposano@vcm.local")

$pubPath = Join-Path $env:USERPROFILE ".ssh\id_ed25519.pub"
if (-not (Test-Path $pubPath)) { Write-Error "No key at $pubPath"; exit 1 }
$pub = (Get-Content -Raw $pubPath).Trim()

# tr -d '\r' strips Windows line endings that PowerShell can add when piping, which would break the key line
$pub | ssh $Target "mkdir -p ~/.ssh && chmod 700 ~/.ssh && tr -d '\r' >> ~/.ssh/authorized_keys && echo >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && echo KEY-INSTALLED"
