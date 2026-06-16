$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
$ExePath = Join-Path $ProjectRoot "dist\FSS_Measurement.exe"

$Running = Get-Process -Name "FSS_Measurement" -ErrorAction SilentlyContinue
if ($Running) {
    throw "Close all running FSS_Measurement.exe windows before building."
}

Write-Host "Installing build dependencies..."
python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed."
}

python -m pip install -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) {
    throw "dependency installation failed."
}

Write-Host "Building one-file Windows executable..."
if (Test-Path $ExePath) {
    Remove-Item -LiteralPath $ExePath -Force
}
python -m PyInstaller --clean --noconfirm main.spec
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed."
}

if (-not (Test-Path $ExePath)) {
    throw "Build finished, but executable was not found: $ExePath"
}

Write-Host ""
Write-Host "Build complete:"
Write-Host $ExePath
