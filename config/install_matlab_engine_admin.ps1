# PowerShell script to install MATLAB Engine (requires Administrator)
# Right-click and select "Run with PowerShell" as Administrator

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "MATLAB Engine Installation Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as admin
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "[ERROR] This script requires Administrator privileges!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please:" -ForegroundColor Yellow
    Write-Host "1. Right-click this file" -ForegroundColor Yellow
    Write-Host "2. Select 'Run with PowerShell'" -ForegroundColor Yellow
    Write-Host "3. Click 'Yes' when prompted for admin access" -ForegroundColor Yellow
    Write-Host ""
    pause
    exit 1
}

Write-Host "[OK] Running with Administrator privileges" -ForegroundColor Green
Write-Host ""

# MATLAB path
$matlabPath = "C:\Program Files\MATLAB\R2025b\extern\engines\python"

if (-not (Test-Path $matlabPath)) {
    Write-Host "[ERROR] MATLAB Engine directory not found at: $matlabPath" -ForegroundColor Red
    Write-Host "Please verify MATLAB R2025b is installed." -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "Installing MATLAB Engine for Python..." -ForegroundColor Cyan
Write-Host "MATLAB Path: $matlabPath" -ForegroundColor Gray
Write-Host ""

# Change to MATLAB directory and install
Set-Location $matlabPath

try {
    python setup.py install
    Write-Host ""
    Write-Host "[SUCCESS] MATLAB Engine installed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Verifying installation..." -ForegroundColor Cyan
    python -c "import matlab.engine; print('MATLAB Engine version:', matlab.engine.__version__)"
    Write-Host ""
    Write-Host "[SUCCESS] Installation verified!" -ForegroundColor Green
} catch {
    Write-Host ""
    Write-Host "[ERROR] Installation failed: $_" -ForegroundColor Red
    Write-Host ""
    pause
    exit 1
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Run: python test_matlab_setup.py" -ForegroundColor Yellow
Write-Host "2. Run: python test_cell1.py" -ForegroundColor Yellow
Write-Host ""
pause

