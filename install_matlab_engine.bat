@echo off
REM Install MATLAB Engine for Python
REM This script requires administrator privileges

echo Installing MATLAB Engine for Python...
echo.

cd "C:\Program Files\MATLAB\R2025b\extern\engines\python"
python setup.py install

echo.
echo Installation complete!
echo.
pause

