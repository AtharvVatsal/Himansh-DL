@echo off
rem scripts\run_all.bat -- Windows launcher: double-click it, or run it from cmd / PowerShell.
rem Runs scripts/run_all.sh in Git Bash (install Git for Windows first). Settings such as
rem JOBS or DATASETS can be set beforehand, e.g. in cmd:  set JOBS=8 ^&^& scripts\run_all.bat
setlocal
set "BASH="
for %%B in ("%ProgramFiles%\Git\bin\bash.exe" "%ProgramFiles(x86)%\Git\bin\bash.exe" "%LOCALAPPDATA%\Programs\Git\bin\bash.exe") do (
  if not defined BASH if exist %%B set "BASH=%%~B"
)
if not defined BASH (
  echo Git Bash not found. Install Git for Windows from https://git-scm.com/download/win and re-run.
  pause
  exit /b 1
)
"%BASH%" "%~dp0run_all.sh"
set RC=%ERRORLEVEL%
echo.
if %RC%==0 (echo Pipeline finished.) else (echo Pipeline stopped with exit code %RC% -- see results\logs\. Re-running resumes where it stopped.)
pause
exit /b %RC%
