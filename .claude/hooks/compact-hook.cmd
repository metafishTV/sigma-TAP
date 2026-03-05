: << 'CMDBLOCK'
@echo off
REM Cross-platform polyglot wrapper for compact_hook.py.
REM On Windows: cmd.exe runs the batch portion.
REM On Unix: the shell interprets this as a bash script.
REM
REM Usage: compact-hook.cmd <pre-compact|post-compact>

if "%~1"=="" (
    echo compact-hook.cmd: missing subcommand >&2
    exit /b 1
)

set "HOOK_SCRIPT=%USERPROFILE%\.claude\skills\buffer\scripts\compact_hook.py"

REM Try python on PATH first
where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    python "%HOOK_SCRIPT%" %*
    exit /b %ERRORLEVEL%
)

REM Try known Python 3.12 install
if exist "C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" (
    "C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" "%HOOK_SCRIPT%" %*
    exit /b %ERRORLEVEL%
)

REM No Python found
echo compact-hook.cmd: python not found >&2
exit /b 0
CMDBLOCK

# Unix: run compact_hook.py directly
exec python "$HOME/.claude/skills/buffer/scripts/compact_hook.py" "$@"
