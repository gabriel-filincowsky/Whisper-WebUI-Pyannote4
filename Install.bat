@echo off

echo ================================================================================
echo WARNING: Windows Installation Limitations
echo ================================================================================
echo.
echo This fork requires pyannote.audio 4.x, which depends on torchcodec.
echo torchcodec has NO pip-compatible wheels for Windows.
echo.
echo Speaker diarization features will NOT work with local installation on Windows.
echo.
echo RECOMMENDED: Use Docker instead (see README.md for Docker installation).
echo.
echo ================================================================================
echo.
echo Press any key to continue with installation (other features will work)...
echo Or press Ctrl+C to cancel and use Docker instead.
pause
echo.

if not exist "%~dp0\venv\Scripts" (
    echo Creating venv...
    python -m venv venv
)
echo checked the venv folder. now installing requirements..

call "%~dp0\venv\scripts\activate"

python -m pip install -U pip
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo Requirements installation failed. please remove venv folder and run install.bat again.
    echo.
    echo NOTE: If the failure is related to torchcodec or pyannote.audio, this is expected
    echo on Windows. Please use Docker instead (see README.md).
) else (
    echo.
    echo Requirements installed successfully.
    echo.
    echo WARNING: Speaker diarization features will not work on Windows without Docker.
    echo Other features (transcription, translation, VAD, BGM separation) should work.
)
pause