@ECHO OFF
SETLOCAL EnableExtensions
REM =====================================================================
REM  PianoPlayer - Windows launcher
REM
REM   Double click this file            -> opens the GUI
REM   Or run it with the usual options  -> e.g.
REM        pianoplayer.bat scores\bach_invention4.xml -v
REM
REM   Python and PianoPlayer are detected automatically,
REM   so there is nothing to edit in this file.
REM   (Copy it to your desktop if you like: it still finds the
REM    installed PianoPlayer; only scores\... relative paths then
REM    have to be written in full.)
REM =====================================================================

REM ---------- 1) Python: the py launcher first, then python on PATH ----
SET "PY="
WHERE py >NUL 2>NUL
IF NOT ERRORLEVEL 1 SET "PY=py -3"
IF NOT DEFINED PY (
    WHERE python >NUL 2>NUL
    IF NOT ERRORLEVEL 1 SET "PY=python"
)
IF NOT DEFINED PY GOTO :NOPYTHON

REM ---------- 2) the pianoplayer entry point ---------------------------
SET "SCRIPTS="
FOR /F "delims=" %%I IN ('%PY% -c "import sysconfig;print(sysconfig.get_path('scripts'))" 2^>NUL') DO SET "SCRIPTS=%%I"

SET "PP="
IF EXIST "%~dp0bin\pianoplayer" (
    REM we are inside the source folder: use the up-to-date copy
    SET "PP=%~dp0bin\pianoplayer"
    SET "PYTHONPATH=%~dp0;%PYTHONPATH%"
)
IF NOT DEFINED PP IF DEFINED SCRIPTS IF EXIST "%SCRIPTS%\pianoplayer.exe" SET "PP=%SCRIPTS%\pianoplayer.exe"
IF NOT DEFINED PP IF DEFINED SCRIPTS IF EXIST "%SCRIPTS%\pianoplayer" SET "PP=%SCRIPTS%\pianoplayer"
IF NOT DEFINED PP GOTO :NOTINSTALLED

REM ---------- 3) run ---------------------------------------------------
ECHO Starting PianoPlayer...
ECHO.

IF "%PP:~-4%"==".exe" (
    "%PP%" %*
) ELSE (
    %PY% "%PP%" %*
)
SET "RC=%ERRORLEVEL%"

ECHO.
IF NOT "%RC%"=="0" (
    ECHO PianoPlayer stopped with an error.
    PAUSE
)
ENDLOCAL & EXIT /B %RC%

REM ---------------------------------------------------------------------
:NOPYTHON
ECHO.
ECHO Python 3 was not found on this computer.
ECHO Install it from https://www.python.org/downloads/ and tick
ECHO "Add python.exe to PATH" during the installation.
ECHO.
PAUSE
EXIT /B 1

:NOTINSTALLED
ECHO.
ECHO PianoPlayer is not installed for this Python.
ECHO Open the pianoplayer folder and run:
ECHO.
ECHO     pip install -e .
ECHO.
PAUSE
EXIT /B 1
