@echo off
setlocal EnableExtensions

set "FORGE_PAINT_DIR=%~dp0"

rem DLLs imported by forge-paint.exe must be visible before the app can
rem run its own startup environment setup.
set "PATH=%FORGE_PAINT_DIR%usd\bin;%FORGE_PAINT_DIR%usd\lib;%PATH%"

set "FORGE_USD_PLUGINS=%FORGE_PAINT_DIR%usd\plugin\usd;%FORGE_PAINT_DIR%usd\lib\usd"
if exist "%FORGE_PAINT_DIR%plugins\usd" (
  set "FORGE_USD_PLUGINS=%FORGE_PAINT_DIR%plugins\usd;%FORGE_USD_PLUGINS%"
)
if exist "%FORGE_PAINT_DIR%plugins\usd\hdNSI" (
  set "FORGE_USD_PLUGINS=%FORGE_PAINT_DIR%plugins\usd\hdNSI;%FORGE_USD_PLUGINS%"
)
if exist "%FORGE_PAINT_DIR%plugins\usd\hdNSI\resources" (
  set "FORGE_USD_PLUGINS=%FORGE_PAINT_DIR%plugins\usd\hdNSI\resources;%FORGE_USD_PLUGINS%"
)

if defined PXR_PLUGINPATH_NAME (
  set "PXR_PLUGINPATH_NAME=%FORGE_USD_PLUGINS%;%PXR_PLUGINPATH_NAME%"
) else (
  set "PXR_PLUGINPATH_NAME=%FORGE_USD_PLUGINS%"
)

set "FORGE_PAINT_FOUND_DELIGHT="
call :use_delight "%FORGE_PAINT_3DELIGHT_DIR%"
if defined FORGE_PAINT_FOUND_DELIGHT goto launch
call :use_delight "%DELIGHT%"
if defined FORGE_PAINT_FOUND_DELIGHT goto launch
call :use_delight "%FORGE_PAINT_DIR%3Delight"
if defined FORGE_PAINT_FOUND_DELIGHT goto launch
call :use_delight "%ProgramFiles%\3Delight"
if defined FORGE_PAINT_FOUND_DELIGHT goto launch
call :use_delight "%ProgramFiles%\3DelightNSI"
if defined FORGE_PAINT_FOUND_DELIGHT goto launch
call :use_delight "%ProgramFiles(x86)%\3Delight"
if defined FORGE_PAINT_FOUND_DELIGHT goto launch
call :use_delight "%ProgramFiles(x86)%\3DelightNSI"

:launch
start "" "%FORGE_PAINT_DIR%forge-paint.exe" %*
exit /b %ERRORLEVEL%

:use_delight
if "%~1"=="" exit /b 0
if exist "%~1\bin\renderdl.exe" (
  set "DELIGHT=%~1"
  set "PATH=%~1\bin;%~1\lib;%PATH%"
  set "FORGE_PAINT_FOUND_DELIGHT=1"
)
exit /b 0
