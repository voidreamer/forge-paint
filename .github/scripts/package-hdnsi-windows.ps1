param(
  [Parameter(Mandatory = $true)]
  [string]$UsdInstallDir,

  [Parameter(Mandatory = $true)]
  [string]$OutputDir,

  [switch]$SkipIfMissing
)

$ErrorActionPreference = "Stop"

function Find-DelightRoot {
  if ($env:DELIGHT -and (Test-Path $env:DELIGHT)) {
    return (Resolve-Path $env:DELIGHT).Path
  }

  if ($env:DELIGHT_WINDOWS_ARCHIVE_PATH -and (Test-Path $env:DELIGHT_WINDOWS_ARCHIVE_PATH)) {
    $extract = Join-Path $env:RUNNER_TEMP "3delight-repo"
    New-Item -ItemType Directory -Path $extract -Force | Out-Null
    Expand-Archive -Path $env:DELIGHT_WINDOWS_ARCHIVE_PATH -DestinationPath $extract -Force

    $renderdl = Get-ChildItem -Path $extract -Recurse -Filter renderdl.exe |
      Select-Object -First 1
    if ($renderdl) {
      return (Split-Path (Split-Path $renderdl.FullName -Parent) -Parent)
    }

    throw "DELIGHT_WINDOWS_ARCHIVE_PATH exists but does not contain bin\renderdl.exe: $env:DELIGHT_WINDOWS_ARCHIVE_PATH"
  }

  if ($env:DELIGHT_WINDOWS_ARCHIVE_URL) {
    $archive = Join-Path $env:RUNNER_TEMP "3delight.zip"
    $extract = Join-Path $env:RUNNER_TEMP "3delight"
    New-Item -ItemType Directory -Path $extract -Force | Out-Null
    Invoke-WebRequest -Uri $env:DELIGHT_WINDOWS_ARCHIVE_URL -OutFile $archive
    Expand-Archive -Path $archive -DestinationPath $extract -Force

    $renderdl = Get-ChildItem -Path $extract -Recurse -Filter renderdl.exe |
      Select-Object -First 1
    if ($renderdl) {
      return (Split-Path (Split-Path $renderdl.FullName -Parent) -Parent)
    }
  }

  $common = @(
    "$env:ProgramFiles\3Delight",
    "${env:ProgramFiles(x86)}\3Delight"
  )
  foreach ($path in $common) {
    if ($path -and (Test-Path (Join-Path $path "bin\renderdl.exe"))) {
      return $path
    }
  }

  $message = @"
hdNSI packaging requires 3Delight.
Add a DELIGHT_WINDOWS_ARCHIVE_URL repository secret that points at a zip
containing a 3Delight install tree, or (on a self-hosted runner) set
DELIGHT to an existing install / DELIGHT_WINDOWS_ARCHIVE_PATH to a local
archive. The proprietary SDK zip must not be committed to this public repo.
"@
  if ($SkipIfMissing) {
    Write-Warning $message
    return $null
  }
  throw $message
}

function Find-PxrConfig {
  param([string]$Root)

  $candidates = @(
    (Join-Path $Root "lib\cmake\pxr"),
    (Join-Path $Root "cmake\pxr")
  )
  foreach ($path in $candidates) {
    if (Test-Path (Join-Path $path "pxrConfig.cmake")) {
      return $path
    }
  }

  $config = Get-ChildItem -Path $Root -Recurse -Filter pxrConfig.cmake |
    Select-Object -First 1
  if ($config) {
    return $config.Directory.FullName
  }

  throw "Could not find pxrConfig.cmake under $Root"
}

$delightRoot = Find-DelightRoot
if (-not $delightRoot) {
  Write-Host "Skipping optional hdNSI packaging; no 3Delight install or archive was found."
  exit 0
}
$pxrDir = Find-PxrConfig -Root $UsdInstallDir

# Fail fast with a clear message if the 3Delight archive is a renderer-
# only runtime (no compiler). hdNSI's osl/CMakeLists.txt compiles its
# bundled .osl shaders with `3Delight::oslc`, and Find3Delight.cmake
# points that target at <root>/bin/oslc.exe WITHOUT checking it exists
# — so a missing oslc otherwise surfaces as an opaque MSBuild "exit
# code 9009" (command not found) deep in the per-shader build rules.
$oslc = Join-Path $delightRoot "bin\oslc.exe"
if (-not (Test-Path $oslc)) {
  throw @"
3Delight archive at $delightRoot has no bin\oslc.exe.
hdNSI needs the OSL compiler to build its shader resources. The archive
must be a 3Delight install that includes bin\oslc.exe plus the osl\ header
dir (stdosl.h is auto-found at ..\osl relative to oslc). A renderer-only
runtime (renderdl.exe + DLLs) is not sufficient.
"@
}
# stdosl.h is auto-resolved by oslc relative to its own exe (<root>\osl),
# so the osl header dir must be a sibling of bin. Warn early if it's
# absent — the shader compiles would otherwise fail with "stdosl.h not
# found" once MSBuild reaches them.
if (-not (Test-Path (Join-Path $delightRoot "osl\stdosl.h"))) {
  Write-Warning "3Delight archive has no osl\stdosl.h next to bin\oslc.exe; OSL shader compilation may fail to find the standard library."
}

$env:DELIGHT = $delightRoot
$env:PATH = "$delightRoot\bin;$delightRoot\lib;$env:PATH"

$src = Join-Path $env:GITHUB_WORKSPACE "HydraNSI"
$build = Join-Path $env:GITHUB_WORKSPACE "HydraNSI-build"

if (Test-Path $src) {
  Remove-Item -Recurse -Force $src
}
if (Test-Path $build) {
  Remove-Item -Recurse -Force $build
}
if (Test-Path $OutputDir) {
  Remove-Item -Recurse -Force $OutputDir
}

git clone --depth 1 https://gitlab.com/3Delight/HydraNSI.git $src

# pxrConfig.cmake's find_package(OpenSubdiv) walks CMake's standard
# search paths — `-Dpxr_DIR` alone isn't enough because OpenSubdiv
# isn't a sub-dep of the pxr package, just a sibling installed by
# build_usd.py into the same prefix. CMAKE_PREFIX_PATH wires the
# whole USD install in so find_package picks up OpenSubdivConfig +
# headers + libs. OpenSubdiv_ROOT covers older FindOpenSubdiv
# modules that don't honour CMAKE_PREFIX_PATH.
cmake -S $src -B $build -G "Visual Studio 17 2022" -A x64 `
  -Dpxr_DIR="$pxrDir" `
  -DCMAKE_PREFIX_PATH="$UsdInstallDir" `
  -DOpenSubdiv_ROOT="$UsdInstallDir"
if ($LASTEXITCODE -ne 0) {
  throw "hdNSI CMake configure failed (exit $LASTEXITCODE). Check the log above for missing dependencies (typically OpenSubdiv / TBB)."
}
cmake --build $build --config Release --parallel
if ($LASTEXITCODE -ne 0) {
  throw "hdNSI build failed (exit $LASTEXITCODE)."
}

# Run the actual install step and stage from the installed prefix. The
# install layout is what the generated plugInfo.json's relative paths
# assume (Root: "..", LibraryPath: "hdNSI.dll", ResourcePath:
# "resources"):
#   <prefix>\hdNSI\hdNSI.dll
#   <prefix>\hdNSI\resources\plugInfo.json
#   <prefix>\hdNSI\resources\osl\*.oso
#   <prefix>\usdNSI\...           (when HydraNSI builds the schema lib)
# An earlier revision of this script skipped the install and staged the
# first plugInfo.json found in the raw CMake BUILD tree. That copy sits
# one directory too high (next to Release\hdNSI.dll instead of inside
# resources\), so USD registered HdNSIRendererPlugin from its metadata
# but resolved LibraryPath to a nonexistent plugins\usd\hdNSI.dll and
# the delegate failed to load with "The specified module could not be
# found".
$installPrefix = Join-Path $env:GITHUB_WORKSPACE "HydraNSI-install"
if (Test-Path $installPrefix) {
  Remove-Item -Recurse -Force $installPrefix
}
cmake --install $build --config Release --prefix $installPrefix
if ($LASTEXITCODE -ne 0) {
  throw "hdNSI install failed (exit $LASTEXITCODE)."
}

foreach ($required in @(
  (Join-Path $installPrefix "hdNSI\hdNSI.dll"),
  (Join-Path $installPrefix "hdNSI\resources\plugInfo.json")
)) {
  if (-not (Test-Path $required)) {
    throw "hdNSI install layout is missing $required"
  }
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
Copy-Item -Recurse (Join-Path $installPrefix "*") $OutputDir

# The release bundle ships the Hydra delegate only. 3Delight itself is
# installed by the user and discovered by forge-paint at startup, so
# keep renderer runtime files out of the artifact if CMake copied any.
$runtimeNames = @(
  "renderdl.exe",
  "i-display.exe",
  "oslc.exe",
  "tdlmake.exe",
  "3Delight*.dll",
  "lib3delight*.dll",
  "libnsi*.dll"
)
foreach ($name in $runtimeNames) {
  Get-ChildItem -Path $OutputDir -Recurse -Filter $name -ErrorAction SilentlyContinue |
    Remove-Item -Force -ErrorAction SilentlyContinue
}

# Link-time leftovers the install rules copy along (hdNSI.lib is
# installed OPTIONAL next to the DLL). Runtime needs neither.
foreach ($pattern in @("*.lib", "*.pdb")) {
  Get-ChildItem -Path $OutputDir -Recurse -Filter $pattern -ErrorAction SilentlyContinue |
    Remove-Item -Force -ErrorAction SilentlyContinue
}

Write-Host "Packaged installed hdNSI layout from $installPrefix to $OutputDir"
