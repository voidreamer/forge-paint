param(
  [Parameter(Mandatory = $true)]
  [string]$UsdInstallDir,

  [Parameter(Mandatory = $true)]
  [string]$OutputDir
)

$ErrorActionPreference = "Stop"

function Find-DelightRoot {
  if ($env:DELIGHT -and (Test-Path $env:DELIGHT)) {
    return (Resolve-Path $env:DELIGHT).Path
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

  throw @"
hdNSI packaging requires 3Delight.
Set DELIGHT on a self-hosted runner, or add a DELIGHT_WINDOWS_ARCHIVE_URL
repository secret that points at a zip containing a 3Delight install tree.
"@
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

function Find-HdNsiPluginRoot {
  param([string]$BuildDir)

  $plugInfo = Get-ChildItem -Path $BuildDir -Recurse -Filter plugInfo.json |
    Where-Object { $_.FullName -match 'hdNSI' } |
    Select-Object -First 1
  if (-not $plugInfo) {
    throw "Could not find hdNSI plugInfo.json under $BuildDir"
  }

  # HydraNSI's README documents builddir/output/hdNSI/resources as the
  # PXR_PLUGINPATH_NAME entry, so the package root is one level above
  # resources.
  if ($plugInfo.Directory.Name -eq "resources") {
    return (Split-Path $plugInfo.Directory.FullName -Parent)
  }
  return $plugInfo.Directory.FullName
}

$delightRoot = Find-DelightRoot
$pxrDir = Find-PxrConfig -Root $UsdInstallDir

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

cmake -S $src -B $build -G "Visual Studio 17 2022" -A x64 `
  -Dpxr_DIR="$pxrDir"
cmake --build $build --config Release --parallel

$pluginRoot = Find-HdNsiPluginRoot -BuildDir $build
$dest = Join-Path $OutputDir "hdNSI"
New-Item -ItemType Directory -Path $dest -Force | Out-Null
Copy-Item -Recurse (Join-Path $pluginRoot "*") $dest

$plugInfo = Get-ChildItem -Path $dest -Recurse -Filter plugInfo.json |
  Select-Object -First 1
if (-not $plugInfo) {
  throw "hdNSI package is missing plugInfo.json"
}

Write-Host "Packaged hdNSI from $pluginRoot to $dest"
