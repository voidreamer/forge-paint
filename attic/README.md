# attic/

Code that has been removed from the build but is worth keeping nearby
for future reference. Files here are NOT compiled — they live outside
`src/` and are never declared as modules. They survive in-tree (rather
than only in git history) so a curious reader can find the prior
attempt without spelunking `git log`.

## Current contents

- `forge-paint-windows.bat` — shell launcher the Windows zips shipped
  as `forge-paint.bat`. It existed to set PATH / PXR_PLUGINPATH_NAME /
  DELIGHT *before* the EXE started, back when direct launches couldn't
  self-configure (USD captures its plugin paths pre-`main()`, and the
  bundle root double-loaded the USD DLLs). Retired once the root EXE
  began relaunching `usd\lib\forge-paint.exe` with the bundle
  environment applied at process start — see
  `relaunch_from_bundled_usd_lib` in `src/main.rs`. Shell-set env vars
  are still honored (merged, not replaced), so the bat's debugging use
  case needs no script.
