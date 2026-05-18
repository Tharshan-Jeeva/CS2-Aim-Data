# Tickrate Enabler — patched build for CS:Source ServerGameDLL012

The plugin in `sourcemod/Tickrate_Enabler.so` is a patched build of [daemon32/tickrate_enabler](https://github.com/daemon32/tickrate_enabler), which is itself a fork of Didrole's original Source-engine tickrate-enabler plugin. Both upstream projects are MIT/BSD-licensed; see those repositories for full license text.

## Why this is needed

Without this plugin, srcds is hard-capped at 66.7 Hz tickrate regardless of the `-tickrate 100` launch flag — an engine limitation of vanilla CS:Source that has been present since 2007. Capturing telemetry at 66.7 Hz silently downsamples the feature space: angular jerk and any spectral component above 33 Hz become aliased or invisible. The plugin patches the in-memory tickrate cap at server load so `-tickrate 100` is honoured. The dissertation captures view-angle telemetry at the full 100 Hz to resolve high-frequency aim-trajectory components that prior CS-based behavioural studies (capped at 64–66 Hz) could not.

## The patch

The single functional change against upstream `serverplugin_empty.cpp` is to query the modern `ServerGameDLL012` interface that this srcds binary exposes, falling back to the older `ServerGameDLL010` so the same source builds against both:

```diff
-    gamedll = (IServerGameDLL*)gameServerFactory("ServerGameDLL010",NULL);
+    gamedll = (IServerGameDLL*)gameServerFactory("ServerGameDLL012",NULL);
+    if(!gamedll)
+        gamedll = (IServerGameDLL*)gameServerFactory("ServerGameDLL010",NULL);
```

The full patched file is in `serverplugin_empty.cpp`. Everything else in the upstream repo (Makefile, sourcehook objects, license files) is unchanged.

## How to rebuild

```bash
# 1. Get the upstream plugin + SDK
git clone https://github.com/daemon32/tickrate_enabler.git
git clone -b css --depth 1 https://github.com/alliedmodders/hl2sdk.git /tmp/hl2sdk-css

# 2. Apply our patch
cp serverplugin_empty.cpp tickrate_enabler/serverplugin_empty.cpp

# 3. Build (32-bit toolchain required: on Arch enable [multilib] and
#    install lib32-gcc-libs, lib32-glibc, gcc-multilib)
cd tickrate_enabler
# The Makefile expects libtier1_i486.a; the SDK ships tier1_i486.a. Symlink:
ln -s tier1_i486.a /tmp/hl2sdk-css/lib/public/linux/libtier1_i486.a
# Override the Makefile's paths and build:
make HL2SDK=/tmp/hl2sdk-css MMSDK=. ENGINE=css \
     INCLUDES="-I/tmp/hl2sdk-css/public -I/tmp/hl2sdk-css/public/tier0 -I/tmp/hl2sdk-css/public/tier1 -I." \
     LINKFLAGS="-shared -m32 -L/tmp/hl2sdk-css/lib/public/linux"
```

Output: `Tickrate_Enabler.so` (~1.7 MB, 32-bit Linux ELF).

## Install

```bash
cp Tickrate_Enabler.so   cssource_server/cstrike/addons/
cp ../Tickrate_Enabler.vdf cssource_server/cstrike/addons/
```

Restart srcds. Confirm in the startup log:

```
SV_ActivateServer: setting tickrate to 100.0
```

If you see `66.7` instead, the plugin didn't load — check the `.vdf` is in place and ldd the `.so` for missing 32-bit dependencies.
