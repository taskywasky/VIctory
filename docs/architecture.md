# Architecture

```
 user's files (outside repo)             VIctory
 ───────────────────────────             ─────────────────────────────────────────
 game assets ──────────────────────────► victory-pipeline + victory-game
                                         (victory-formats, victory-shader)
                                                    │
                                                    ▼
                                         cache_dir: converted data
                                                    │
 game executable ─► victory-loader ────► title code (native x86-64)
                    prepare, map, bind      │ imports
                                            ▼
                                         victory-hle ── host Windows (shared APIs)
                                            │ D3D12.X
                                            ▼
                                         victory-gpu ─► host D3D12
                                            │ shaders at pipeline creation
                                            ▼
                                         victory-shader ◄─► cache_dir
```

Shaders are translated when the title creates a pipeline, from the bytes it
passes, wherever it loaded them from (executable, archive, memory).
Converting them ahead of time from files (`victory translate --cache`) is
only a warm-up: it fills the same cache, under the same key. With
`--pipelines` it also makes the host pipelines of the title's shader pack
into the pipeline library, from the creation streams the engine would pass
(`victory-game` builds them, `victory-gpu` reads them as it reads the
title's), so a run finds them under the names it gives them itself.

## Crates

| Crate | Role |
|-------|------|
| `victory-core` | config, asset-policy guard, logging |
| `victory-pe` | read-only PE32+ and import-library analysis |
| `victory-loader` | prepare, map, bind and start the title image |
| `victory-hle` | console API implementations, host pass-through policy |
| `victory-gpu` | D3D12.X objects, command lists and GPU memory → host D3D12 |
| `victory-shader` | Xbox shader containers → host shaders |
| `victory-formats` | the GPU's texture layouts (sizes, tiling, detiling), BCPack, audio codecs |
| `victory-pipeline` | converter trait, content-addressed cache |
| `victory-game` | Grand Theft Auto VI-specific formats, build identities, hooks |
| `apps/victory` | command-line front end |
| `xtask` | asset-policy check, git hooks |
| `tests/fixtures/hle-probe` | a stand-in "title" (our own code) for loader tests |

Everything except `victory-game` is platform-level and can be tested with
any Xbox Series title.

## Import resolution

For every `dll!function` the title imports, in order:

1. our implementation in `victory-hle::modules` (always wins);
2. the host's own function, if `HostPolicy` allows that DLL;
3. an export of a DLL the title ships beside its executable, mapped like
   the executable (only DLLs neither of the above serves);
4. a generated stub that names the import when called (`--unimplemented
   trap` breaks into the debugger and aborts; `zero` logs once and returns 0).

`LoadLibrary*` of a console-only DLL we implement (e.g. `xgameruntime.dll`)
returns a pseudo handle; `GetProcAddress` on it finds our functions.
`LoadLibrary*` of a title DLL returns its mapped image, mapping it first if
nothing imports it. By name, as by import, a title gets only host DLLs
`HostPolicy` passes through, and `GetProcAddress` on one returns our
override where import binding would.
`QueryApiImpl` hands out our runtime objects by provider and interface id
(`victory-hle::modules::xgameruntime` lists the areas); online areas answer
as an offline console. Game saves are files under `VICTORY_DATA_DIR`.

Title storage is a drive on the console, and titles build paths on that.
`victory-hle::modules::files` binds every path-taking title import
(kernel32 and the C/C++ runtime) and rewrites paths on a virtual drive to
the host folder behind it; persistent local storage is one such drive over
a folder under `VICTORY_DATA_DIR`. Other paths reach the host unchanged.

An exe's code reaches its static TLS as slot 0; mapped as a DLL it gets
another slot, so the loader redirects those loads (`victory-loader::tls`).
`victory run --execute` installs a crash reporter that names faults as
`module+rva` and unwinds the faulting thread's stack
(`victory-loader::crash`).

`victory census` shows this classification without running anything (it
does not map title DLLs; `victory run` without `--execute` does).

## Shaders

`victory_shader::ir::build` turns a native program into SSA, one
instruction per machine instruction. `victory_shader::structure` then
removes EXEC: lane masks become per-lane booleans, wave-wide mask tests
become per-lane branches, and vector instructions keep a guard only where
a lane can reach them with its EXEC bit clear. `victory_shader::range::build`
runs both, again where a bound on the index of a register-relative move
(`v_movrel*`) is proven on the per-lane form, so the move names only the
registers it can reach. `victory_shader::resource`
traces each descriptor and scalar memory address to the root parameter it
comes from (table, heap, root descriptor or constant). A descriptor a
waterfall loop reads from the first lane (`v_readfirstlane_b32`) is traced
to the per-lane load or choice that gives each lane its own (a mask's
`v_cndmask_b32`, or an if/else whose arms each set it: traced on the paths
the lanes take, from structurisation), and accessed with a non-uniform heap
index in one pass. `victory translate`
runs every shader it finds (containers, SGD5 files) through these and
lists what they cannot handle yet.

Compute programs that depend on their lane layout (cross-lane reads, lane
masks read as numbers, LDS shared within one wave) declare the console's
wave size. A console wave64 program on a host without 64-lane waves
(NVIDIA: `WaveLaneCountMax` 32, which `victory-gpu` reports to the
translator when it creates the device) runs *split* instead
(`victory_shader::emit::split`): it declares `WaveSize(32)`, and each
console wave runs as two host waves, its low and high half. Thread ids,
lane indices and masks stay the console's. DPP, `ds_swizzle` and
`v_permlanex16` stay within 32 lanes and need nothing else. Ballots,
`v_readlane`, `v_readfirstlane` (of a value that differs between lanes),
and whole-wave reductions and prefix sums combine the
two halves' parts through a few group-shared dwords between thread-group
barriers, which gives the console's results exactly. A barrier needs the
whole group, so that is only possible where every thread runs together:
with one console wave per group, outside any lane branch's region; with
several, before the first branch. Elsewhere each half works over its own
32 lanes, as pixel and vertex programs do over the host's wave (counts,
first lanes, prefixes and reductions of the half, console lane numbers
kept): right for aggregated atomics and scalarization loops, wrong where
the code needs one value for all 64 lanes. `v_readfirstlane` of a value
that is the same in every lane needs neither.
One-wave groups also get barriers after LDS writes and at joins, which the
console compiler leaves out for a single wave. `VICTORY_HOST_WAVE_MAX=32`
translates split on any device; the shader cache keys on the width.
