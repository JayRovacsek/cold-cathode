# halogen-healthcheck — per-binary contract spec

> **Status:** frozen contract for `cc-halogen-healthcheck` (Phase 3/4).
> **Source artifact:** `/nix/store/fn6y2w20kbgjd8da8pwxbbc3gvmiycpy-halogen-flash-server-0.18.21/bin/halogen-healthcheck`
> **Provenance:** halogen-flash-server v0.18.21 (see `full-plan.md` Appendix B).
> **Evidence:** none required — this is a **plain-source contract** (Phase 2
> transcription, like the launcher). The script itself is the evidence; this
> spec embeds it verbatim below. No `docs/analysis/evidence/halogen-healthcheck/`
> directory is produced (no REA was run; see §Method).

## Artifact identity

| Fact | Value |
|---|---|
| Format | **shell script** — **not an ELF binary** (shebang `#!/nix/store/nnm9gdr70ihc6zc9298mzz8jpcgaciql-bash-5.3p20/bin/bash`, nixpkgs bash 5.3p20) |
| Size | 937 bytes (35 lines) |
| SHA-256 | `e9d2f67c73823b90eea705608388f64df6d2fb4536db4c0de8801b22d4ac3db9` |
| File mode | `0555` (`-r-xr-xr-x`, nix-store default) |
| Second header line | `# halogen-flash-server: source shipped as-is, comments stripped. The README is the documentation.` |

> **Plan correction (for the decision log):** `full-plan.md` R4 groups
> `halogen-healthcheck` with the three native binaries ("the four upstream
> binaries (`flash_serve`, `halogen-tools`, `halogen-npu`,
> `halogen-healthcheck`)"), which reads as four ELF targets. In v0.18.21 it
> is a 937-byte bash script shipped in the same `$out/bin/`. Phase 2
> therefore treats it as a plain-source transcription (per §4 Phase 2's
> launcher rule extended to this artifact), not an REA target.

## Entry points and modes

One entry point: `$out/bin/halogen-healthcheck`. It takes a single optional
positional argument (`argv[1]`, default `auto`); there are no flags — any
non-empty `argv[1]` is matched as a mode word:

| Mode (`argv[1]`) | Behaviour |
|---|---|
| `engine` | TCP-connect to `127.0.0.1:$HALOGEN_PORT`, send `PING\n`, require the reply line to equal `PONG` |
| `api` | HTTP `GET http://127.0.0.1:$HALOGEN_API_PORT/health` via `python3` + `urllib` |
| `auto` (default) | Bare TCP-connect probe to the **API** port first; if connectable → run `api`, else → run `engine` |
| anything else | `usage: halogen-healthcheck [auto|engine|api]` → **stderr**, exit **2** |

The usage string is the literal text `halogen-healthcheck` (hard-coded, not
`$0`), so contract tests need no argv[0] normalisation for this binary
(unlike `flash_serve`, whose usage embeds `argv[0]`).

## Environment variables

| Variable | Default | Used by | Notes |
|---|---|---|---|
| `HALOGEN_PORT` | `8730` | `engine`; `auto` fallback | engine port |
| `HALOGEN_API_PORT` | `8731` | `api`; `auto` probe | API port |
| `HALOGEN_ENGINE_PING_S` | `30` | `engine`; `api` | PONG read timeout (`read -t "$WAIT"`); the same value is reused as the urllib request timeout in `api` mode |

The script runs `set -u` (no `set -e`): the three variables above are always
defined via `${VAR:-default}`, so an unset environment is fine, but a
*reference to any other unset variable* would abort the script (not
reachable in practice). `HALOGEN_ENGINE_PING_S` must be an integer accepted
by bash's `read -t`; a non-numeric value makes `read` fail → exit 1.

## Behaviour contract (verbatim script)

```bash
#!/nix/store/nnm9gdr70ihc6zc9298mzz8jpcgaciql-bash-5.3p20/bin/bash
# halogen-flash-server: source shipped as-is, comments stripped. The README is the documentation.
set -u
ENG_PORT="${HALOGEN_PORT:-8730}"
API_PORT="${HALOGEN_API_PORT:-8731}"
WAIT="${HALOGEN_ENGINE_PING_S:-30}"

engine_ok() {

  ( exec 3<>"/dev/tcp/127.0.0.1/$ENG_PORT" || exit 1
    printf 'PING\n' >&3 || exit 1
    read -r -t "$WAIT" reply <&3 || exit 1
    [ "$reply" = "PONG" ] ) 2>/dev/null
}

api_ok() {

  python3 -c "
import sys, urllib.request
try:
    urllib.request.urlopen('http://127.0.0.1:$API_PORT/health', timeout=$WAIT)
except Exception as e:
    print(e, file=sys.stderr); sys.exit(1)
" 2>/dev/null
}

case "${1:-auto}" in
  engine) engine_ok ;;
  api)    api_ok ;;
  auto)

    if (exec 3<>"/dev/tcp/127.0.0.1/$API_PORT") 2>/dev/null; then api_ok
    else engine_ok; fi ;;
  *) echo "usage: halogen-healthcheck [auto|engine|api]" >&2; exit 2 ;;
esac
```

### `engine` mode (`engine_ok`)

- Uses the bash `/dev/tcp/<host>/<port>` pseudo-device (requires bash built
  with network redirections; the nix bash 5.3p20 referenced by the shebang
  has them). A plain `exec 3<>` is a TCP **connect** — success means the
  port is listening.
- Protocol: write `PING\n` to the socket, read one line (`read -r`, ≤
  `WAIT` seconds), and require it to equal exactly `PONG` (byte-exact,
  trailing CR would fail the `[ "$reply" = "PONG" ]` test).
- The whole probe runs in a subshell with `2>/dev/null`: **no output on
  either path** (bash's own connect/read errors are suppressed).
- Exit status: `0` when `PONG` is received; `1` on connect failure, write
  failure, read timeout, or a reply that is not `PONG` (each failure path is
  `|| exit 1` / the failing `[` test).

### `api` mode (`api_ok`)

- Shells out to `python3 -c` (a bare `python3` lookup on `PATH` — the
  script does **not** use the wrapper's `_hg_python`-style store path) and
  performs `urllib.request.urlopen('http://127.0.0.1:<API_PORT>/health',
  timeout=<WAIT>)`.
- On any exception the inline program prints the error to stderr and exits
  1 — but the invocation's stderr is redirected to `/dev/null`, so **a
  failed request produces no visible output**.
- `urlopen` raises on HTTP status ≥ 400 (HTTPError), so a non-2xx `/health`
  response counts as unhealthy.
- Exit status is python3's status: `0` success, `1` request failed, and
  **`127` when `python3` is not on `PATH`** (bash command-not-found;
  observed on this host — see §Ground truth). The Rust port must decide
  whether to reproduce the 127-by-shelling-out or implement the probe
  natively (see §Open questions).

### `auto` mode

- Probes the **API** port with a bare TCP connect (`exec
  3<>"/dev/tcp/127.0.0.1/$API_PORT"`, no HTTP request, stderr suppressed).
- Connect succeeds → run `api_ok`; connect fails → run `engine_ok`. The
  probe target is the API port even when falling back to the engine check.

## Exit codes

| Code | Meaning | Observed via |
|---|---|---|
| 0 | healthy | `engine` PONG; `api` 2xx response; `auto` via either path |
| 1 | unhealthy | `engine`: no TCP connect / write failure / PONG timeout / wrong reply; `api`: request failed (connection refused, timeout, HTTP ≥ 400) |
| 2 | usage error | unrecognised mode word (e.g. `--help`, `bogus`, `AUTO`) |
| 127 | environment failure | `api` mode when `python3` is absent from `PATH` (bash command-not-found status) |

## Exact output strings

| Context | Stream | Exact text |
|---|---|---|
| unknown mode | stderr | `usage: halogen-healthcheck [auto|engine|api]` |
| any success | — | *(no output)* |
| engine failure | — | *(no output — stderr suppressed)* |
| api failure | — | *(no output — the Python traceback/error is printed to the suppressed stderr)* |

## Ground truth (this host, 2026-11-03)

Host state at capture time: TCP `127.0.0.1:8730` **open** (an engine
answering the PING/PONG protocol — a manual probe returned `PONG`);
`127.0.0.1:8731` **closed**; `python3` **not on `PATH`** (outside a nix
develop shell); `/models` absent.

| Invocation | stdout | stderr | exit |
|---|---|---|---|
| `halogen-healthcheck` (auto) | *(empty)* | *(empty)* | `0` (auto probed 8731 → closed → fell back to `engine` → PONG) |
| `halogen-healthcheck engine` | *(empty)* | *(empty)* | `0` |
| `halogen-healthcheck api` | *(empty)* | *(empty)* | `127` (python3 absent; the command-not-found message is suppressed by `2>/dev/null`) |
| `halogen-healthcheck bogus` | *(empty)* | `usage: halogen-healthcheck [auto|engine|api]` | `2` |
| manual socket probe: `printf 'PING\n' >&3; read -r reply <&3` on 8730 | `reply=[PONG]` | — | 0 |

## Implementation notes for `cc-halogen-healthcheck`

1. Implement the engine probe natively (TCP connect → write `PING\n` → read
   a line with a timeout → compare to `PONG`); do not shell out to bash.
2. Reproduce the exact usage string and the `argv[1]` defaulting
   (`${1:-auto}` semantics: empty/absent → `auto`).
3. Reproduce the `auto` dispatch order (API-port connect probe first, then
   engine) — including that the probe uses the API port, not the engine port.
4. Silence all diagnostic output on failure paths (the shell script emits
   nothing on failure; the Rust port must likewise print nothing).
5. The `WAIT` value gates both the PONG read timeout and (in the upstream
   script) the HTTP request timeout.

## Open questions (escalate to the orchestrator)

1. **R4 wording** — the plan's "four upstream binaries" phrasing implies an
   ELF target; the artifact is a shell script (see §Artifact identity). The
   orchestrator will record this in the decision log.
2. **`api` mode without `python3`:** the shell script exits **127** when
   `python3` is absent (observed). If `cc-halogen-healthcheck` implements
   the HTTP probe natively, 127 becomes unreachable and the port diverges
   from the reference on hosts without `python3`. Options: (a) shell out to
   `python3` exactly like the script (byte-compatible, keeps 127), or
   (b) native HTTP probe (cleaner, diverges only in the python3-absent
   case). Recommendation: (a) for contract fidelity, since the matrix can
   pin `PATH` to control it.
3. **Evidence directory:** confirm no `docs/analysis/evidence/halogen-healthcheck/`
   is expected (R9 speaks of REA evidence; this contract needs none).

## Method and limitations

- The contract is a direct transcription of the 937-byte script (no REA, no
  decompilation — the source *is* the artifact).
- Runtime probes were run on this host on 2026-11-03 against a live engine
  on port 8730; the `api` success path (HTTP 200) could not be observed here
  because port 8731 is closed and `python3` is absent outside a nix shell —
  the `api` contract is therefore transcribed from source plus the observed
  127 case.
- The script's behaviour on a host where `/dev/tcp` redirections are
  unavailable (non-bash shells) is out of scope: the shebang pins nix bash.
