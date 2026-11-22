# Security policy

## Supported versions

bigleague is pre-1.0. Security fixes go into the latest released version;
older releases are not patched.

| Package | Supported |
| --- | --- |
| `bigleague` (PyPI) | latest `0.x` |

## Reporting a vulnerability

Please **do not open a public issue** for a suspected vulnerability.

Report it privately through GitHub:
<https://github.com/klobucar/bigleague/security/advisories/new>

Include the affected version, what you observed, and a minimal reproduction
(the call, its arguments, and any environment variables). If you can, say
whether it needs attacker-controlled input to trigger.

You can expect an acknowledgement within a few days. This is a small project
maintained by one person, so there is no formal SLA, but confirmed issues are
fixed and released as a patch version with an advisory and a `CHANGELOG.md`
entry. Reporters are credited unless they ask otherwise.

## What counts

In scope:

- **Path traversal or file overwrite through the cache.** Cache keys are built
  from caller input (team codes, player ids, dates). A key that resolves
  outside the cache directory is a vulnerability.
- **Request forgery through caller input.** Input that changes the host, path
  or credentials of a request, rather than only its documented parameters.
- **Unsafe handling of data fetched from a provider.** A crafted response from
  a data source that crashes the interpreter, executes code, or causes
  unbounded work or memory use in bigleague's own parsing.
- **Supply-chain issues** in the release process, the workflows, or the
  pinned Chadwick register fetch.

Out of scope, by design:

- **A local attacker who can already write to the cache directory.** The cache
  (`~/.bigleague/cache` or `BIGLEAGUE_CACHE_DIR`) is trusted: it holds parquet
  files that polars reads back. Keep it writable only by you.
- **Provider behavior.** Baseball Savant, FanGraphs and the MLB Stats API can
  rate-limit, block or change their responses at any time. A
  `FangraphsBlockedError` is expected, not a vulnerability.
- **Vulnerabilities in dependencies** such as polars, httpx, curl_cffi or
  matplotlib. Report those upstream; if bigleague's use of one makes it
  reachable, tell us.
- **Dev-only dependencies** that do not ship in the published package.
- **Terms-of-use questions** about the data providers. See the README.
