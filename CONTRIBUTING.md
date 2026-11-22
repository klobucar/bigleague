# Contributing to bigleague

Thanks for your interest. Bug reports, docs fixes and focused pull requests
are all welcome. For anything larger than a small fix, please open an issue
first so we can agree on the approach before you spend time on it.

Security problems go through [SECURITY.md](SECURITY.md), not public issues.

## Layout

| Path | What it is |
| --- | --- |
| `bigleague/` | The library. See the layout section of the README. |
| `tests/` | The test suite. All HTTP is mocked. |
| `bench/` | A network benchmark against pybaseball. Not run in CI. |
| `docs/` | The README gallery and the script that regenerates it. |
| `.github/workflows` | CI and the Trusted-Publishing release workflow. |

## Setup

bigleague uses [uv](https://docs.astral.sh/uv/) and requires Python 3.12 or
newer.

```bash
git clone https://github.com/klobucar/bigleague
cd bigleague
uv sync --locked --extra dev
```

Use `--locked` so you test what CI tests. `pyproject.toml` sets a one-week
dependency cooldown (`exclude-newer`), so a freshly published release cannot
enter the lockfile before it has aged.

Before opening a PR, run what CI runs:

```bash
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync ty check
uv run --no-sync pytest
```

### Test with and without the bloom extra

The cache has an optional accelerator (`bouquet-bloom`), and CI runs the suite
both ways. The bloom tests skip themselves when it is absent, so touching
`bigleague/cache.py` means running both:

```bash
uv sync --locked --extra dev && uv run --no-sync pytest                 # without
uv sync --locked --extra dev --extra bloom && uv run --no-sync pytest   # with
```

`uv sync` is exact, so it removes the extra when you leave the flag off. Plain
`uv run --extra dev pytest` does not uninstall anything, which can leave
bouquet installed and make a "without bloom" run meaningless. The bloom filter
is shared across threads (the cache is used from `asyncio.to_thread`), so
keep it on `AtomicFilter`, which is thread-safe.

## Tests

- **No network.** Tests mock HTTP with `respx` (httpx) or monkey-patching
  (`curl_cffi`), and `tests/conftest.py` sets `BIGLEAGUE_NO_CACHE=1`. A test
  that reaches a real server is a bug.
- A bug fix needs a regression test that fails without the fix.
- To record a VCR cassette against a live endpoint while developing:
  `uv run pytest tests/test_savant.py --record-mode=once`. Review cassettes
  before committing them.

## Adding or changing a public function

- Export it from `bigleague/__init__.py` (and `__all__`, which is kept sorted
  by ruff).
- Add it to the API table in the README, with an example if it is a new area.
- Cover the happy path, the empty or error case, and invalid arguments.
- Be kind to the data providers. Reuse the shared client in `bigleague/http.py`
  (it owns the Savant concurrency cap and retries), and don't add code that
  raises request volume without a reason.

## Pull requests

- Keep a PR to one logical change, and include tests for new behavior or a
  fixed bug.
- Commit messages follow
  [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `ci:`, `chore(deps):` ...), matching the existing
  history.
- Add a line to `CHANGELOG.md` for any user-visible change.
- Behavior and performance claims in docs should be backed by a measurement or
  a test. If you change a number in the README, say how you got it.
- CI must be green. Workflow changes are linted with `actionlint` and
  `zizmor`; third-party actions are pinned to full commit SHAs, and Dependabot
  keeps the pins current.

## License

By contributing you agree that your contribution is licensed under the
repository's [MIT license](LICENSE).
