# Releasing bigleague

`bigleague` is published to PyPI by `.github/workflows/release.yml`. A `vX.Y.Z`
tag builds the wheel and source distribution, runs the full verification suite,
and publishes through PyPI Trusted Publishing (OIDC). No PyPI token is stored in
GitHub.

## One-time setup

### 1. Check the name and the repository settings

Bigleague is MIT licensed. The name `bigleague` was unclaimed on PyPI when this
was written (2026-11-21), but a pending publisher does not reserve it, so
re-check `https://pypi.org/project/bigleague/` returns 404 right before the
first release.

In `klobucar/bigleague` → **Settings**:

- Make the repository public. The README gallery images use
  `raw.githubusercontent.com` URLs, which only load for a public repository.
- Under **Code security**, enable **Private vulnerability reporting**.
  `SECURITY.md` directs reporters to that form.
- Require the `CI` workflow to pass on `main` (branch protection or a ruleset).

### 2. Create the GitHub environment

In `klobucar/bigleague`, open **Settings → Environments → New environment** and
create `release`. Recommended protections:

- Add yourself as the required reviewer.
- Allow deployment only from tags matching `v*`.
- Protect matching `v*` tags with a repository tag rule.

The environment name is part of the OIDC identity and must match PyPI exactly.

### 3. Register the PyPI publisher

If `bigleague` does not exist on PyPI, sign in and open **Your account →
Publishing → Add a new pending publisher**. Enter:

| Field | Value |
| --- | --- |
| PyPI project name | `bigleague` |
| Owner | `klobucar` |
| Repository | `bigleague` |
| Workflow name | `release.yml` |
| Environment | `release` |

If the project already exists, add the same identity under **Manage project →
Publishing** instead. Do not add a `PYPI_TOKEN` secret. The first successful use
of a pending publisher creates the project and converts it to a normal trusted
publisher.

## Release checklist

1. Start from a clean `main` branch with CI passing.
2. Update `project.version` in `pyproject.toml` using semantic versioning.
3. In `CHANGELOG.md`, rename `[Unreleased]` to the new version with today's
   date and add a fresh empty `[Unreleased]` section above it.
4. If `exclude-newer-package` in `pyproject.toml` still has a `bouquet-bloom`
   entry and that release is more than a week old, delete the entry. The
   global one-week cooldown covers it from then on.
5. Update `uv.lock` and review the release diff. The test suite runs twice
   because the bloom tests skip themselves without `bouquet-bloom`:

   ```bash
   uv lock
   uv sync --extra dev --locked
   uv run --no-sync ruff format --check .
   uv run --no-sync ruff check .
   uv run --no-sync ty check
   uv run --no-sync pytest
   uv sync --extra dev --extra bloom --locked
   uv run --no-sync pytest
   uv build --out-dir dist
   uvx --from twine twine check dist/*
   ```

6. Commit the version change and merge it to `main`.
7. In **Actions → Release to PyPI → Run workflow**, leave `dry_run` enabled.
   Inspect the resulting wheel and source-distribution artifact if desired.
8. Tag the exact verified commit and push the tag:

   ```bash
   git switch main
   git pull --ff-only
   git tag -a v0.1.0 -m "bigleague 0.1.0"
   git push origin v0.1.0
   ```

9. Approve the `release` environment deployment in GitHub Actions.
10. Confirm the workflow published both a `.whl` and `.tar.gz`, then verify from
   a clean environment:

   ```bash
   uv run --isolated --with bigleague==0.1.0 \
     python -c 'from importlib.metadata import version; print(version("bigleague"))'
   ```

11. Create a GitHub Release from the same tag and include the user-facing changes.

PyPI releases are immutable. If a release is bad, do not reuse its version or
move its tag: yank it on PyPI, fix the problem, and publish a new patch version.

## OIDC troubleshooting

- `invalid-publisher` almost always means the owner, repository, workflow file,
  or environment differs between PyPI and the workflow.
- The publish job—not the build job—must have `id-token: write`.
- A pending publisher's project name must exactly match `project.name`.
- Renaming the repository or workflow requires updating the PyPI publisher.
