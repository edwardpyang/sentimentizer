## Brief overview

Every pull request carries a semantic version bump, and the branch is pushed after
the bump. Merging a version change to `main` publishes to PyPI through
`.github/workflows/publish.yaml`, so the bump is what ships the change.

## Choosing the level

The package is pre-1.0, so a breaking change does not force a major release.

- **PATCH** (`0.331.0` → `0.331.1`): bug fixes with no contract change, docs, tests, CI,
  and internal refactors.
- **MINOR** (`0.331.0` → `0.332.0`): new features, and any change a caller can observe:
  an HTTP field or error code, a CLI exit status, a Python signature, a config default, or
  which model artifact gets loaded.
- **MAJOR**: only when deliberately declaring a stable or new-generation API. Ask first.

Pick the highest level that any change in the pull request needs. State the level and the
reason in the pull request description.

## How to bump

```bash
uv version --bump minor --no-sync   # updates pyproject.toml and uv.lock together
```

- Never edit only `pyproject.toml`. CI and the Dockerfile read `uv.lock`.
- Bump once per pull request, relative to the current `main`. If `main` moves to the same
  or a higher version before merge, rebase and bump again.
- Push after committing the bump. A pull request whose latest commit is not pushed has
  not shipped the bump.
