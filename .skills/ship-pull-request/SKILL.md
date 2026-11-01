---
name: ship-pull-request
description: Use when opening or updating a pull request in this repository, or when asked to commit and push finished work for review. Covers the required semantic version bump, the verification gate, pushing, the PR description, and confirming CI. Do NOT use for local-only commits or work the user has not asked to publish.
---

# Ship a Pull Request

Do not run `mdformat` on this or any `.skills/**/SKILL.md`; it corrupts the YAML frontmatter.

## Steps

1. **Check the branch.** Never commit to `main`. Confirm `git status` holds only intended
   changes, and preserve unrelated user work.
1. **Bump the version.** Choose the level with `.rules/versioning.md`, then:

   ```bash
   uv version --bump <patch|minor|major> --no-sync
   git diff --stat pyproject.toml uv.lock   # both must change
   ```

1. **Verify.** Run `make ci` and fix failures before continuing. Documentation-only pull
   requests may skip the test suite, but still bump the version.
1. **Commit and push** the code, tests, and bump together, then `git push -u origin <branch>`.
1. **Open the pull request against `main`** with `gh pr create`. Describe what changed and
   why, the version level and reason, how it was tested, and anything left undone.
1. **Confirm CI.** Run `gh pr checks <number> --watch`. Fix failures and push again. Report
   the result, including the coverage figure from the job summary.

## Reminders

- Merging a version change publishes to PyPI. Mention any release-sensitive change.
- If `main` moves past your version before merge, rebase and bump again.
- A version bump in every pull request means parallel branches conflict on `uv.lock`.
  Resolve that by re-running the bump on top of `main`, not by hand-editing the lock.
