---
name: review-and-harden
description: Use when asked to review the codebase or a module for bugs, improvements, reliability, security, or test coverage gaps, or to fix the findings of such a review. Guides measuring a coverage baseline, reproducing each finding before reporting it, ranking by real impact, and fixing with regression tests. Do NOT use for reviewing a single small diff or for planning new features.
---

# Review and Harden

Find defects that matter, prove each one, then fix it with a test that keeps it fixed.

Do not run `mdformat` on this or any `.skills/**/SKILL.md`; it corrupts the YAML frontmatter.

## 1. Measure first

1. Inspect the branch and worktree. Preserve unrelated changes.
1. Run the suite with coverage and keep the data outside the repository:

   ```bash
   COVERAGE_FILE=<scratch>/.coverage uv run pytest --cov --cov-report=json:<scratch>/cov.json -q
   ```

1. Rank files by **missing lines**, not by percentage. Note that Ray worker code is not traced.
1. Use the gaps to direct reading. Untested success paths in core modules, such as
   `SentimentPredictor.predict_batch`, outrank untested logging branches.

## 2. Read for these defect classes

Each class has caused a real defect in this repository.

| Class | Question to ask | Rule |
|---|---|---|
| Partial state | What does a reader see mid-write, or after a crash? Does "exists" mean "complete"? | `.rules/durable-state-and-failures.md` |
| Hidden failure | Does a failure only log? Does the command still exit 0? | `.rules/durable-state-and-failures.md` |
| Degraded dependency | Which handlers still run when this failed to load? | `.rules/durable-state-and-failures.md` |
| Untrusted fields | Is every externally controlled field validated? What about `null` or a list? | `.rules/untrusted-input.md` |
| Unowned budget | Who resets this limit or counter? Is that code actually called? | `.rules/untrusted-input.md` |
| Middleware order | Which layers does an error response actually pass through? | `.rules/durable-state-and-failures.md` |
| False-confidence test | Does the test call production code or a copy of it? | `.rules/testing-workflow.md` |
| Stale documentation | Do `AGENTS.md`, manifests, env vars, and error messages match the code? | `AGENTS.md` |

Also read `.rules/` for the areas you touch, and use
`.skills/change-concurrent-code-safely/SKILL.md` for threads, Ray, or shared files.

## 3. Prove each finding

Write a throwaway reproduction in the scratch directory, and run it. Report only what you
observed or can show directly from the code.

- Drop or correct any suspicion that the reproduction refutes.
- Check `.dockerignore`, comments, and git history before calling something a mistake. It
  may be deliberate. If so, leave it and say so.
- Rank by consequence and likelihood, not by how easy the fix is.

## 4. Fix

1. Work on a branch, and keep fixes small and within the finding.
1. Add a regression test per fix. See `.rules/testing-workflow.md`.
1. Replace test helpers that reimplement production code when you find them.
1. Update `AGENTS.md`, docs, and manifests when a contract changes.
1. Split structural refactors of large modules into their own plan
   (`.skills/plan-feature/SKILL.md`) rather than mixing them with fixes.
1. Run `make ci`, then ship with `.skills/ship-pull-request/SKILL.md`.

## 5. Report

List what was fixed, what was deliberately left and why, and any decision that belongs to the
owner. Give coverage before and after from the same command. Never call a finding fixed
without a test or an observed result.
