## Brief overview
Rule for mandatory linting and testing during the implementation of plans or
features. This ensures code quality and prevents compounding errors as the
implementation progresses.

## Lint and Test Workflow
- **Continuous Validation**: Whenever you complete a logical section, milestone,
  or component of a larger implementation plan, you MUST run the project's
  verification suite before proceeding to the next section.
- **Do Not Defer**: Do not wait until the entire plan is implemented to run the
  checks. Catching errors early makes debugging significantly easier and
  prevents cascading failures.
- **Tools**: Use the project's checking commands (`make check` for the fast
  inner loop — format + lint only; `make ci` for the full gate — format + lint
  + pyright + pytest).
- **Fix Before Moving On**: If the linter or tests fail, you MUST fix the
  issues before beginning work on the next logical component or step in the
  plan.
- **Reporting**: When you communicate with the user that a step is completed,
  mention that you have successfully ran the checks and that the codebase
  remains in a healthy state.

## Regression tests

- **Every bug fix includes a regression test.** Watch it fail against the old behavior when
  practical, and name the behavior rather than the ticket or the fix.
- **Test the real entry point.** Call the actual handler, stage, or graph function. A test
  helper that reimplements a handler verifies the copy, and it keeps passing after the real
  code breaks. Mock the boundaries (network, models, Ray), not the code under test.
- **Cover the failure path.** Include malformed input, interrupted work, and degraded
  dependencies, and assert on what is left behind: no partial file, previous file intact,
  error type, status code, and exit behavior.
- Do not add tests that only execute lines to raise a percentage. Assert on behavior.

## Coverage

- `make test-cov` reports line and branch coverage using the scope in `pyproject.toml`.
- CI and publishing reuse `.github/workflows/verify.yaml`, including wheel and
  container smoke checks outside the source tree.
- CI publishes the same numbers and missing lines to the job summary and a
  `coverage-report` artifact. Reporting also runs after test failures when data
  exists; unsuccessful test runs are explicitly marked partial.
- Code running in Ray worker processes is not traced, so low numbers in
  `trainer.py::_train_func` and similar paths can understate real coverage. Check whether a
  test exercises the path before treating it as a gap.
