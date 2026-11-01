"""Render coverage directly in a GitHub Actions job summary."""

import argparse
import io
from pathlib import Path

from coverage import Coverage


def render_summary(test_outcome: str) -> str:
    """Return a complete or explicitly partial report when coverage data exists."""
    coverage = Coverage()
    if not Path(coverage.config.data_file).is_file():
        return ""
    coverage.load()
    if not coverage.get_data().measured_files():
        return ""

    table = io.StringIO()
    total = coverage.report(
        file=table, output_format="markdown", sort="cover", skip_covered=True, show_missing=True
    )
    description = "Line and branch coverage, least covered files first."
    if test_outcome != "success":
        description += (
            "\n\n**Partial report:** tests did not pass; coverage reflects only the tests that ran."
        )
    return f"## Test coverage: {total:.0f}%\n\n{description}\n\n{table.getvalue()}"


def main() -> None:
    """Write Markdown to stdout for the workflow's summary redirection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-outcome", required=True)
    arguments = parser.parse_args()
    print(render_summary(arguments.test_outcome), end="")


if __name__ == "__main__":
    main()
