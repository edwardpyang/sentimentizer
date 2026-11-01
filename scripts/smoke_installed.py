"""Check an installed wheel or container without importing from the checkout."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

IMPORT_CHECK = """
import importlib
import sysconfig
from importlib.resources import files
from pathlib import Path

install_roots = {Path(sysconfig.get_path(key)).resolve() for key in ('purelib', 'platlib')}
for name in ('sentimentizer', 'workflows', 'workflows.cli', 'sentimentizer.serve.__main__'):
    module = importlib.import_module(name)
    origin = Path(module.__file__).resolve()
    assert any(origin.is_relative_to(root) for root in install_roots), (name, origin)

config = files('sentimentizer.serve').joinpath('service.yaml')
assert config.is_file(), 'service.yaml is missing from the installed package'
from sentimentizer.serve.config import load_serve_config
load_serve_config(str(config))
"""


def main() -> None:
    """Smoke installed imports, package resources, and both CLI entry points."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("SENTIMENTIZER_SERVE_CONFIG", None)
    console = Path(sys.executable).with_name("sentimentizer")
    commands = [
        [sys.executable, "-I", "-c", IMPORT_CHECK],
        [str(console), "--help"],
        [sys.executable, "-I", "-m", "sentimentizer.serve", "--help"],
    ]
    with tempfile.TemporaryDirectory(prefix="sentimentizer-smoke-") as directory:
        for command in commands:
            subprocess.run(command, cwd=directory, env=environment, check=True, timeout=60)
    print("Installed package smoke checks passed.")


if __name__ == "__main__":
    main()
