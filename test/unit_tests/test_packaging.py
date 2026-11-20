"""Tests for the distribution packages declared in setup.py."""

import runpy
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).parents[2]


@pytest.mark.skipif(not (ROOT / "setup.py").exists(), reason="requires a source checkout")
def test_setup_only_packages_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    with patch("setuptools.setup") as setup:
        runpy.run_path(str(ROOT / "setup.py"))
    packages = setup.call_args.kwargs["packages"]
    assert "qiskit_braket_provider" in packages
    # anything else, such as the top-level "test" package, would be installed into site-packages
    unexpected = [
        package
        for package in packages
        if package != "qiskit_braket_provider" and not package.startswith("qiskit_braket_provider.")
    ]
    assert not unexpected, f"unexpected packages in the distribution: {unexpected}"
