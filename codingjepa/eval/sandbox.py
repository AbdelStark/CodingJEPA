"""Isolated subprocess runner for CJ-EXEC (RFC-0010 §D3 E3, #111).

Falls back to a plain ``subprocess.run`` when ``nsjail`` / ``firejail`` are
unavailable so the dev and CI environments don't require system-level
sandboxing tools. The contract is intentionally tiny: write a ``subject.py``
+ ``test_subject.py`` pair into a temp dir, run ``pytest`` against it, and
return ``(passed, stdout, returncode)``.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile

SANDBOX_BACKENDS = ("nsjail", "firejail", "none")


def run_in_sandbox(
    code: str,
    test_source: str,
    *,
    timeout: int = 30,
    backend: str = "none",
) -> tuple[bool, str, int]:
    """Run ``test_source`` against ``code`` in an isolated subprocess.

    Parameters
    ----------
    code
        The ``subject.py`` content (the function under test).
    test_source
        The ``test_subject.py`` content. Must ``from subject import ...``.
    timeout
        Wall-clock timeout for the pytest subprocess.
    backend
        ``nsjail``, ``firejail``, or ``none``. Unknown / unavailable
        backends silently fall back to ``none``.

    Returns
    -------
    tuple
        ``(passed, stdout, returncode)`` where ``passed`` is True iff
        ``returncode == 0``.
    """

    with tempfile.TemporaryDirectory() as tmp:
        t = pathlib.Path(tmp)
        (t / "subject.py").write_text(code, encoding="utf-8")
        (t / "test_subject.py").write_text(test_source, encoding="utf-8")
        cmd = _build_cmd(backend, t)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(t),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout.decode("utf-8", errors="replace") if exc.stdout else "") + (
                exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            )
            return False, stdout + f"\nTIMEOUT after {timeout}s", 124
        stdout = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode == 0, stdout, proc.returncode


def _build_cmd(backend: str, workdir: pathlib.Path) -> list[str]:
    """Build the subprocess command for the requested backend.

    Falls back to running ``pytest`` directly (or ``python -m pytest`` if
    the ``pytest`` script is not on the PATH) when the requested sandbox
    binary is not installed.
    """

    test_file = "test_subject.py"
    if backend == "nsjail" and shutil.which("nsjail"):
        return [
            "nsjail",
            "--mode",
            "o",
            "--",
            "pytest",
            test_file,
            "-x",
            "-q",
            "--tb=no",
        ]
    if backend == "firejail" and shutil.which("firejail"):
        return [
            "firejail",
            "--quiet",
            "--",
            "pytest",
            test_file,
            "-x",
            "-q",
            "--tb=no",
        ]
    # Plain subprocess fallback. Prefer the installed ``pytest`` script so
    # we don't depend on PYTHONPATH; otherwise invoke via ``python -m``.
    workdir_unused = workdir  # workdir is the subprocess cwd; arg kept for symmetry.
    del workdir_unused
    pytest_bin = shutil.which("pytest")
    if pytest_bin:
        return [pytest_bin, test_file, "-x", "-q", "--tb=no"]
    return [sys.executable, "-m", "pytest", test_file, "-x", "-q", "--tb=no"]


__all__ = ["SANDBOX_BACKENDS", "run_in_sandbox"]
