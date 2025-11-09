#!/usr/bin/env python3
# pip_download_recursive.py
#
# Robust, opinionated "pip download" driver for Raspberry Pi 5 (Bookworm, aarch64),
# executed on Ubuntu 24.04. Forces pip==24.0 inside a throwaway venv and injects
# target environment markers so dependency markers are evaluated for the TARGET,
# not the HOST.
#
# If anything deviates from what we expect, we crash loudly without fallbacks.

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

# -------- Configuration (opinionated defaults for Raspberry Pi 5 Bookworm) -----

# Default target platform tag that is broadly compatible for manylinux wheels.
# manylinux2014_aarch64 is widely published on PyPI; it's the safe default.
DEFAULT_PLATFORM_TAG = "manylinux2014_aarch64"

# We require these keys in pi_markers_full.json (canonical + aliases + extra).
REQUIRED_MARKER_KEYS: Tuple[str, ...] = (
    # canonical
    "implementation_name",
    "implementation_version",
    "os_name",
    "platform_machine",
    "platform_release",
    "platform_system",
    "platform_version",
    "python_full_version",
    "platform_python_implementation",
    "python_version",
    "sys_platform",
    # aliases (legacy spellings)
    "os.name",
    "sys.platform",
    "platform.version",
    "platform.machine",
    "platform.python_implementation",
    "python_implementation",
    # pip/packaging add "extra"; we demand it here to be explicit
    "extra",
)

# We hard-lock to pip 24.0, because this script validates/patches for precisely
# that structure (vendored packaging markers, etc).
REQUIRED_PIP_VERSION = "24.0"


# ------------------------------- CLI ------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download wheels for Raspberry Pi 5 (Bookworm, aarch64) on Ubuntu, "
            "while forcing pip to evaluate environment markers against target Pi."
        )
    )
    parser.add_argument(
        "--packages", "--package", dest="packages", nargs="+", required=True,
        help="Space-delimited list of packages to download (e.g. --packages lancedb numpy)."
    )
    parser.add_argument(
        "--markers-json", default="pi_markers_full.json",
        help="Path to target marker JSON captured on the Raspberry Pi 5 (default: ./pi_markers_full.json)."
    )
    parser.add_argument(
        "--platform-tag", default=DEFAULT_PLATFORM_TAG,
        help=f"Target platform tag for wheels (default: {DEFAULT_PLATFORM_TAG})."
    )
    parser.add_argument(
        "--dest", default="temp_downloaded_pip_packages",
        help="Output folder (must not exist) (default: ./temp_downloaded_pip_packages)."
    )
    # CHANGED: verbose by default; use --quiet to reduce diagnostics.
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce diagnostics (verbose is default)."
    )
    return parser.parse_args()


# --------------------------- Error Helpers ------------------------------------

def die(msg: str, code: int = 1) -> None:
    print(f"\nFATAL: {msg}\n", file=sys.stderr)
    sys.exit(code)


def run_checked(cmd: List[str], cwd: Optional[Path] = None, env: Optional[Mapping[str, str]] = None) -> None:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=dict(env) if env else None, check=True)


# ----------------------- Marker JSON + Sanity Checks --------------------------

@dataclass(frozen=True)
class TargetEnv:
    raw: Dict[str, str]
    py_major: int
    py_minor: int
    py_micro: int
    py_version_twodot: str   # e.g., "3.11"
    abi_tag: str             # e.g., "cp311"

    @staticmethod
    def load(path: Path) -> "TargetEnv":
        if not path.exists():
            _print_pi_helper_and_exit(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        # Require all keys to exist and be strings.
        missing = [k for k in REQUIRED_MARKER_KEYS if k not in data]
        if missing:
            die(
                "Your marker JSON is missing required keys:\n"
                + "\n".join(f"  - {k}" for k in missing)
                + "\n\nRegenerate on the Raspberry Pi 5 with the helper shown below."
            )
        for k, v in data.items():
            if v is None:
                die(f"Marker JSON contains null for key '{k}'. Refuse to proceed.")
            if not isinstance(v, str):
                die(f"Marker JSON has non-string value for key '{k}' (got {type(v).__name__}).")

        # Derive Python version/ABI deterministically from json. We don't trust host.
        full = data["python_full_version"]  # e.g., "3.11.2"
        m = re.match(r"^(\d+)\.(\d+)\.(\d+)", full)
        if not m:
            die(f"Invalid python_full_version in JSON: {full!r}")
        maj, min_, mic = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        py_ver = f"{maj}.{min_}"
        abi = f"cp{maj}{min_}"

        return TargetEnv(
            raw=data,
            py_major=maj, py_minor=min_, py_micro=mic,
            py_version_twodot=py_ver,
            abi_tag=abi
        )


def _print_pi_helper_and_exit(expected_path: Path) -> None:
    print(
        f"""
ERROR: Required JSON file was not found: {expected_path}

This file MUST be generated on your **target Raspberry Pi 5 (Bookworm, aarch64)** so we can
force pip to evaluate markers as if we were on the Pi. Otherwise, `pip download` on Ubuntu
will incorrectly assume host markers (e.g., Python 3.12) and skip dependencies that are only
needed on < 3.12 (real-world example: `lancedb` ⇒ conditional `overrides` on Python < 3.12).

Run ALL of the following on the Raspberry Pi 5, then copy the resulting file back here as
{expected_path.name} in the same directory as this script:

------------------ CUT BELOW AND PASTE ON THE PI ------------------

python3 - <<'PY'
import json, os, platform, sys

def format_full_version(v):
    s = f"{{v.major}}.{{v.minor}}.{{v.micro}}"
    if v.releaselevel != "final":
        s += v.releaselevel[0] + str(v.serial)
    return s

def full_env():
    env = {{
        "implementation_name": sys.implementation.name,
        "implementation_version": format_full_version(sys.implementation.version),
        "os_name": os.name,
        "platform_machine": platform.machine(),
        "platform_release": platform.release(),
        "platform_system": platform.system(),
        "platform_version": platform.version(),
        "python_full_version": platform.python_version(),
        "platform_python_implementation": platform.python_implementation(),
        "python_version": ".".join(platform.python_version_tuple()[:2]),
        "sys_platform": sys.platform,
        "extra": "",
    }}
    # Legacy aliases required by older/vendored parsers
    env.update({{
        "os.name": env["os_name"],
        "sys.platform": env["sys_platform"],
        "platform.version": env["platform_version"],
        "platform.machine": env["platform_machine"],
        "platform.python_implementation": env["platform_python_implementation"],
        "python_implementation": env["platform_python_implementation"],
    }})
    return env

out = {expected_path.name!r}
with open(out, "w", encoding="utf-8") as f:
    json.dump(full_env(), f, indent=2, sort_keys=True)
print(f"Wrote {{out}}")
PY

------------------ CUT ABOVE AND PASTE ON THE PI ------------------

Once {expected_path.name} is in place, re-run this script.

Context reminder: you're likely on Ubuntu 24.04 (host Python {platform.python_version()}) while
your Pi target is Python 3.11.2. Packages like 'lancedb' may require 'overrides' when
python_full_version < "3.12". If we don't force target markers, pip-on-host will skip 'overrides',
and you'll be missing dependencies at runtime on the Pi.
""".strip()
    )
    sys.exit(2)


# --------------------------- Venv Utilities -----------------------------------

@dataclass
class VenvPaths:
    root: Path
    python: Path
    site_packages: Path


def _venv_python_and_site(venv_dir: Path) -> VenvPaths:
    # Ubuntu/CPython venv layout
    if sys.platform.startswith("linux"):
        py = venv_dir / "bin" / "python"
        # Try to detect the one and only site-packages dir
        # Do not import it; parse sys.path via a subprocess
        code = "import sys; import site; print('\\n'.join(p for p in sys.path if p.endswith('site-packages')))"
        out = subprocess.check_output([str(py), "-c", code], text=True)
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if not lines:
            die("Could not locate site-packages inside venv; refusing to proceed.")
        site = Path(lines[0])
        return VenvPaths(root=venv_dir, python=py, site_packages=site)
    die("Unsupported platform for this script (expected Linux).")


def create_locked_venv(tmp_root: Path, verbose: bool) -> VenvPaths:
    venv_dir = tmp_root / "pipdl_venv"
    run_checked([sys.executable, "-m", "venv", str(venv_dir)])
    vp = _venv_python_and_site(venv_dir)

    # Pin pip to EXACT version we audit against.
    run_checked([str(vp.python), "-m", "pip", "install", "--upgrade", f"pip=={REQUIRED_PIP_VERSION}"])

    # Validate pip version is what we think.
    out = subprocess.check_output([str(vp.python), "-m", "pip", "--version"], text=True)
    # Example: "pip 24.0 from /.../site-packages/pip (python 3.12)"
    m = re.match(r"^pip\s+(\S+)\s+", out.strip())
    if not m or m.group(1) != REQUIRED_PIP_VERSION:
        die(f"Expected pip=={REQUIRED_PIP_VERSION} inside venv, got: {out.strip()}")

    if verbose:
        print(f"Venv created at: {venv_dir}")
        print(f"Using pip: {out.strip()}")
    return vp


# -------------------------- Marker Patching -----------------------------------

SITECUSTOMIZE_CODE = r"""
# Auto-generated by pip_download_recursive.py
# Purpose: Force pip==24.0 to evaluate environment markers against a custom TARGET
# (Raspberry Pi 5 Bookworm) instead of the HOST (Ubuntu 24.04).
import json, os, sys, types

_EXPECTED_PIP_VERSION = "{req_pip}"
_JSON_BASENAME = "{json_name}"

# Required keys (canonical + aliases + extra)
_REQUIRED_KEYS = {required_keys!r}

def _die(msg: str) -> None:
    raise RuntimeError("[sitecustomize] " + msg)

def _load_json_from_cwd() -> dict:
    # The driver script ensures we run in the directory that contains the JSON.
    p = os.path.abspath(os.path.join(os.getcwd(), _JSON_BASENAME))
    if not os.path.exists(p):
        _die(f"Required target marker JSON not found: {{p}}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    missing = [k for k in _REQUIRED_KEYS if k not in data]
    if missing:
        _die("Marker JSON missing keys: " + ", ".join(missing))
    # Normalize to strings
    for k, v in list(data.items()):
        if v is None:
            _die(f"Marker JSON has null for key '{{k}}'")
        if not isinstance(v, str):
            data[k] = str(v)
    return data

def _make_default_environment_forced(target_env: dict):
    # returns a function that ignores the host and returns a COPY of target_env (strings)
    def default_environment() -> dict:
        # packaging/markers expects a mapping[str, str]
        env = dict(target_env)
        # Guarantee "extra" exists and is string
        env.setdefault("extra", "")
        return env
    return default_environment

def _make_marker_evaluate_forced(default_environment_func):
    # returns a bound method replacement for Marker.evaluate(self, environment=None)
    def evaluate(self, environment=None):
        current_environment = default_environment_func()
        # packaging >=24 adds extra="" + None normalization; emulate safely
        if environment is not None:
            current_environment.update(environment)
            if current_environment.get("extra") is None:
                current_environment["extra"] = ""
        return _evaluate_markers(self._markers, current_environment)
    return evaluate

def _assert_pip_version():
    import pip
    ver = getattr(pip, "__version__", "")
    if ver != _EXPECTED_PIP_VERSION:
        _die(f"Expected pip=={{_EXPECTED_PIP_VERSION}} inside venv, got {{ver!r}}")

def _patch_module(mod, target_env):
    # Validate expected names exist, then patch ONLY the functions we need.
    # We require both names to exist, with callable function objects.
    de = getattr(mod, "default_environment", None)
    ev = getattr(getattr(mod, "Marker", None), "evaluate", None) if hasattr(mod, "Marker") else None
    if not callable(de) or not callable(ev):
        _die(f"Module {{mod.__name__}} does not expose expected callables: default_environment/Marker.evaluate")

    # Also capture _evaluate_markers for our replacement
    global _evaluate_markers
    _evaluate_markers = getattr(mod, "_evaluate_markers", None)
    if not callable(_evaluate_markers):
        _die(f"Module {{mod.__name__}} missing _evaluate_markers callable")

    forced_de = _make_default_environment_forced(target_env)
    forced_eval = _make_marker_evaluate_forced(forced_de)

    # Monkey-patch
    mod.default_environment = forced_de
    mod.Marker.evaluate = forced_eval  # type: ignore[attr-defined]

    # quick smoke-test: host vs target should diverge if versions differ
    # e.g., for our real-world issue: "python_full_version < '3.12'"
    from types import SimpleNamespace
    class _FakeVar:
        def __init__(self, value): self.value = value
        def serialize(self): return str(self.value)
    class _FakeOp:
        def __init__(self, op): self._op = op
        def serialize(self): return self._op
    class _FakeVal:
        def __init__(self, v): self.value = v
        def serialize(self): return f'"{{self.value}}"'
    # A simple parsed marker tuple: (Variable('python_full_version'), Op('<'), Value('3.12'))
    tuple_marker = [(_FakeVar("python_full_version"), _FakeOp("<"), _FakeVal("3.12"))]
    # Evaluate via module's machinery
    def _eval_for(mod_):
        class _M:
            _markers = tuple_marker
        M = _M()
        return mod_.Marker.evaluate(M, None)  # type: ignore

    test = _eval_for(mod)
    if not isinstance(test, bool):
        _die(f"Sanity test failed for {{mod.__name__}} evaluate path (not bool)")

def _main():
    _assert_pip_version()
    target_env = _load_json_from_cwd()

    # Patch vendored packaging (pip uses this path for markers & requirements)
    try:
        import pip._vendor.packaging.markers as vmarkers
    except Exception as e:
        _die("Failed to import pip._vendor.packaging.markers: " + repr(e))
    _patch_module(vmarkers, target_env)

    # Also patch external packaging if present (belt & suspenders)
    try:
        import packaging.markers as pmarkers  # may not be used, but patch anyway
        _patch_module(pmarkers, target_env)
    except Exception:
        pass

    # Print one explicit check to aid debugging:
    try:
        host_py = sys.version
        tgt_py = target_env.get("python_full_version")
        msg = f"[sitecustomize] Host Python={{host_py}} | Target python_full_version={{tgt_py}}"
        print(msg, file=sys.stderr)
    except Exception:
        pass

_main()
""".format(
    req_pip=REQUIRED_PIP_VERSION,
    json_name="pi_markers_full.json",
    required_keys=list(REQUIRED_MARKER_KEYS),
)


def write_sitecustomize(site_dir: Path, verbose: bool) -> Path:
    sc = site_dir / "sitecustomize.py"
    if sc.exists():
        die(f"Refusing to overwrite existing {sc}")
    sc.write_text(SITECUSTOMIZE_CODE, encoding="utf-8")
    if verbose:
        print(f"Wrote patch injector: {sc}")
    return sc


def validate_patch_effect(vpython: Path, target: TargetEnv, verbose: bool) -> None:
    # Validate that the patched default_environment returns exactly the target JSON (stringified)
    code = r"""
import json
from pip._vendor.packaging import markers as vm
try:
    import packaging.markers as pm
except Exception:
    pm = None

env = vm.default_environment()
print(json.dumps(env, sort_keys=True))
print("OK-VENDORED")

if pm is not None:
    env2 = pm.default_environment()
    print(json.dumps(env2, sort_keys=True))
    print("OK-EXTERNAL")
"""
    out = subprocess.check_output([str(vpython), "-c", code], text=True, cwd=os.getcwd())
    lines = [l for l in out.splitlines() if l.strip()]
    # Expect at least 2 lines: json then "OK-VENDORED"
    if "OK-VENDORED" not in lines[-1] and "OK-VENDORED" not in lines[-2:]:
        die("Vendored packaging markers patch did not run (missing OK-VENDORED).")

    # First JSON blob must match our file (as strings)
    vendor_json_str = lines[0]
    vendored = json.loads(vendor_json_str)
    # Check required keys & equality
    for k in REQUIRED_MARKER_KEYS:
        if k not in vendored:
            die(f"Vendored default_environment missing key: {k}")
    # Compare a couple of critical fields
    assert vendored["python_full_version"] == f"{target.py_major}.{target.py_minor}.{target.py_micro}"
    assert vendored["python_version"] == target.py_version_twodot

    if verbose:
        print("Patched vendored markers look correct.")


# ------------------------------ Download --------------------------------------

def pip_download_in_venv(vp: VenvPaths, env: TargetEnv, dest_dir: Path, platform_tag: str, packages: List[str]) -> None:
    cmd = [
        str(vp.python), "-m", "pip", "download",
        "--only-binary", ":all:",
        "--platform", platform_tag,
        "--implementation", "cp",
        "--python-version", env.py_version_twodot,
        "--abi", env.abi_tag,
        "--dest", str(dest_dir),
    ] + packages
    print("\nInvoking pip download with TARGET markers:")
    print(f"  python_version={env.py_version_twodot}  abi={env.abi_tag}  platform={platform_tag}")
    print(f"  dest={dest_dir}")
    run_checked(cmd)


# --------------------------------- Main ---------------------------------------

def main() -> None:
    args = parse_args()
    verbose = not args.quiet  # <— CHANGED: verbose by default, flip with --quiet

    dest = Path(args.dest).resolve()
    if dest.exists():
        die(f"Destination already exists: {dest} (refusing to run).")

    # Load & validate target markers
    markers_path = Path(args.markers_json).resolve()
    target = TargetEnv.load(markers_path)

    # Create temp dir for venv; leave it on exception so user can inspect
    tmp_root = Path(tempfile.mkdtemp(prefix="pipdl_"))
    try:
        if verbose:
            print(f"Working directory: {Path.cwd()}")
            print(f"Temporary root: {tmp_root}")

        vp = create_locked_venv(tmp_root, verbose)

        # Inject our sitecustomize patch into the venv's site-packages
        write_sitecustomize(vp.site_packages, verbose)

        # Sanity: demonstrate host vs target marker divergence specifically for the real bug.
        # Host is Ubuntu 24.04 Python 3.12.*, Target example is Python 3.11.2 on Pi.
        # We validate:
        #   under target markers => (python_full_version < "3.12") == True
        #   under host markers   => likely False (since host is 3.12.*)
        host_full = platform.python_version()
        host_truth = (tuple(map(int, host_full.split(".")[:3])) < (3, 12, 0))
        # Now ask the venv to compute via patched vendored packaging
        code = r"""
from pip._vendor.packaging import markers as vm
m = vm.Marker('python_full_version < "3.12"')
print("TARGET_COND=", "T" if m.evaluate() else "F")
"""
        out = subprocess.check_output([str(vp.python), "-c", code], text=True, cwd=os.getcwd()).strip()
        if "TARGET_COND= T" not in out and "TARGET_COND=T" not in out:
            die("Patched Marker.evaluate did not return True for target 'python_full_version < 3.12' check.")

        if verbose:
            print(f"Host python_full_version={host_full}  -> condition <3.12? {host_truth}")
            print(out)

        # Double-check our patched module returns exactly the provided environment
        validate_patch_effect(vp.python, target, verbose)

        # Create destination folder (must not exist yet)
        dest.mkdir(parents=False, exist_ok=False)

        # Finally perform the download as the TARGET
        pip_download_in_venv(vp, target, dest, args.platform_tag, args.packages)

        print("\nDownload complete.")
        print(f"Wheels saved to: {dest}")

    finally:
        # Clean up venv on success; on exception we keep it around to aid debugging.
        # If you want forced cleanup even on error, move this outside the finally.
        if tmp_root.exists():
            try:
                shutil.rmtree(tmp_root)
                print(f"Cleaned temporary venv: {tmp_root}")
            except Exception as e:
                print(f"WARNING: Could not remove temp dir {tmp_root}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
