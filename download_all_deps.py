#!/usr/bin/env python3
"""
download_all_deps.py

Purpose:
    Fetch EVERYTHING (except apt packages) required to run the offline
    Hailo RPi5 examples at a precise repo commit on a Raspberry Pi 5 (aarch64).
    Strongly opinionated: this script expects specific tools, tags, layouts,
    and will fail loudly with actionable guidance if anything deviates.

Where it runs:
    On an ONLINE Ubuntu 24.04 x86_64 machine.

What it does (high-level):
  1) Safety check: if ./hailo-rpi_examples exists in CWD -> REFUSE to run (as requested).
  2) Clone https://github.com/hailo-ai/hailo-rpi5-examples.git at EXACT COMMIT:
       9fd0e087577009fd30ee1981d1e21e69387edba7   (branch main)
     into ./hailo-rpi5-examples (new directory). If the directory already exists, fail.
  3) Populate ./hailo-rpi5-examples/resources/ with:
       - ./resources/models/hailo8/*.hef
       - ./resources/models/hailo8l/*.hef
       - ./resources/videos/example.mp4 and example_640.mp4
     These are normally fetched at runtime by:
       - ./hailo-rpi5-examples/download_resources.sh
       - or via "hailo-download-resources" from hailo-apps-infra post-install.
     We fetch them now so offline Pi has them immediately.
  4) Download Python wheels for the project (target: aarch64, chosen CPython version).
     These are normally installed online by:
       - ./hailo-rpi5-examples/install.sh  (pip install -r requirements.txt)
     We produce:
       ./hailo-rpi5-examples/offline_wheels/base/  (project & tests)
  5) Download Hailo wheels (Python bindings), normally fetched online by:
       - ./hailo-rpi5-examples/hailo_python_installation.sh
     We produce:
       ./hailo-rpi5-examples/offline_wheels/hailo/hailort-<V>-cpXY-cpXY-linux_aarch64.whl
       ./hailo-rpi5-examples/offline_wheels/hailo/tappas_core_python_binding-<V>-py3-none-any.whl
  6) Clone hailo-apps-infra locally (so install.sh can install from path OFFLINE):
       ./hailo-rpi5-examples/third_party/hailo-apps-infra  (tag/branch configurable)
     Then PATCH ./hailo-rpi5-examples/config.yaml:
       hailo_apps_infra_path: "third_party/hailo-apps-infra"
     so ./hailo-rpi5-examples/install.sh installs from the local path.
  7) Community projects (OPTIONAL but included by default):
       For each ./community_projects/*/requirements.txt, download aarch64 wheels into:
         ./hailo-rpi5-examples/offline_wheels/community/<project>/
       If a requirement uses a Git URL (e.g., CLIP), clone it under:
         ./hailo-rpi5-examples/third_party/<name>/
       These are normally installed ad-hoc by users; we fetch all now so you stay offline.
  8) Patch ./hailo-rpi5-examples/.gitignore (append with a DOUBLE NEWLINE first, then absolute-root ignores):
       /resources/models/
       /resources/videos/
       /offline_wheels/
       /third_party/
       (etc.)

How to use offline later on the Pi (overview):
  - Copy the entire ./hailo-rpi5-examples folder to the Pi.
  - Ensure apt packages are installed (mirror separately).
  - In the repo on the Pi:
      source ./setup_env.sh
      # To force local Hailo wheels without network, run install.sh with:
      #   ./install.sh -h offline_wheels/hailo/hailort-<...>.whl -p offline_wheels/hailo/tappas_core_python_binding-<...>.whl
      # Or simply run ./install.sh and it will detect packages (if already visible).
      python basic_pipelines/detection_simple.py
"""

import argparse
import os
import sys
import shutil
import subprocess
import textwrap
import re
from pathlib import Path
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    print("FATAL: Python 'requests' is required on this Ubuntu machine.\n"
          "Install it with:  python3 -m pip install requests\n"
          "Then re-run this script.")
    sys.exit(1)

# -----------------------
# CONSTANTS / CATALOGS
# -----------------------

REPO_URL = "https://github.com/hailo-ai/hailo-rpi5-examples.git"
REPO_COMMIT = "9fd0e087577009fd30ee1981d1e21e69387edba7"
REPO_DIR = Path("hailo-rpi5-examples")

# Hailo public bucket (as used by the repo scripts)
HAILO_BASE_URL = "http://dev-public.hailo.ai/2025_01"
HAILORT_VERSION = "4.20.0"
TAPPAS_CORE_VERSION = "3.31.0"

# Expected tools in this Ubuntu environment
REQUIRED_TOOLS = ["git", "python3", "pip", "wget"]  # strongly opinionated

# Model URLs (must map EXACTLY to filenames used by tests & examples)
H8_HEFS = [
    "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/yolov8m.hef",
    "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/yolov5m_wo_spp.hef",
    "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/yolov8s.hef",
    "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/yolov8m_pose.hef",
    "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/yolov8s_pose.hef",
    "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/yolov5m_seg.hef",
    "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/yolov5n_seg.hef",
    "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/yolov6n.hef",
    "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/yolov11n.hef",
    "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/yolov11s.hef",
    "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/scdepthv3.hef",
]

H8L_HEFS = [
    "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8l/yolov5m_wo_spp.hef",
    "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8l/yolov8m.hef",
    "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8l/yolov11n.hef",
    "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8l/yolov11s.hef",
    "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8l/yolov8s.hef",
    "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8l/yolov6n.hef",
    "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8l/scdepthv3.hef",
    "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8l/yolov8s_pose.hef",
    "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8l/yolov5n_seg.hef",
]

VIDEO_URLS = [
    "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/video/example.mp4",
    "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/video/example_640.mp4",
]

# Pip requirements for the base project (root + tests)
# NOTE: NumPy is intentionally pinned to a specific pre-2.0 release to avoid resolver drift
# and to match the HailoRT wheels' "<2" requirement on aarch64.
BASE_REQUIREMENTS = [
    "numpy==1.26.4",
    "setproctitle",
    "opencv-python",
    "pytest",
    "pytest-timeout",
    "pytest-mock",
    "python-dotenv",
]

# ---------------------------------------------------------------------------
# Community projects (DISABLED)
#
# EXACT REASON FOR DISABLING:
# - These optional demos (e.g., dynamic_captioning) introduce heavy deps like
#   PyTorch/Transformers that are unrelated to core HailoRT pipelines.
# - When cross-resolving on an x86_64 host with `pip download --platform aarch64`,
#   pip does not consistently evaluate environment markers for the *target*
#   platform. Recent PyTorch wheels split CUDA into `nvidia-*` packages guarded
#   by `platform_machine == "x86_64"`. On a real aarch64 machine those are
#   skipped, but on the x86_64 host pip attempts to include them; this conflicts
#   with `--platform ...aarch64` and resolution fails (your exact error).
# - Skipping these projects eliminates that entire class of failures and still
#   leaves you with a complete, working offline bundle for the official basic
#   examples (which only require HailoRT + Apps Infra).
#
# If you later want these demos:
#   * Resolve on the target (Pi) and bring back a locked list, or
#   * use an aarch64 userspace (QEMU/container) for resolution, or
#   * use PyTorch CPU index and build wheels for any git+ deps.
# ---------------------------------------------------------------------------

# If a req string starts with "git+", we will clone the repo into third_party/.
# COMMUNITY_REQUIREMENT_FILES = {
#     "dynamic_captioning": ["transformers", "onnxruntime", "torch==2.5.1", "git+https://github.com/openai/CLIP.git"],
#     "Navigator": ["torch", "opencv-python", "onnxruntime", "tqdm"],
#     "NeoPixel": ["pi5neo"],
#     "RoboChess": ["onnxruntime", "tensorflow", "matplotlib", "pyclipper", "scipy", "scikit-learn",
#                   "chess", "stockfish", "cairosvg", "IPython", "pyttsx3"],
#     "sailted_fish": ["pyttsx3"],
#     "TAILO": ["gdown", "playsound"],
#     "TEMPO": ["adafruit-circuitpython-ads1x15==2.4.1", "matplotlib", "scipy", "gdown", "gradio==5.3.0", "pyfluidsynth"],
#     "traffic_sign_detection": ["pynmea2", "serial"],  # 'asyncio' is builtin
#     "wled_display": ["numpy", "opencv-python"],
# }

# .gitignore lines to append (with leading slash and a double newline BEFORE these)
GITIGNORE_APPEND = [
    "/resources/models/",
    "/resources/videos/",
    "/offline_wheels/",
    "/third_party/",
]

# -----------------------
# UTILITIES
# -----------------------

def fail(msg: str):
    print("\nFATAL ERROR:\n" + textwrap.indent(msg, prefix="  ") + "\n")
    sys.exit(2)

def run(cmd, cwd=None, check=True, capture=False):
    if capture:
        result = subprocess.run(cmd, cwd=cwd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout, result.stderr
    else:
        subprocess.run(cmd, cwd=cwd, check=check)

def check_tools():
    missing = []
    for t in REQUIRED_TOOLS:
        if shutil.which(t) is None:
            missing.append(t)
    if missing:
        fail(
            "Required tools are missing on this Ubuntu machine: {}\n"
            "Install them and try again. For example:\n"
            "  sudo apt update && sudo apt install git python3-pip wget".format(", ".join(missing))
        )

def assert_clean_cwd():
    # As requested: if a folder named 'hailo-rpi_examples' exists in CWD, refuse to run.
    if Path("hailo-rpi_examples").exists():
        fail(
            "Folder './hailo-rpi_examples' exists in the current directory.\n"
            "Per the specification, this script refuses to run in this case.\n"
            "Rename or remove that folder, or run this script from a different directory."
        )

def clone_repo_exact():
    if REPO_DIR.exists():
        fail(
            f"Directory './{REPO_DIR}' already exists.\n"
            "This script expects to create it fresh by cloning the repo.\n"
            "Delete or move it, then re-run."
        )
    print(f"Cloning repo into ./{REPO_DIR} ...")
    run(["git", "clone", "--no-tags", "--filter=blob:none", REPO_URL, str(REPO_DIR)])
    print(f"Checking out exact commit {REPO_COMMIT} ...")
    run(["git", "checkout", REPO_COMMIT], cwd=REPO_DIR)
    # Verify commit
    stdout, _ = run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, capture=True)
    head = stdout.strip()
    if head != REPO_COMMIT:
        fail(
            f"After checkout, HEAD is {head}, expected {REPO_COMMIT}.\n"
            "Something changed on the remote repository. Re-try later or update this script with a new commit hash.\n"
            "If you must proceed now, ask: 'Can you update the script to the current commit on main?'"
        )
    print("Repo at exact commit OK.")

def ensure_dirs():
    # Create exact target directories that examples/tests expect
    (REPO_DIR / "resources" / "models" / "hailo8").mkdir(parents=True, exist_ok=True)
    (REPO_DIR / "resources" / "models" / "hailo8l").mkdir(parents=True, exist_ok=True)
    (REPO_DIR / "resources" / "videos").mkdir(parents=True, exist_ok=True)
    (REPO_DIR / "offline_wheels" / "base").mkdir(parents=True, exist_ok=True)
    (REPO_DIR / "offline_wheels" / "hailo").mkdir(parents=True, exist_ok=True)
    # (REPO_DIR / "offline_wheels" / "community").mkdir(parents=True, exist_ok=True)  # DISABLED: see rationale above
    (REPO_DIR / "offline_wheels" / "hailo_apps_infra").mkdir(parents=True, exist_ok=True)  # apps-infra deps
    (REPO_DIR / "offline_wheels" / "hailo_apps_infra_build").mkdir(parents=True, exist_ok=True)  # Build backend
    (REPO_DIR / "third_party").mkdir(parents=True, exist_ok=True)

def filename_from_url(u: str) -> str:
    return os.path.basename(urlparse(u).path)

def download_file(url: str, dest: Path, desc: str):
    print(f"  - Downloading {desc}: {url}")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1<<20):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        fail(
            f"Failed to download {desc} from:\n  {url}\n"
            f"Reason: {e}\n"
            "If the URL has changed or is unavailable, ask: 'What are the current model/video URLs for this repo?'"
        )
    if dest.stat().st_size == 0:
        fail(
            f"Downloaded file is empty:\n  {dest}\n"
            "The server may have returned an error page. Verify the URL is valid."
        )

def download_models_and_videos():
    print("Downloading Hailo HEF models and demo videos into ./hailo-rpi5-examples/resources/ ...")
    # HAILO8
    for u in H8_HEFS:
        fn = filename_from_url(u)
        download_file(u, REPO_DIR / "resources" / "models" / "hailo8" / fn, f"HEF (HAILO8) {fn}")
    # HAILO8L
    for u in H8L_HEFS:
        fn = filename_from_url(u)
        download_file(u, REPO_DIR / "resources" / "models" / "hailo8l" / fn, f"HEF (HAILO8L) {fn}")
    # Videos
    for u in VIDEO_URLS:
        fn = filename_from_url(u)
        download_file(u, REPO_DIR / "resources" / "videos" / fn, f"Video {fn}")
    print("Models and videos downloaded.")

def pip_download(pkgs, dest_dir: Path, py_ver: str, abi: str, platform_tag: str):
    """
    Use pip download to fetch wheels for a target platform/py version/abi.
    Opinionated flags: only wheels (no source), specific platform and ABI.
    Fail if pip cannot resolve a wheel (we do not accept source archives).
    """
    cmd = [
        sys.executable, "-m", "pip", "download",
        "--only-binary", ":all:",
        "--platform", platform_tag,
        "--implementation", "cp",
        "--python-version", py_ver,     # e.g., "3.11"
        "--abi", abi,                   # e.g., "cp311"
        "--dest", str(dest_dir),
    ] + pkgs
    print("Running:", " ".join(cmd))
    try:
        run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        fail(
            "pip download failed for the specified package set.\n"
            f"Command was:\n  {' '.join(cmd)}\n\n"
            "Reasons this happens:\n"
            "  • The package has no prebuilt wheel for manylinux aarch64 and your Python version.\n"
            "  • The exact version pinned does not exist for that platform.\n"
            "  • A package uses 'git+' sources (requires cloning instead).\n\n"
            "What to ask to fix:\n"
            "  'Which package lacks an aarch64 wheel for my Python version, and what version should I pin instead?'\n"
            "  'For git+ dependencies, can you clone them to third_party/ so I can install offline?'\n"
            f"pip returned: {e}"
        )

def compute_abi(py_ver: str) -> str:
    # py_ver like "3.11" -> abi "cp311"
    m = re.fullmatch(r"3\.(\d+)", py_ver.strip())
    if not m:
        fail("Invalid --python-version format. Use like: 3.11, 3.10, 3.9")
    return "cp3" + m.group(1) + "1" if int(m.group(1)) < 10 else f"cp3{m.group(1)}"

def download_base_wheels(py_ver: str):
    print("Downloading base project wheels for aarch64 ...")
    dest = REPO_DIR / "offline_wheels" / "base"
    abi = compute_abi(py_ver)             # 'cp311' for 3.11, etc.
    plat = "manylinux2014_aarch64"
    pip_download(BASE_REQUIREMENTS, dest, py_ver, abi, plat)
    print("Base wheels downloaded.")

def _try_download_hailo_file_sequence(candidates, dest_path: Path, desc: str):
    last_err = None
    for url in candidates:
        try:
            download_file(url, dest_path, desc)
            return
        except SystemExit as e:
            last_err = e
            continue
    if last_err:
        raise last_err

def download_hailo_wheels(py_ver: str):
    print("Downloading Hailo Python wheels (hailort & tappas_core_python_binding) ...")
    abi = compute_abi(py_ver)
    dest_dir = REPO_DIR / "offline_wheels" / "hailo"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # HailoRT wheel URL is versioned by Python + arch tags
    hailort_names = [
        f"hailort-{HAILORT_VERSION}-{abi}-{abi}-linux_aarch64.whl",
        f"hailort-{HAILORT_VERSION}-{abi}-{abi}-manylinux2014_aarch64.whl",
    ]
    hailort_urls = [f"{HAILO_BASE_URL}/{name}" for name in hailort_names]
    hailort_dest = dest_dir / hailort_names[-1]  # name doesn't matter; file content identical
    _try_download_hailo_file_sequence(hailort_urls, hailort_dest, "hailort wheel")

    # TAPPAS Core python binding may have build metadata; try a small set of candidates
    tappas_basenames = [
        f"tappas_core_python_binding-{TAPPAS_CORE_VERSION}-py3-none-any.whl",
        f"tappas_core_python_binding-{TAPPAS_CORE_VERSION}+1-1-py3-none-any.whl",
        f"tappas_core_python_binding-{TAPPAS_CORE_VERSION}+1-py3-none-any.whl",
    ]
    tappas_urls = [f"{HAILO_BASE_URL}/{name}" for name in tappas_basenames]
    tappas_dest = dest_dir / tappas_basenames[0]
    _try_download_hailo_file_sequence(tappas_urls, tappas_dest, "tappas_core_python_binding wheel")

    print("Hailo wheels downloaded.")

def clone_hailo_apps_infra(tag_or_branch: str):
    print(f"Cloning hailo-apps-infra (tag/branch: {tag_or_branch}) into third_party/ ...")
    dest = REPO_DIR / "third_party" / "hailo-apps-infra"
    if dest.exists():
        fail(f"Path already exists: {dest}\nRemove it and re-run.")
    run(["git", "clone", "--depth", "1", "--branch", tag_or_branch,
         "https://github.com/hailo-ai/hailo-apps-infra", str(dest)])
    # Validate that the expected Python package path is present:
    expected = dest / "hailo_apps"
    if not expected.exists():
        fail(
            f"Expected 'hailo_apps' folder not found inside {dest}\n"
            "The hailo-apps-infra repository layout may have changed.\n"
            "Ask: 'What is the correct module path in the current hailo-apps-infra tag?'"
        )
    print("hailo-apps-infra cloned.")

# def clone_git_dep_to_third_party(git_url: str, name_hint: str):
#     print(f"Cloning git dependency {git_url} into third_party/{name_hint} ...")
#     dest = REPO_DIR / "third_party" / name_hint
#     if dest.exists():
#         fail(f"Path already exists: {dest}\nRemove it and re-run.")
#     run(["git", "clone", "--depth", "1", git_url, str(dest)])
#     return dest

# def download_community_wheels(py_ver: str, include=True):
#     if not include:
#         print("Skipping community projects wheel downloads (by configuration).")
#         return
#     print("Downloading community projects wheels (large, may take time) ...")
#     abi = compute_abi(py_ver)
#     plat = "manylinux2014_aarch64"
#     for proj, reqs in COMMUNITY_REQUIREMENT_FILES.items():
#         print(f"  * Project: {proj}")
#         outdir = REPO_DIR / "offline_wheels" / "community" / proj
#         outdir.mkdir(parents=True, exist_ok=True)
#         pkgs = []
#         for r in reqs:
#             if r.startswith("git+"):
#                 # Clone to third_party/ for offline installs
#                 repo_url = r[4:]
#                 name_hint = "CLIP" if "openai/CLIP" in repo_url else Path(urlparse(repo_url).path).stem
#                 clone_git_dep_to_third_party(repo_url, name_hint)
#             else:
#                 pkgs.append(r)
#         if pkgs:
#             pip_download(pkgs, outdir, py_ver, abi, plat)
#     print("Community wheels download completed.")

def patch_gitignore():
    gi = REPO_DIR / ".gitignore"
    print("Patching .gitignore with offline cache paths ...")
    existing = gi.read_text(encoding="utf-8") if gi.exists() else ""
    # ensure a double newline at the end, then add our lines if not present
    if not existing.endswith("\n\n"):
        existing = existing.rstrip("\n") + "\n\n"
    # prevent duplicates
    lines = set(l.strip() for l in existing.splitlines() if l.strip())
    for entry in GITIGNORE_APPEND:
        if entry not in lines:
            existing += entry + "\n"
            lines.add(entry)
    gi.write_text(existing, encoding="utf-8")
    print("  .gitignore patched.")

def patch_config_yaml_for_local_infra():
    cfg = REPO_DIR / "config.yaml"
    if not cfg.exists():
        fail(f"Missing config.yaml at {cfg} after clone. The repo layout may have changed.")
    text = cfg.read_text(encoding="utf-8")
    # Replace hailo_apps_infra_path: "auto" -> "third_party/hailo-apps-infra"
    new_text = re.sub(
        r'(^\s*hailo_apps_infra_path:\s*)"(?:auto|[^"]*)"\s*',
        r'\1"third_party/hailo-apps-infra"\n',
        text,
        flags=re.MULTILINE
    )
    if text == new_text:
        print("  config.yaml: hailo_apps_infra_path set to third_party/hailo-apps-infra (or already was).")
    else:
        cfg_backup = cfg.with_suffix(".yaml.bak")
        shutil.copyfile(cfg, cfg_backup)
        cfg.write_text(new_text, encoding="utf-8")
        print("  config.yaml patched (backup saved as config.yaml.bak).")

# ---------- Mirror hailo-apps-infra Python dependencies (exact wheels + transitive deps) ----------
def _parse_apps_infra_dependencies(pyproject_path: Path):
    """
    Minimal, robust parser for the [project] dependencies array in pyproject.toml.
    Avoids adding an external TOML parser; works for the known simple format.
    """
    if not pyproject_path.exists():
        fail(f"pyproject.toml not found at: {pyproject_path}")

    text = pyproject_path.read_text(encoding="utf-8")
    # Find the block: [project] ... dependencies = [ ... ]
    # This is simplistic but resilient for the provided structure.
    deps_block = None
    in_project = False
    in_array = False
    buf = []
    for line in text.splitlines():
        if line.strip().startswith("[project]"):
            in_project = True
            continue
        if in_project and "dependencies" in line and "=" in line and "[" in line:
            in_array = True
            # capture after the first '['
            after = line.split("[", 1)[1]
            buf.append(after)
            continue
        if in_array:
            buf.append(line)
            if "]" in line:
                deps_block = "\n".join(buf)
                break

    if not deps_block:
        fail("Could not locate [project] dependencies array in pyproject.toml")

    # Extract quoted items "pkg[extras]==ver" or "pkg" etc.
    deps = []
    for m in re.finditer(r'"([^"]+)"', deps_block):
        item = m.group(1).strip()
        if item:
            deps.append(item)

    if not deps:
        fail("Parsed an empty dependencies list from pyproject.toml (unexpected).")

    return deps

def download_hailo_apps_infra_deps(py_ver: str, apps_infra_root: Path):
    """
    Read hailo-apps-infra/pyproject.toml, extract [project].dependencies,
    and `pip download` all of them for aarch64/cp311 (including transitives).
    """
    print("Resolving and downloading hailo-apps-infra dependencies (including transitives) for aarch64 ...")
    pyproject = apps_infra_root / "pyproject.toml"
    deps = _parse_apps_infra_dependencies(pyproject)

    # pip will resolve and download the exact versions it decides; that is our 'exact set' snapshot.
    dest = REPO_DIR / "offline_wheels" / "hailo_apps_infra"
    abi = compute_abi(py_ver)
    plat = "manylinux2014_aarch64"
    pip_download(deps, dest, py_ver, abi, plat)
    print("hailo-apps-infra dependency wheels downloaded.")

# ---------- Mirror the build backend (setuptools, wheel) used by hailo-apps-infra ----------
def download_hailo_apps_infra_build_backend(py_ver: str):
    """
    Mirror the build backend declared in hailo-apps-infra's pyproject:
      [build-system] requires = ["setuptools>=61.0", "wheel"]
    These must be installed on the Pi BEFORE `pip install -e third_party/hailo-apps-infra`
    to avoid any network access during editable install.
    """
    print("Downloading hailo-apps-infra build backend wheels (setuptools, wheel) ...")
    dest = REPO_DIR / "offline_wheels" / "hailo_apps_infra_build"
    abi = compute_abi(py_ver)
    plat = "manylinux2014_aarch64"
    pip_download(["setuptools>=61.0", "wheel"], dest, py_ver, abi, plat)
    print("hailo-apps-infra build backend wheels downloaded.")

# ---------- Write a Heredoc README with offline install steps ----------
def write_offline_readme(py_ver: str):
    readme_path = REPO_DIR / "OFFLINE_INSTALL_README.txt"
    contents = textwrap.dedent(f"""
    ================================================================================
    OFFLINE INSTALL GUIDE — Raspberry Pi 5 (aarch64, Python {py_ver})
    ================================================================================

    This repository folder contains:
      * Pre-fetched Hailo model HEFs and demo videos (./resources/...).
      * Python wheels for the base project (./offline_wheels/base).
      * Hailo Python wheels (hailort + tappas_core_python_binding) (./offline_wheels/hailo).
      * hailo-apps-infra code (./third_party/hailo-apps-infra) and its mirrored deps
        (./offline_wheels/hailo_apps_infra).
      * Build backend wheels for hailo-apps-infra (./offline_wheels/hailo_apps_infra_build).

    Prereqs on the Pi (already mirrored via apt by you):
      - python3-venv, python3, pip
      - python3-gi, python3-gi-cairo, gstreamer1.0-*, rpicam-apps, etc.
      - hailo-all (runtime + tappas core debs) already installed

    ------------------------------------------------------------------------------
    1) Activate this repo & venv
    ------------------------------------------------------------------------------
      cd hailo-rpi5-examples
      # NOTE: setup_env.sh contains a kernel guard; patch if necessary before sourcing.
      source ./setup_env.sh

    ------------------------------------------------------------------------------
    2) Install Python wheels OFFLINE (no network)
    ------------------------------------------------------------------------------
      # (A) Install build backend required by hailo-apps-infra editable install
      pip install --no-index --find-links offline_wheels/hailo_apps_infra_build setuptools wheel

      # (B) Base project deps (pytest etc.)
      pip install --no-index --find-links offline_wheels/base -r requirements.txt

      # (C) Hailo Python wheels (exact wheels provided)
      pip install --no-index --find-links offline_wheels/hailo \\
          offline_wheels/hailo/hailort-*.whl \\
          offline_wheels/hailo/tappas_core_python_binding-*.whl

      # (D) hailo-apps-infra dependencies (resolved + mirrored, including transitives)
      pip install --no-index --find-links offline_wheels/hailo_apps_infra \\
          numpy 'setproctitle' 'opencv-python' 'python-dotenv' 'pyyaml' 'gradio' \\
          'fastrtc' 'lancedb' 'matplotlib' 'Pillow'

      # (E) Finally, install hailo-apps-infra from local source (editable)
      pip install -e third_party/hailo-apps-infra

    ------------------------------------------------------------------------------
    3) (Optional) Ensure models are also available system-wide
    ------------------------------------------------------------------------------
      # If you want to seed /usr/local/hailo/resources/... ahead of time:
      # sudo mkdir -p /usr/local/hailo/resources/models/hailo8 /usr/local/hailo/resources/models/hailo8l
      # sudo cp resources/models/hailo8/*.hef  /usr/local/hailo/resources/models/hailo8/ || true
      # sudo cp resources/models/hailo8l/*.hef /usr/local/hailo/resources/models/hailo8l/ || true

    ------------------------------------------------------------------------------
    4) Run examples
    ------------------------------------------------------------------------------
      python basic_pipelines/detection_simple.py
      # or
      python basic_pipelines/detection.py --input rpi
      python basic_pipelines/pose_estimation.py
      python basic_pipelines/instance_segmentation.py
      python basic_pipelines/depth.py

    ------------------------------------------------------------------------------
    Notes
    ------------------------------------------------------------------------------
      • If anything tries to download, verify you used --no-index with --find-links
        and that the relevant wheels are present under ./offline_wheels.
      • If GStreamer plugins complain about TLS memory (libgomp), follow the repo
        doc’s LD_PRELOAD workaround and clear the gst registry cache.

    ================================================================================
    """).strip("\n") + "\n"
    readme_path.write_text(contents, encoding="utf-8")
    print(f"Wrote offline guide: {readme_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Download EVERYTHING (except apt) for hailo-rpi5-examples to run offline on a Pi.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--python-version", default="3.11",
                        help="Target CPython version for the Pi wheels (e.g., 3.11, 3.10).")
    parser.add_argument("--infra-tag", default="25.7.0",
                        help="hailo-apps-infra tag/branch to clone into third_party/hailo-apps-infra.")
    # parser.add_argument("--skip-community", action="store_true",
    #                     help="Skip downloading wheels/clones for community projects (saves time/space).")
    args = parser.parse_args()

    check_tools()
    assert_clean_cwd()
    clone_repo_exact()
    ensure_dirs()
    download_models_and_videos()
    download_base_wheels(args.python_version)
    download_hailo_wheels(args.python_version)
    clone_hailo_apps_infra(args.infra_tag)
    # Mirror build backend for hailo-apps-infra editable install
    download_hailo_apps_infra_build_backend(args.python_version)
    # After the clone, parse its pyproject and mirror ALL deps (with transitives) for aarch64/cp311
    download_hailo_apps_infra_deps(args.python_version, REPO_DIR / "third_party" / "hailo-apps-infra")
    # download_community_wheels(args.python_version, include=(not args.skip_community))  # DISABLED: see rationale
    patch_gitignore()
    patch_config_yaml_for_local_infra()
    write_offline_readme(args.python_version)

    print("\nALL DONE.\n")
    print(textwrap.dedent(f"""
        Next steps (on the Raspberry Pi, OFFLINE):
          1) Mirror/install APT packages listed in your plan (on the Pi).
          2) Copy the entire './hailo-rpi5-examples' folder to the Pi.
          3) In that folder on the Pi:
                source ./setup_env.sh
                # Install wheels fully OFFLINE:

                #   - (A) Build backend required by hailo-apps-infra editable install
                pip install --no-index --find-links offline_wheels/hailo_apps_infra_build setuptools wheel

                #   - (B) Base project wheels
                pip install --no-index --find-links offline_wheels/base -r requirements.txt

                #   - (C) Hailo wheels
                pip install --no-index --find-links offline_wheels/hailo offline_wheels/hailo/hailort-*.whl offline_wheels/hailo/tappas_core_python_binding-*.whl

                #   - (D) Apps-Infra dependencies (resolved snapshot, including transitives)
                pip install --no-index --find-links offline_wheels/hailo_apps_infra pyyaml gradio fastrtc lancedb matplotlib Pillow

                #   - (E) Apps-Infra package (editable, from local clone)
                pip install -e third_party/hailo-apps-infra

          4) Run examples, e.g.:
                python basic_pipelines/detection_simple.py
                python basic_pipelines/detection.py --input rpi
                python basic_pipelines/pose_estimation.py
                python basic_pipelines/instance_segmentation.py
                python basic_pipelines/depth.py

        A full, step-by-step guide is also saved to:
            ./hailo-rpi5-examples/OFFLINE_INSTALL_README.txt
    """).strip())

if __name__ == "__main__":
    main()
