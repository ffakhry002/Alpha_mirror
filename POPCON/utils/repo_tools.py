"""
Helper functions for Git info
"""

from pathlib import Path
import os
import subprocess
import tomllib

def get_repo_root() -> Path:
    """
    Returns the path of the Git repo as a Path object
    """
    return Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"]
    ).decode().strip())

def get_git_hash() -> str:
    """
    Returns the Git hash of the current commit
    """
    return subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']
    ).decode('ascii').strip()

def load_config():
    config_path = get_repo_root() / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return config