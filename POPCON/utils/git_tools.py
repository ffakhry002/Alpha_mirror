"""
Helper functions for Git info
"""

import subprocess
from pathlib import Path

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