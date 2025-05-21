import os

def get_TMP_DIR() -> str:
    """Return the default cache directory inside the package install path."""

    filepath_to_this_file = os.path.abspath(__file__)
    tmp_dir = os.path.join(os.path.dirname(filepath_to_this_file), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def set_TMP_DIR(path: str) -> str:
    """Override the cache directory used by :mod:`quick_vllm`.

    The directory is created if it does not already exist and the new path is
    returned for convenience.
    """

    global TMP_DIR
    TMP_DIR = os.path.abspath(path)
    os.makedirs(TMP_DIR, exist_ok=True)
    return TMP_DIR


TMP_DIR = get_TMP_DIR()
