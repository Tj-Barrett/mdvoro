"""Paths of the external binaries (tetgen, voro++).

Sources, later ones override earlier ones:

1. ``mdvoro/config.yaml`` (tracked): defaults, the bare commands ``tetgen`` and
   ``voro++`` looked up on PATH.
2. ``~/.config/mdvoro/config.yaml`` (``$XDG_CONFIG_HOME/mdvoro/config.yaml``):
   per-user paths, also works for non-editable installs.
3. ``mdvoro/config.local.yaml`` (git-ignored): per-checkout paths.
4. Environment variables ``MDVORO_TETGEN`` and ``MDVORO_VOROPP``.

Each file uses the same keys as config.yaml (``tetgen:``, ``voro++:``) and may
set only one of them. ``~`` and environment variables in paths are expanded.
"""
import os
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KEYS = {"tetgen": "alias_tetgen", "voro++": "alias_voropp"}
ENV = {"tetgen": "MDVORO_TETGEN", "voro++": "MDVORO_VOROPP"}


def config_files():
    """candidate config files, lowest priority first"""
    xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return [os.path.join(ROOT_DIR, "config.yaml"),
            os.path.join(xdg, "mdvoro", "config.yaml"),
            os.path.join(ROOT_DIR, "config.local.yaml")]


def _expand(path):
    return os.path.expandvars(os.path.expanduser(str(path)))


def read_config(args=None):
    if args is None:
        args = config_args()
    for conf in config_files():
        if not os.path.isfile(conf):
            continue
        with open(conf) as stream:
            yamlin = yaml.safe_load(stream) or {}
        for key, attr in KEYS.items():
            if yamlin.get(key):
                setattr(args, attr, _expand(yamlin[key]))
    for key, var in ENV.items():
        if os.environ.get(var):
            setattr(args, KEYS[key], _expand(os.environ[var]))
    return args


class config_args:
    def __init__(self):
        self.alias_tetgen = None
        self.alias_voropp = None
