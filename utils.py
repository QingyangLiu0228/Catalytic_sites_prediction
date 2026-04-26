#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils.py

Helpers for config loading, snapshotting, and dict -> Namespace conversion.
The rest of the codebase uses `cfg.lr` / `cfg.csv_path` style access, so we
convert the loaded YAML dict into an argparse.Namespace-like object.
"""

import argparse
import copy
from pathlib import Path

import yaml


# ============================================================================
# Config loading
# ============================================================================
_DEFAULT_CONFIG_NAME = "default.yaml"


def load_config(yaml_path):
    """
    Load a YAML config file.

    If the file is NOT named 'default.yaml', we first load 'default.yaml'
    from the same directory and merge the target file on top. This lets
    debug.yaml (and any other experiment configs) override only the fields
    they need.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError("Config file not found: {}".format(yaml_path))

    with yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # If this is not default.yaml itself, merge on top of default.yaml
    if yaml_path.name != _DEFAULT_CONFIG_NAME:
        default_path = yaml_path.parent / _DEFAULT_CONFIG_NAME
        if default_path.exists():
            with default_path.open("r", encoding="utf-8") as f:
                default_cfg = yaml.safe_load(f) or {}
            merged = copy.deepcopy(default_cfg)
            merged.update(cfg)
            cfg = merged
        else:
            print("[utils] Warning: no default.yaml found next to {}, "
                  "running with only its fields.".format(yaml_path.name))

    return cfg


def save_config_snapshot(cfg_dict, run_dir, run_stamp=None):
    """
    Save the *resolved* config under run_dir.

    If run_stamp is set (e.g. %Y%m%d_%H%M%S), writes config_<stamp>.yaml so
    repeated runs in the same run_seed* folder do not overwrite each other.
    If run_stamp is None, keeps legacy name config.yaml.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    name = "config_{}.yaml".format(run_stamp) if run_stamp else "config.yaml"
    with (run_dir / name).open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, default_flow_style=False, sort_keys=False)


# ============================================================================
# dict -> Namespace conversion
# ============================================================================
def dict_to_namespace(d):
    """
    Convert a (possibly nested) dict into argparse.Namespace so that callers
    can use `cfg.lr` instead of `cfg['lr']`. This keeps the training code
    compatible with the pre-refactor argparse style.
    """
    ns = argparse.Namespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, dict_to_namespace(v))
        else:
            setattr(ns, k, v)
    return ns


def namespace_to_dict(ns):
    """Inverse of dict_to_namespace, used when writing config.yaml snapshots."""
    out = {}
    for k, v in vars(ns).items():
        if isinstance(v, argparse.Namespace):
            out[k] = namespace_to_dict(v)
        else:
            out[k] = v
    return out
