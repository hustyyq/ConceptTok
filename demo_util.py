# Concept Tokenizer note: this helper was reduced from the 1d-tokenizer demo utilities.
"""Utilities for loading OmegaConf configs used by training and sampling scripts."""

from omegaconf import OmegaConf


def get_config_cli():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)


def get_config(config_path):
    return OmegaConf.load(config_path)
