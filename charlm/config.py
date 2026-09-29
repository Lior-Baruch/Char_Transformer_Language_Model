"""Loading dataclass configs from JSON files with command-line overrides."""
import dataclasses
import json
import typing


def from_dict(cls, data):
    """ build dataclass cls from a (nested) dict, rejecting unknown keys so typos don't go unnoticed """
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} option(s): {', '.join(sorted(unknown))}")
    kwargs = {}
    for name, value in data.items():
        if dataclasses.is_dataclass(hints[name]) and isinstance(value, dict):
            value = from_dict(hints[name], value)
        kwargs[name] = value
    return cls(**kwargs)


def parse_override(text):
    """ 'model.n_layer=4' -> (['model', 'n_layer'], 4); values are parsed as JSON when possible """
    key, sep, raw = text.partition('=')
    if not sep:
        raise ValueError(f"override {text!r} must look like key=value")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw  # plain strings don't need quotes
    return key.split('.'), value


def load_config(cls, path=None, overrides=()):
    """ defaults of cls, updated by the JSON file at path, updated by key=value overrides """
    data = {}
    if path:
        with open(path) as f:
            data = json.load(f)
    for override in overrides:
        keys, value = parse_override(override)
        target = data
        for k in keys[:-1]:
            target = target.setdefault(k, {})
        target[keys[-1]] = value
    return from_dict(cls, data)


def to_dict(config):
    return dataclasses.asdict(config)
