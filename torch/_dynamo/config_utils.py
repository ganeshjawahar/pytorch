import contextlib
import copy

import dataclasses
import inspect
import os

import pickle

import tempfile
import unittest
from os.path import abspath, dirname
from types import FunctionType, ModuleType
from typing import Any, Dict, Optional, Set, Type
from unittest import mock

import torch

from . import external_utils


class ConfigMixin:

    def save_config(self):
        config = copy.copy(self)
        for key in self._save_config_ignore:
            delattr(config, key)
        return pickle.dumps(config, protocol=2)

    def load_config(self, content):
        state = pickle.loads(content)
        self.__dict__.update(state.__dict__)
        return self

    def _update_single(self, key, val):
        pieces = key.split('.')
        current = self
        for token in pieces[:-1]:
            current = getattr(current, token)
        setattr(current, pieces[-1], val)

    def update(self, content_dict):
        for k, v in content_dict.items():
            self._update_single(k, v)

    def is_fbcode(self):
        return not hasattr(torch.version, "git_version")

    def patch(self, arg1=None, arg2=None, **kwargs):
        """
        Decorator and/or context manager to make temporary changes to a config.

        As a decorator:

            @config.patch("name", val)
            @config.patch(name1=val1, name2=val2):
            @config.patch({"name1": val1, "name2", val2})
            def foo(...):
                ...

        As a context manager:

            with config.patch("name", val):
                ...
        """
        if arg1 is not None:
            if arg2 is not None:
                # patch("key", True) syntax
                changes = {arg1: arg2}
            else:
                # patch({"key": True}) syntax
                changes = arg1
            assert not kwargs
        else:
            # patch(key=True) syntax
            changes = kwargs
            assert arg2 is None
        assert isinstance(changes, dict), f"expected `dict` got {type(changes)}"
        prior = {}
        config = self

        class ConfigPatch(ContextDecorator):
            def __enter__(self):
                assert not prior
                for key in changes.keys():
                    # KeyError on invalid entry
                    prior[key] = getattr(config, key)
                config.update(changes)

            def __exit__(self, exc_type, exc_val, exc_tb):
                config.update(prior)
                prior.clear()

        return ConfigPatch()


# Types saved/loaded in configs
CONFIG_TYPES = (int, float, bool, type(None), str, list, set, tuple, dict)


def make_config_dataclass(name, config_module):
    fields = []
    for fname, annotation in config_module.__annotations__.items():
        default_value = getattr(config_module, fname)
        def f():
            return default_value
        field = dataclasses.field(default_factory=f)
        fields.append((fname, annotation, field))
    return dataclasses.make_dataclass(name, fields, bases=(ConfigMixin, ))


class ContextDecorator(contextlib.ContextDecorator):
    """
    Same as contextlib.ContextDecorator, but with support for
    `unittest.TestCase`
    """

    def __call__(self, func):
        if isinstance(func, type) and issubclass(func, unittest.TestCase):

            class _TestCase(func):
                @classmethod
                def setUpClass(cls):
                    self.__enter__()
                    try:
                        super().setUpClass()
                    except Exception:
                        self.__exit__(None, None, None)
                        raise

                @classmethod
                def tearDownClass(cls):
                    try:
                        super().tearDownClass()
                    finally:
                        self.__exit__(None, None, None)

            _TestCase.__name__ = func.__name__
            return _TestCase

        return super().__call__(func)


def patch_object(obj, name, value):
    """
    Workaround `mock.patch.object` issue with ConfigModule
    """
    if isinstance(obj, ConfigMixin):
        return obj.patch(name, value)
    return mock.patch.object(obj, name, value)
