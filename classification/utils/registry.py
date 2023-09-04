# -*- coding: utf-8 -*-
# @Time : 2022/11/29 上午11:08
# @Author : YANG.C
# @File : registry.py
import logging


class Registry:
    def __init__(self, register_name):
        self._dict = {}
        self._name = register_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f'Value of a Registry must be a callable!\n value: {value}')
        if key is None:
            key = value.__name__
        if key in self._dict:
            logging.warning(f'Key {key} is already in Registry {self._name}')
        self._dict[key] = value

    def register_module(self, target):
        """Decorator to register a function or class"""

        def add(key, value):
            self[key] = value
            return value

        if callable(target):
            return add(None, target)

        return lambda x: add(target, x)

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        return self._dict.keys()


if __name__ == '__main__':
    reg = Registry('models')

    BACKBONES = reg
    NECKS = reg
    print(id(BACKBONES), id(NECKS))
    print(reg._name)
    print(reg.keys())


    @reg.register_module
    class ModelA:
        def __init__(self):
            pass


    print(reg.keys())
    print(reg['ModelA'])
