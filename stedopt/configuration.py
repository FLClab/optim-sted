
import os
import yaml
import json
import copy
import pprint
import ast

import collections.abc

def update(d, u):
    """
    Recursively update the `key:value` of a dictionary with another dictionary

    :param d: A `dict` to be updated
    :param u: A `dict` of the update

    :returns : The updated `dict`
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

class Configurator:
    """
    `Configurator` that allows to load a configuration file.
    """
    def __init__(self, config, configpath=os.path.join(".", "configs"), options=None, load_base=True):
        """
        Instantiates the `Configurator`

        :param config: A `str` of the file OR a `dict` of the configuration file
        :param configpath: A `str` of the config folder
        :param options: A `list` of options to overwrite in the configuration
        :param load_base: A `bool` to load the base configuration file
        """
        # Reads the configuration file
        config = self.load_config(config)
        self.configpath = configpath

        if load_base:
            self.config = self.update_base_config(config)
        else:
            self.config = config
        self.cached = None

        self.update_config_options(options)

    def __getattr__(self, attr):
        """
        Implements the `__getattr__` method of the `Configurator`
        """
        if attr in self.__dict__:
            return self.__dict__[attr]
        if attr in self.config:
            return self.config[attr]

    def __setattr__(self, attr, value):
        """
        Implements the `__setattr__` method of the `Configurator`
        """
        self.__dict__[attr] = value

    def update_config_options(self, options):
        """
        Updates the current configuration with the given options

        Options are given as a list of strings in the following format:
        `[path/to/key, value, path/to/key2, value2, ...]`

        To access inner dictionaries, use `/` as a separator

        :param options: A `list` of options
        """
        if isinstance(options, type(None)):
            return
        for i in range(0, len(options), 2):
            keys, value = options[i].split("/"), options[i + 1]
            _dict = self.config
            for key in keys:
                if key == keys[-1]:
                    # Ensures desired value has the same type
                    _type = type(_dict[key])
                    if isinstance(_dict[key], (float, int, bool)):
                        _dict[key] = _type(eval(value))
                    elif isinstance(_dict[key], (list, tuple)):
                        _dict[key] = ast.literal_eval(value)
                    else:
                        _dict[key] = _type(value)
                else:
                    _dict = _dict[key]

    def update_base_config(self, config):
        """
        Updates the current configuration with the base configuration file.

        The order of the `_BASE_` files is important if there are many. We update
        the base configuration file with the current configuration file.

        Here's an example of the expected behavior
        A = {
            "key1" : 1,
            "key2" : 2,
            "key3" : 3
        }
        B = {
            "key1" : 2,
            "key4" : 4
        }

        # Case 1
        C = {
            "_BASE_" : [A, B]
        }
        >>> C = {
            "_BASE_" : [A, B]
            "key1" : 1,
            "key2" : 2,
            "key3" : 3,
            "key4" : 4
        }

        # Case 2
        C = {
            "_BASE_" : [B, A]
        }
        >>> C = {
            "_BASE_" : [B, A]
            "key1" : 2,
            "key2" : 2,
            "key3" : 3,
            "key4" : 4
        }

        :param config: A `dict` of the configuration file

        :returns : An updated configuration
        """
        if "_BASE_" in config:
            if isinstance(config["_BASE_"], (list, tuple)):
                for item in config["_BASE_"]:
                    base_config = self.load_config(item)
                    base_config = self.update_base_config(base_config)
                    config = update(base_config, config)
            else:
                base_config = self.load_config(config["_BASE_"])
                base_config = self.update_base_config(base_config)
                config = update(base_config, config)
        return config

    def get_config(self):
        """
        Gets the configuration file from the `Configurator`

        :returns : A `dict` of the configuration file
        """
        return self.config

    def load_config(self, config):
        """
        Loads a configuration file. 
        
        If `config` is already a `dict` we simply return a deep copy of it.

        :param config: A `str` or `dict` of the configuration file

        :returns : A `dict` of the configuration file
        """
        if isinstance(config, str):
            if os.path.isfile(config):
                config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
            else:
                config = yaml.load(open(os.path.join(self.configpath, config), "r"), Loader=yaml.FullLoader)
        return copy.deepcopy(config)

    def __repr__(self):
        """
        Implements the print method of the `Configurator`
        """
        to_print = []
        to_print.append("========================================================")
        to_print.append("==================== Configuration =====================")
        to_print.append("========================================================")
        to_print.append(json.dumps(self.config, indent=2, sort_keys=True))
        to_print.append("========================================================")
        to_print.append("================== End Configuration ===================")
        to_print.append("========================================================\n")
        return "\n".join(to_print)

if __name__ == "__main__":

    A = {
        "key1" : 1,
        "key2" : 2,
        "key3" : 3
    }
    B = {
        "key1" : 2,
        "key4" : 4
    }

    # Case 1
    C = {
        "_BASE_" : [A, B]
    }
    configurator = Configurator(C)
    print(configurator)

    # Case 2
    C = {
        "_BASE_" : [B, A]
    }
    configurator = Configurator(C)
    print(configurator)
