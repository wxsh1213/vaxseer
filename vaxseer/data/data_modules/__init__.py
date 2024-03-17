"""isort:skip_file"""

import importlib
import os

DM_REGISTRY = {}

def register_dm(name):
    def register_model_cls(cls):
        if name in DM_REGISTRY:
            raise ValueError("Cannot register duplicate data module ({})".format(name))
        DM_REGISTRY[name] = cls
        return cls
    return register_model_cls

def build_data_module(name):
    if name in DM_REGISTRY:
        return DM_REGISTRY[name]
    else:
        raise ValueError("Cannot find data module %s" % name)

def import_data_modules(dm_dir, namespace):
    for file in os.listdir(dm_dir):
        path = os.path.join(dm_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            model_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + model_name)

# automatically import any Python files in the models/ directory
dm_dir = os.path.dirname(__file__)
import_data_modules(dm_dir, "data.data_modules")
