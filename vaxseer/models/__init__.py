"""isort:skip_file"""

import argparse
import importlib
import os

MODEL_REGISTRY = {}

def register_model(name):
    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(name))
        MODEL_REGISTRY[name] = cls
        return cls
    return register_model_cls


def build_model(name):
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    else:
        raise ValueError("Cannot find model named %s" % name)

def import_models(models_dir, namespace):
    for file in os.listdir(models_dir):
        path = os.path.join(models_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            model_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + model_name)

            # # extra `model_parser` for sphinx
            # if model_name in MODEL_REGISTRY:
            #     parser = argparse.ArgumentParser(add_help=False)
            #     group_archs = parser.add_argument_group("Named architectures")
            #     group_archs.add_argument(
            #         "--arch", choices=ARCH_MODEL_INV_REGISTRY[model_name]
            #     )
            #     group_args = parser.add_argument_group(
            #         "Additional command-line arguments"
            #     )
            #     MODEL_REGISTRY[model_name].add_args(group_args)
            #     globals()[model_name + "_parser"] = parser


# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
import_models(models_dir, "models")
