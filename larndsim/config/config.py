import os
import glob
import yaml
import pathlib
import copy

CONFIG_FILENAME = os.path.join(pathlib.Path(__file__).parents[0],'config.yaml')

CONFIG_MAP = yaml.safe_load(open(CONFIG_FILENAME,'r'))

MODULE_DIR = pathlib.Path(__file__).parents[1]
CONFIG_DIR = dict(SIM_PROPERTIES=f'{MODULE_DIR}/simulation_properties/',
    PIXEL_LAYOUT=f'{MODULE_DIR}/pixel_layouts/',
    DET_PROPERTIES=f'{MODULE_DIR}/detector_properties/',
    RESPONSE=f'{MODULE_DIR}/bin',
    LIGHT_LUT=f'{MODULE_DIR}/bin',
    LIGHT_DET_NOISE=f'{MODULE_DIR}/bin',
    )


def list_config_keys():
    return CONFIG_MAP.keys()

def print_configs():
    print(yaml.dump(CONFIG_MAP))

def test_configs():

    for cfg_name,cfg_map in CONFIG_MAP.items():

        for key in CONFIG_DIR.keys():
            if not key in cfg_map.keys():
                raise RuntimeError(f'[CONFIG TEST ERROR] Key {key} missing in the config {cfg_name}')

        for key in cfg_map.keys():
            if not key in CONFIG_DIR.keys():
                raise RuntimeError(f'[CONFIG TEST ERROR] Unknown key {key} in the config {cfg_name}')

def get_config(keyname):

    if not keyname in list_config_keys():
        raise KeyError(f'Key {keyname} not in supported keywords {list_config_keys()}')

    cfg_map = CONFIG_MAP[keyname]

    res = {}
    for key in CONFIG_DIR.keys():
        res[key] = os.path.join(CONFIG_DIR[key], cfg_map[key])

    return res

