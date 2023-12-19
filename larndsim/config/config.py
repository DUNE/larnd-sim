import os
import glob
import yaml
import pathlib
import copy

CONFIG_FILENAME = os.path.join(pathlib.Path(__file__).parents[0],'config.yaml')

CONFIG_MAP = yaml.safe_load(open(CONFIG_FILENAME,'r'))

MODULE_DIR = pathlib.Path(__file__).parents[1]
CONFIG_DIR = dict(
    SIM_PROPERTIES=f'{MODULE_DIR}/simulation_properties/',
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

        # allow the yaml have more keys than which defined in "CONFIG_DIR"
#        for key in cfg_map.keys():
#            if not key in CONFIG_DIR.keys():
#                raise RuntimeError(f'[CONFIG TEST ERROR] Unknown key {key} in the config {cfg_name}')

def get_config(keyname):

    if not keyname in list_config_keys():
        raise KeyError(f'Key {keyname} not in supported keywords {list_config_keys()}')

    cfg_map = CONFIG_MAP[keyname]

    res = {}
    # allow user to provide the full path to the file
#    for key in CONFIG_DIR.keys():
#        res[key] = os.path.join(CONFIG_DIR[key], cfg_map[key])

    for key in cfg_map.keys():
        if key not in CONFIG_DIR.keys():
            res[key] = cfg_map[key]
        else:
            if isinstance(cfg_map[key], str):
                if '/' in cfg_map[key]:
                    res[key] = cfg_map[key]
                else:
                    res[key] = os.path.join(CONFIG_DIR[key], cfg_map[key])
            elif isinstance(cfg_map[key], list):
                res[key] = []
                for this_config in cfg_map[key]:
                    if '/' in this_config:
                        res[key].append(this_config) 
                    else:
                        res[key].append(os.path.join(CONFIG_DIR[key], this_config))

    return res

