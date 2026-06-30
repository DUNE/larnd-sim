import os
import glob
import importlib
import yaml
import pathlib
import copy
import warnings

from larndsim import consts
from larndsim.consts.detector import get_module_ids

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
    PIXEL_THRESHOLDS_FILE=f'{MODULE_DIR}/bin',
    PIXEL_PEDESTALS_FILE=f'{MODULE_DIR}/bin',
    )

def list_config_keys():
    return CONFIG_MAP.keys()

def print_configs():
    print(yaml.dump(CONFIG_MAP))

def test_configs():

    for cfg_name,cfg_map in CONFIG_MAP.items():

        for key in CONFIG_DIR.keys():
            if key == 'PIXEL_THRESHOLDS_FILE' or key == 'PIXEL_PEDESTALS_FILE':
                # Don't throw an error for optional keys
                continue
            if not key in cfg_map.keys():
                raise RuntimeError(f'[CONFIG TEST ERROR] Key {key} missing in the config {cfg_name}')

        # allow the yaml have more keys than which defined in "CONFIG_DIR"
#        for key in cfg_map.keys():
#            if not key in CONFIG_DIR.keys():
#                raise RuntimeError(f'[CONFIG TEST ERROR] Unknown key {key} in the config {cfg_name}')

def get_config(keyname: str):
    """Read config parameters from a section of config.yaml.

    Valid parameters:
        pixel_layout (str or list): path of the YAML file containing the pixel
            layout and connection details.
        detector_properties (str): path of the YAML file containing
            the detector properties
        simulation_properties (str): path of the YAML file containing
            the simulation properties
        response_file (str or list): path of the Numpy array containing the pre-calculated
            field responses. 
        light_lut_file (str, optional): path of the Numpy array containing the light
            look-up table. 
        light_det_noise_filename (str, optional): path of the Numpy array containing the light noise information
        bad_channels (str, optional): path of the YAML file containing the channels to be
            disabled. Defaults to None
        pixel_thresholds_file (str or list, optional): path to npz file containing pixel thresholds. Defaults
            to None.
        pixel_gains_file (str or list): path to npz file containing pixel gain values. Defaults to None (the value of fee.GAIN)
        pixel_pedestals_file (str or list, optional): path to npx files containing pixel pedestals. Defaults to None.

    For the parameters that can be either a str or a list, if a list is
    provided, then it is also necessary to specify e.g. pixel_thresholds_id,
    mapping each module to an element of pixel_thresholds_file

    Args:
        keyname: Name of the configuration (i.e. the section in config.yaml).

    """

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
                if cfg_map[key] == "None":
                    res[key] = None
                elif '/' in cfg_map[key]:
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

    check_config(res)

    return res


def check_config(cfg: dict):
    pixel_layout_id = cfg.get('PIXEL_LAYOUT_ID', None)
    response_id = cfg.get('RESPONSE_FILE_ID', None)

    if pixel_layout_id and response_id:
        if pixel_layout_id != response_id:
            warnings.warn("Simulation with module variation activated, the pixel layout and response files may not be consistent with each other. Please double check!")


def _load_mod2mod_prop(cfg_files: list[str], ids: list[int], n_modules: int, message=""):
    if not isinstance(cfg_files, list):
        return [cfg_files]

    if ids is None:
        if isinstance(cfg_files, list) and len(cfg_files) != n_modules:
            raise KeyError(f"Simulation with module variation activated, but the number of {message} is incorrect!")
        elif isinstance(cfg_files, list) and len(cfg_files) == n_modules:
            warnings.warn("Simulation with module variation activated, using default orders for the {message}.")
    else:
        if len(ids) != n_modules or max(ids) >= len(cfg_files):
            raise KeyError(f"Simulation with module variation activated, but the number of pointer for {message} is incorrect!")
        else:
            module_files = [cfg_files[idx] for idx in ids]
            cfg_files = module_files

    return cfg_files


def load_mod2mod_prop(cfg: dict, key: str):
    # TODO: Pick one convention
    if key.endswidth('_FILE'):
        id_key = key[:-5] + '+ID'
    else:
        id_key = key + '_ID'

    mod_ids = get_module_ids(cfg['DET_PROPERTIES'])
    n_modules = len(mod_ids)

    if not cfg.get(key):
        warnings.warn(f'{key} not provided; using the default')

    return _load_mod2mod_prop(cfg.get(key), cfg.get(id_key), n_modules, key)


def reload_modules(detector_properties: list[str], pixel_layout: list[str], response_file: list[str], i_mod: int):
    """Reload modules after variables have been redefined."""
    consts.detector.set_detector_properties(detector_properties, pixel_layout, response_file[i_mod-1], i_mod)
    consts.light.set_light_properties(detector_properties)

    from larndsim import pixels_from_track, active_volume, detsim, light_sim, lightLUT, fee
    importlib.reload(pixels_from_track)
    importlib.reload(active_volume)
    importlib.reload(detsim)
    importlib.reload(light_sim)
    importlib.reload(lightLUT)
    importlib.reload(fee)
