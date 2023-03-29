"""
Set global variables with detector and physics properties
"""
from . import detector, light, sim

def load_properties(detprop_file, pixel_file, sim_file):
    """
    The function loads the detector properties,
    the pixel geometry, and the simulation YAML files
    and stores the constants as global variables

    Args:
        detprop_file (str): detector properties YAML filename
        pixel_file (str): pixel layout YAML filename
        sim_file (str): simulation properties YAML filename
    """
    detector.set_detector_properties(detprop_file, pixel_file)
    light.set_light_properties(detprop_file)
    sim.set_simulation_properties(sim_file)
