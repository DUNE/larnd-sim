"""
Set global variables with detector and physics properties
"""
from . import detector, light, sim, ff_induction

def load_properties(detprop_file, pixel_file, response_file, sim_file, farfield_file=None):
    """
    The function loads the detector properties,
    the pixel geometry, and the simulation YAML files
    and stores the constants as global variables

    Args:
        detprop_file (str): detector properties YAML filename
        pixel_file (str): pixel layout YAML filename
        response_file (str): pixel response filename
        sim_file (str): simulation properties YAML filename
        farfield_file (str, optional): far-field induction properties YAML filename
    """
    sim.set_simulation_properties(sim_file) # must be first!
    if farfield_file:
        ff_induction.set_ff_induction_properties(farfield_file)

    detector.set_detector_properties(detprop_file, pixel_file, response_file)
    light.set_light_properties(detprop_file)

