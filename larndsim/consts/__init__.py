"""Install effective properties and calculate dependent detector state."""

from . import detector, light, sim, ff_induction, physics
from ..config.properties import read_properties, resolve_properties


def set_properties(properties, pixel_file, response_file, i_module=-1):
    """Install resolved inputs. No systematics are interpreted here."""
    physics.set_physics_properties(properties['physics'])
    detector.set_detector_properties(
        properties['detector'], pixel_file, response_file, i_module=i_module)
    light.set_light_properties(properties['detector'])
    return {'detector': detector.derived_properties()}


def load_properties(detprop_file, pixel_file, response_file, sim_file, farfield_file=None,
                    systematics_config=None, i_module=-1, physics_file=None):
    """Compatibility entry point: read, resolve, then install properties.

    Callers must prepare simulation consumers after loading and before launching
    compiled functions. The CLI does this through prepare_simulation().
    """
    nominal = read_properties(detprop_file, physics_file)
    effective, metadata = resolve_properties(nominal, systematics_config, i_module)
    sim.set_simulation_properties(sim_file)  # Must precede detector setup.
    if farfield_file:
        ff_induction.set_ff_induction_properties(farfield_file)
    set_properties(effective, pixel_file, response_file, i_module)
    return metadata
