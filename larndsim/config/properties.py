"""Read nominal properties and resolve variations before installing any globals."""

from collections.abc import Mapping
from copy import deepcopy
from math import isfinite
from numbers import Real
from pathlib import Path
import warnings

import yaml

CONFIG_PATH = Path(__file__).parent


def read_property_file(source):
    """Read YAML or copy an already loaded mapping."""
    if isinstance(source, Mapping):
        return deepcopy(dict(source))
    with open(source) as stream:
        properties = yaml.load(stream, Loader=yaml.FullLoader)
    if not isinstance(properties, dict):
        raise ValueError(f'Properties must be a mapping: {source}')
    return properties


def read_defaults():
    """Return fresh defaults, independent of previously installed globals."""
    return {
        'detector': read_property_file(CONFIG_PATH / 'detector_defaults.yaml'),
        'physics': read_property_file(CONFIG_PATH / 'physics.yaml'),
    }


def read_properties(detector_properties, physics_properties=None):
    """Collect nominal inputs without changing simulation state."""
    properties = read_defaults()
    properties['detector'].update(read_property_file(detector_properties))
    if physics_properties is not None:
        overrides = read_property_file(physics_properties)
        unknown = set(overrides) - set(properties['physics'])
        if unknown:
            raise ValueError(f'Unknown physics properties: {unknown}')
        properties['physics'].update(overrides)
    return properties


def _validate(value, definition, parameter):
    if 'choices' in definition:
        if not isinstance(value, str) or value not in definition['choices']:
            raise ValueError(f'{parameter} must be one of {definition["choices"]}')
        return value
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f'{parameter} must be a finite number')
    if 'minimum_exclusive' in definition and value <= definition['minimum_exclusive']:
        raise ValueError(f'{parameter} must be greater than {definition["minimum_exclusive"]}')
    if 'minimum' in definition and value < definition['minimum']:
        raise ValueError(f'{parameter} must be at least {definition["minimum"]}')
    if 'maximum' in definition and value > definition['maximum']:
        raise ValueError(f'{parameter} must be at most {definition["maximum"]}')
    return float(value)


def _select_module(value, module_ids, i_module, parameter):
    if not isinstance(value, list):
        return value
    if not value or len(value) not in (1, len(module_ids)):
        raise ValueError(f'{parameter} must have one value or one per module')
    if len(value) == 1:
        return value[0]
    if i_module < 0:
        warnings.warn(f'{parameter}: module variation is disabled; using the first value')
        return value[0]
    if i_module not in module_ids:
        raise ValueError(f'Unknown module ID {i_module}')
    return value[module_ids.index(i_module)]


def resolve_properties(nominal, config=None, i_module=-1, registry=None):
    """Return effective properties and provenance; never mutate nominal inputs.

    Definitions describe source keys, validation and module selection. Only
    independent inputs are registered: derived quantities cannot be overridden.
    """
    config = {} if config is None else config
    registry = (read_property_file(CONFIG_PATH / 'systematics.yaml')
                if registry is None else registry)
    definitions = registry['parameters']
    requests = config.get('DETECTOR_SYSTEMATICS', {})
    if not isinstance(requests, dict):
        raise ValueError('DETECTOR_SYSTEMATICS must be a mapping')
    unknown = set(requests) - set(definitions)
    if unknown:
        raise ValueError(f'Unsupported detector systematics: {unknown}')
    profile = config.get('SYSTEMATICS_PROFILE')
    effective = deepcopy(nominal)
    module_ids = list(effective['detector'].get('module_to_tpcs', {}))
    metadata = {}
    for parameter, definition in definitions.items():
        section, key = definition['section'], definition['key']
        value = nominal[section][key]
        if definition.get('per_module', False):
            value = _select_module(value, module_ids, i_module, parameter)
        baseline = _validate(value, definition, parameter)
        requested = requests.get(parameter, 'nominal')
        record = dict(nominal=baseline, requested=deepcopy(requested),
                      units=definition['units'], profile=profile)
        value = baseline
        if parameter in requests:
            if isinstance(requested, str) and requested in ('up', 'down'):
                if 'choices' in definition:
                    raise ValueError(f'{parameter} only accepts an explicit model value')
                profiles = registry['profiles']
                if (not isinstance(profile, str) or profile not in profiles
                        or parameter not in profiles[profile]):
                    raise ValueError(f'No {parameter} uncertainty registered for profile {profile!r}')
                uncertainty_definition = profiles[profile][parameter]
                uncertainty = uncertainty_definition['relative_uncertainty']
                if (isinstance(uncertainty, bool) or not isinstance(uncertainty, Real)
                        or not isfinite(uncertainty) or uncertainty < 0):
                    raise ValueError(f'{parameter} uncertainty must be finite and non-negative')
                if uncertainty_definition.get('demonstration', False):
                    warnings.warn(f'{profile}: illustrative {parameter} uncertainty; not approved for production')
                value = baseline * (1 + (1 if requested == 'up' else -1) * uncertainty)
                record['definition'] = deepcopy(uncertainty_definition)
            elif isinstance(requested, dict) and set(requested) == {'value'}:
                value = requested['value']
            else:
                raise ValueError(f'{parameter} must be up, down, or a mapping containing only value')
        value = _validate(value, definition, parameter)
        effective[section][key] = value
        record['value'] = value
        metadata[parameter] = record
    return effective, metadata
