from copy import deepcopy
from contextlib import nullcontext

import pytest

from larndsim.config import get_config, read_properties, resolve_properties
from larndsim.config.properties import CONFIG_PATH, read_property_file


@pytest.fixture
def nominal():
    return read_properties(get_config('ndlar')['DET_PROPERTIES'])


@pytest.mark.parametrize('preset, birks, field', [
    ('ndlar', 0.800, 0.500),
    ('ndlar_birks_up_demo', 0.820, 0.500),
    ('ndlar_birks_down_efield_up_demo', 0.780, 0.510),
    ('ndlar_birks_custom_demo', 0.780, 0.500),
])
def test_presets(nominal, preset, birks, field):
    with pytest.warns(UserWarning, match='illustrative') if 'up_demo' in preset else nullcontext():
        effective, metadata = resolve_properties(nominal, get_config(preset), i_module=1)
    assert effective['physics']['BIRKS_Ab'] == pytest.approx(birks)
    assert effective['detector']['e_field'] == pytest.approx(field)
    assert metadata['physics.BIRKS_Ab']['value'] == pytest.approx(birks)
    assert metadata['detector.E_FIELD']['nominal'] == 0.500
    assert metadata['detector.E_FIELD']['units'] == 'kV/cm'
    assert nominal['physics']['BIRKS_Ab'] == 0.800
    assert nominal['detector']['e_field'] == 0.500


@pytest.mark.parametrize('parameter, section, key, value', [
    ('detector.E_FIELD', 'detector', 'e_field', 0.510),
    ('detector.TEMPERATURE', 'detector', 'temperature', 88.0),
    ('detector.ELECTRON_LIFETIME', 'detector', 'lifetime', 1500),
    ('detector.LONG_DIFF', 'detector', 'long_diff', 4.4e-6),
    ('detector.TRAN_DIFF', 'detector', 'tran_diff', 1.1e-5),
    ('physics.BIRKS_Ab', 'physics', 'BIRKS_Ab', 0.780),
    ('physics.BIRKS_kb', 'physics', 'BIRKS_kb', 0.050),
    ('physics.W_ION', 'physics', 'W_ION', 24e-6),
    ('physics.BOX_ALPHA', 'physics', 'BOX_ALPHA', 0.95),
    ('physics.BOX_BETA', 'physics', 'BOX_BETA', 0.212),
    ('physics.RECOMBINATION_MODEL', 'physics', 'RECOMBINATION_MODEL', 'box'),
])
def test_explicit_values(nominal, parameter, section, key, value):
    before = deepcopy(nominal)
    effective, metadata = resolve_properties(
        nominal, {'DETECTOR_SYSTEMATICS': {parameter: {'value': value}}})
    assert effective[section][key] == value
    assert metadata[parameter]['requested'] == {'value': value}
    assert nominal == before


@pytest.mark.parametrize('direction, factor', [('up', 1.02), ('down', 0.98)])
def test_registry_driven_shifts_for_all_numeric_inputs(nominal, direction, factor):
    registry = read_property_file(CONFIG_PATH / 'systematics.yaml')
    parameters = {name: definition for name, definition in registry['parameters'].items()
                  if 'choices' not in definition}
    registry['profiles']['test'] = {
        name: {'relative_uncertainty': 0.02} for name in parameters}
    cfg = {'SYSTEMATICS_PROFILE': 'test',
           'DETECTOR_SYSTEMATICS': {name: direction for name in parameters}}
    effective, metadata = resolve_properties(nominal, cfg, registry=registry)
    for parameter, definition in parameters.items():
        section, key = definition['section'], definition['key']
        assert effective[section][key] == pytest.approx(nominal[section][key] * factor)
        assert metadata[parameter]['definition']['relative_uncertainty'] == 0.02


def test_module_selection_precedes_variation_without_accumulation(nominal):
    nominal['detector']['module_to_tpcs'] = {1: [0, 1], 2: [2, 3]}
    nominal['detector'].update(e_field=[0.5, 0.6], lifetime=[2000, 1500],
                               temperature=[87, 88], long_diff=[4e-6],
                               tran_diff=[8e-6, 9e-6])
    cfg = {'SYSTEMATICS_PROFILE': 'ndlar',
           'DETECTOR_SYSTEMATICS': {'detector.E_FIELD': 'up'}}
    first, _ = resolve_properties(nominal, cfg, i_module=1)
    second, metadata = resolve_properties(nominal, cfg, i_module=2)
    assert first['detector']['e_field'] == pytest.approx(0.510)
    assert second['detector']['e_field'] == pytest.approx(0.612)
    assert second['detector']['lifetime'] == 1500
    assert second['detector']['temperature'] == 88
    assert second['detector']['long_diff'] == 4e-6
    assert second['detector']['tran_diff'] == 9e-6
    assert metadata['detector.E_FIELD']['nominal'] == 0.6
    assert resolve_properties(nominal, cfg, i_module=1)[0] == first
    with pytest.warns(UserWarning, match='first value'):
        non_varying, _ = resolve_properties(nominal, cfg)
    assert non_varying['detector']['e_field'] == pytest.approx(0.510)
    assert nominal['detector']['e_field'] == [0.5, 0.6]


@pytest.mark.parametrize('parameter, value', [
    ('detector.E_FIELD', 0), ('detector.TEMPERATURE', -1),
    ('detector.ELECTRON_LIFETIME', 0), ('detector.LONG_DIFF', -1),
    ('detector.TRAN_DIFF', -1), ('physics.BIRKS_Ab', 1.1),
    ('physics.BIRKS_kb', 0), ('physics.W_ION', 0),
    ('physics.BOX_ALPHA', 0), ('physics.BOX_BETA', -1),
    ('physics.RECOMBINATION_MODEL', 'invalid'),
])
def test_invalid_parameter_values(nominal, parameter, value):
    with pytest.raises(ValueError, match=parameter):
        resolve_properties(nominal, {'DETECTOR_SYSTEMATICS': {parameter: {'value': value}}})


@pytest.mark.parametrize('variation', [
    'upp', 'nominal', 0.780, {'value': True}, {'value': float('nan')},
    {'value': float('inf')}, {'value': '0.780'}, {'value': 0.780, 'shift': 'up'},
])
def test_invalid_requests(nominal, variation):
    with pytest.raises(ValueError):
        resolve_properties(nominal, {'DETECTOR_SYSTEMATICS': {'physics.BIRKS_Ab': variation}})


@pytest.mark.parametrize('config', [
    {'DETECTOR_SYSTEMATICS': []},
    {'DETECTOR_SYSTEMATICS': {'detector.V_DRIFT': {'value': 0.16}}},
    {'DETECTOR_SYSTEMATICS': {'detector.DRIFT_MAX_TIME': 'down'}},
    {'DETECTOR_SYSTEMATICS': {'physics.BIRKS_Ab': 'up'}},
    {'SYSTEMATICS_PROFILE': 'missing', 'DETECTOR_SYSTEMATICS': {'physics.BIRKS_Ab': 'up'}},
    {'SYSTEMATICS_PROFILE': 'ndlar', 'DETECTOR_SYSTEMATICS': {'physics.W_ION': 'up'}},
    {'DETECTOR_SYSTEMATICS': {'physics.RECOMBINATION_MODEL': 'up'}},
])
def test_invalid_configuration(nominal, config):
    with pytest.raises(ValueError):
        resolve_properties(nominal, config)


def test_defaults_and_physics_file(tmp_path):
    source = tmp_path / 'physics.yaml'
    source.write_text('BOX_BETA: 0.212\nRECOMBINATION_MODEL: box\n')
    nominal = read_properties({}, source)
    effective, _ = resolve_properties(nominal)
    assert effective['physics']['BOX_BETA'] == 0.212
    assert effective['physics']['RECOMBINATION_MODEL'] == 'box'
    assert effective['physics']['BIRKS_Ab'] == 0.800
    assert effective['detector']['e_field'] == 0.500
    source.write_text('misspelled_coefficient: 0.5\n')
    with pytest.raises(ValueError, match='Unknown physics properties'):
        read_properties({}, source)


def test_loaded_mapping_and_preset_are_not_mutated(nominal):
    copied = read_property_file(nominal)
    copied['physics']['BIRKS_Ab'] = 0.7
    assert nominal['physics']['BIRKS_Ab'] == 0.8
    cfg = get_config('ndlar_birks_custom_demo')
    cfg['DETECTOR_SYSTEMATICS']['physics.BIRKS_Ab']['value'] = 0.7
    assert get_config('ndlar_birks_custom_demo')['DETECTOR_SYSTEMATICS'][
        'physics.BIRKS_Ab']['value'] == 0.780


def test_zero_diffusion_and_bad_module_lists(nominal):
    nominal['detector']['long_diff'] = 0
    assert resolve_properties(nominal)[0]['detector']['long_diff'] == 0
    nominal['detector']['e_field'] = []
    with pytest.raises(ValueError, match='one value or one per module'):
        resolve_properties(nominal)
