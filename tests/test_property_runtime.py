"""Exercise real loaders/kernels in an isolated Numba CUDA simulator process.

NumPy supplies only the host-side response-file operations normally provided
by CuPy. Packet-class placeholders avoid needing LArPix for these physics tests.
This does not validate real CUDA compilation, device memory or packet output.
"""

import os
from pathlib import Path
import subprocess
import sys


def test_property_installation_and_kernel_refresh():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ, NUMBA_ENABLE_CUDASIM='1', PYTHONDONTWRITEBYTECODE='1'),
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _check_runtime():
    import math
    import tempfile
    import types

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.modules['cupy'] = np
    packets = types.ModuleType('larpix.packet')
    packets.Packet_v2 = type('Packet_v2', (), {})
    packets.Packet_v3 = type('Packet_v3', (), {})
    sys.modules['larpix'] = types.ModuleType('larpix')
    sys.modules['larpix.packet'] = packets

    from larndsim import consts, drifting, quenching, pixels_from_track
    from larndsim.config import read_properties, resolve_properties
    from larndsim.config.runtime import prepare_simulation
    from larndsim.consts import detector, physics, light

    layout = dict(
        pixel_pitch=4.434,
        chip_channel_to_position={1000: [0, 0], 1001: [1, 0],
                                  1002: [0, 1], 1003: [1, 1]},
        tile_chip_to_io={}, tile_indeces={0: [0, 0, 0], 1: [1, 0, 0]},
        tile_orientations={0: [1, 0, 0], 1: [-1, 0, 0]},
        tile_positions={0: [0., 0., 0.], 1: [1000., 0., 0.]}, drift_length=500,
    )
    detector_inputs = dict(
        tpc_offsets=[[0, 0, 0]], tile_map=[[[0]]],
        module_to_io_groups={1: [1]}, module_to_tpcs={1: [0, 1]},
    )
    nominal = read_properties(detector_inputs)
    cases = [
        {},
        {'physics.BIRKS_Ab': 0.780},
        {'detector.E_FIELD': 0.510},
        {'detector.TEMPERATURE': 88.0},
        {'detector.ELECTRON_LIFETIME': 1500},
        {'detector.LONG_DIFF': 4.4e-6},
        {'detector.TRAN_DIFF': 1.1e-5},
        {'physics.BIRKS_kb': 0.054},
        {'physics.W_ION': 24e-6},
        {'physics.RECOMBINATION_MODEL': 'box'},
        {'physics.RECOMBINATION_MODEL': 'box', 'physics.BOX_ALPHA': 0.95},
        {'physics.RECOMBINATION_MODEL': 'box', 'physics.BOX_BETA': 0.212},
        {'physics.BIRKS_Ab': 0.780, 'detector.E_FIELD': 0.510},
        {},  # A nominal run after variations must not inherit their state.
    ]
    fields = ['dEdx', 'dE', 'n_electrons', 'n_photons', 'x', 'y', 'z',
              'z_start', 'z_end', 't', 't0', 't_start', 't_end', 'long_diff', 'tran_diff']
    dtype = [(name, 'f8') for name in fields] + [('pdg_id', 'i4'), ('pixel_plane', 'u4')]
    with tempfile.TemporaryDirectory() as directory:
        response_path = Path(directory) / 'response.npz'
        np.savez(response_path, time_tick=0.1, bin_size=0.1,
                 drift_length=50., response=np.ones((2, 2, 2)))
        previous = None
        reference = None
        for values in cases:
            config = {'DETECTOR_SYSTEMATICS': {
                name: {'value': value} for name, value in values.items()}}
            effective, _ = resolve_properties(nominal, config)
            derived = consts.set_properties(effective, layout, response_path)
            detector.load_response(response_path)
            old_kernel = quenching.quench
            prepare_simulation()
            assert quenching.quench is not old_kernel
            state = (quenching.quench, drifting.drift, pixels_from_track.pixel2id)
            if previous is not None:
                assert all(new is not old for new, old in zip(state, previous))
            previous = state
            assert pixels_from_track.pixel2id(1, 1, 0) == 3
            p, d = effective['physics'], effective['detector']
            velocity = d['e_field'] * detector.electron_mobility(
                d['e_field'], d['temperature'])
            assert math.isclose(detector.V_DRIFT, velocity)
            assert math.isclose(derived['detector']['DRIFT_MAX_TIME'], 50 / velocity)
            tracks = np.zeros(1, dtype=dtype)
            tracks['dEdx'], tracks['dE'], tracks['pdg_id'] = 2., 1., 13
            tracks['z'], tracks['z_start'], tracks['z_end'] = 25., 24., 26.
            if p['RECOMBINATION_MODEL'] == 'birks':
                recombination = p['BIRKS_Ab'] / (
                    1 + p['BIRKS_kb'] * 2 / (d['e_field'] * detector.LAR_DENSITY))
            else:
                xi = p['BOX_BETA'] * 2 / (d['e_field'] * detector.LAR_DENSITY)
                recombination = max(0, math.log(p['BOX_ALPHA'] + xi) / xi)
            electrons = recombination / p['W_ION']
            quenching.quench[1, 1](tracks, physics.recombination_mode())
            np.testing.assert_allclose(tracks['n_electrons'], electrons)
            np.testing.assert_allclose(
                tracks['n_photons'], (1 / light.W_PH - electrons) * light.SCINT_PRESCALE)
            drifting.drift[1, 1](tracks)
            drift_time = 25 / velocity
            np.testing.assert_allclose(tracks['t'], drift_time)
            np.testing.assert_allclose(tracks['t_start'], 24 / velocity)
            np.testing.assert_allclose(tracks['t_end'], 26 / velocity)
            np.testing.assert_allclose(
                tracks['n_electrons'], electrons * math.exp(-drift_time / d['lifetime']))
            np.testing.assert_allclose(tracks['long_diff'], math.sqrt(2 * d['long_diff'] * drift_time))
            np.testing.assert_allclose(tracks['tran_diff'], math.sqrt(2 * d['tran_diff'] * drift_time))
            if not values:
                if reference is None:
                    reference = tracks.copy()
                else:
                    np.testing.assert_array_equal(tracks, reference)

        # Select a module, resolve, install, and refresh twice in the same process.
        nominal['detector'].update(
            module_to_tpcs={1: [0, 1], 2: [2, 3]},
            module_to_io_groups={1: [1], 2: [2]},
            tpc_offsets=[[0, 0, 0], [100, 0, 0]],
            e_field=[0.5, 0.6], lifetime=[2200, 1500])
        for module_id, field in [(1, 0.510), (2, 0.612), (1, 0.510)]:
            effective, _ = resolve_properties(
                nominal, {'SYSTEMATICS_PROFILE': 'ndlar',
                          'DETECTOR_SYSTEMATICS': {'detector.E_FIELD': 'up'}},
                i_module=module_id)
            consts.set_properties(effective, layout, response_path, i_module=module_id)
            prepare_simulation()
            assert math.isclose(detector.E_FIELD, field)
            tracks = np.zeros(1, dtype=dtype)
            tracks['dEdx'], tracks['dE'], tracks['pdg_id'] = 2., 1., 13
            quenching.quench[1, 1](tracks, physics.BIRKS)
            expected = 0.8 / (1 + 0.0486 * 2 / (field * 1.38)) / 23.6e-6
            np.testing.assert_allclose(tracks['n_electrons'], expected)

        # A real CPU-compiled helper must also discard captured geometry values.
        layout['chip_channel_to_position'][1004] = [2, 0]
        consts.set_properties(effective, layout, response_path, i_module=1)
        prepare_simulation()
        assert pixels_from_track.pixel2id(1, 1, 0) == 4


if __name__ == '__main__':
    _check_runtime()
