import pytest
import subprocess
import os
import h5py

# FIXME: use a dedicated test file
TEST_EDEP_FILE_URL = ('https://portal.nersc.gov/project/dune/data/Module0/'
                      'simulation/stopping_muons/stopping_muons.edep.h5')

TEST_PIXEL_LAYOUT_PATH = 'larndsim/pixel_layouts/multi_tile_layout-2.2.16.yaml'
TEST_DET_PROP_PATH = 'larndsim/detector_properties/module0.yaml'
TEST_N_TRACKS = 1000


@pytest.fixture
def test_edep_file(tmp_path):
    # download test example
    dest = os.path.join(tmp_path, 'test-edep.h5')
    subprocess.run(['curl', '-f', '-o', dest, TEST_EDEP_FILE_URL], check=True)
    return dest


@pytest.fixture
def cli_sim_outfile(tmp_path, test_edep_file):
    outfile = os.path.join(tmp_path, 'test-out.h5')
    subprocess.run(['python', 'cli/simulate_pixels.py', test_edep_file,
                    TEST_PIXEL_LAYOUT_PATH, TEST_DET_PROP_PATH,
                    f'output_filename={outfile}', f'n_tracks={TEST_N_TRACKS}'],
                   check=True)
    return outfile


def test_cli_simulate_pixels(cli_sim_outfile):
    of = h5py.File(cli_sim_outfile, 'r')
    for dset_name in ('packets', 'tracks', 'mc_packets_assn'):
        assert dset_name in of
        assert len(of[dset_name])
        # FIXME: use actual values here
        # dset_shapes = {'packets': (1000,), 'tracks': (1000,),
        #                'mc_packets_assn': (1000, 10)}
        # assert of[dset_name].shape == dset_shapes[dset_name]
