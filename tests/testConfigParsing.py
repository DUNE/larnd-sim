#!/usr/bin/env python

import pytest
import warnings
from larndsim import config, consts

class TestConfigs:
    """
    Config module testing
    """

    def test_Configs(self):
        config.test_configs()
    
    def test_getConfig(self):
        for config_name in config.list_config_keys():
            config.get_config(config_name)

    def test_loadDetectorProperties(self):
        for config_name in config.list_config_keys():
            try:
                consts.detector.load_detector_properties(config_name,
                                                         use_backend = 'numpy')
            except FileNotFoundError:
                warnings.warn("Unable to load detector properties from config key: "\
                              + config_name +\
                              "\nPerhaps you're not running on NERSC?")
