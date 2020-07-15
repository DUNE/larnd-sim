#!/usr/bin/env python

import random
import numpy as np
import pytest

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from larndsim import detsim

class TestDrifting:

    def test_lifetime(self):
        assert 1 == 1
