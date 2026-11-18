"""
Set physics constants
"""
from ..config.properties import read_defaults

_DEFAULTS = read_defaults()['physics']

## Physical params
#: Recombination :math:`\alpha` constant for the Box model
BOX_ALPHA = _DEFAULTS['BOX_ALPHA']
#: Recombination :math:`\beta` value for the Box model in :math:`(kV/cm)(g/cm^2)/MeV`
BOX_BETA = _DEFAULTS['BOX_BETA'] # R. Acciarri et al JINST 8 (2013) P08005
#: Recombination :math:`A_b` value for the Birks Model
BIRKS_Ab = _DEFAULTS['BIRKS_Ab']
#: Recombination :math:`k_b` value for the Birks Model in :math:`(kV/cm)(g/cm^2)/MeV`
BIRKS_kb = _DEFAULTS['BIRKS_kb'] # Amoruso, et al NIM A 523 (2004) 275
#: Electron charge in Coulomb
E_CHARGE = 1.602e-19
#: Average energy expended per ion pair in LAr in :math:`MeV` from Phys. Rev. A 10, 1452
W_ION = _DEFAULTS['W_ION']
RECOMBINATION_MODEL = _DEFAULTS['RECOMBINATION_MODEL']

## Quenching parameters
BOX = 1
BIRKS = 2


def set_physics_properties(properties):
    """Install complete, validated physics inputs prepared by the loader."""
    globals().update({name: properties[name] for name in _DEFAULTS})


def recombination_mode():
    """Translate the configured model name to the existing kernel identifier."""
    return {'birks': BIRKS, 'box': BOX}[RECOMBINATION_MODEL]
