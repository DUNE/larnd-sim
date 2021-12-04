"""
Set physics constants
"""

## Physical params
#: Recombination :math:`\alpha` constant for the Box model
BOX_ALPHA = 0.93
#: Recombination :math:`\beta` value for the Box model in :math:`(kV/cm)(g/cm^2)/MeV`
BOX_BETA = 0.207 #0.3 (MeV/cm)^-1 * 1.383 (g/cm^3)* 0.5 (kV/cm), R. Acciarri et al JINST 8 (2013) P08005
#: Recombination :math:`A_b` value for the Birks Model
BIRKS_Ab = 0.800
#: Recombination :math:`k_b` value for the Birks Model in :math:`(kV/cm)(g/cm^2)/MeV`
BIRKS_kb = 0.0486 # g/cm2/MeV Amoruso, et al NIM A 523 (2004) 275
#: Electron charge in Coulomb
E_CHARGE = 1.602e-19
#: Average energy expended per ion pair in LAr in :math:`MeV` from Phys. Rev. A 10, 1452
W_ION = 23.6e-6

## Quenching parameters
BOX = 1
BIRKS = 2
