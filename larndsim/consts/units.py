"""
HEP coherent system of units

The basic units are :
    millimeter              (millimeter)
    nanosecond              (nanosecond)
    Mega electron Volt      (MeV)
    positron charge         (e)
    degree Kelvin           (kelvin)
    the amount of substance (mole)
    luminous intensity      (candela)
    radian                  (radian)
    steradian               (steradian)

"""

# Length [L]

euro = 1.
millimeter = 1.
millimeter2 = millimeter * millimeter
millimeter3 = millimeter * millimeter2

centimeter = 10. * millimeter
centimeter2 = centimeter * centimeter
centimeter3 = centimeter * centimeter2

decimeter = 100. * millimeter
decimeter2 = decimeter * decimeter
decimeter3 = decimeter * decimeter2
liter = decimeter3
l = liter
ml  = 1e-3 * l
mul = 1e-6 * l
nl  = 1e-9 * l
pl  = 1e-12 * l

meter = 1000. * millimeter
meter2 = meter * meter
meter3 = meter * meter2

kilometer = 1000. * meter
kilometer2 = kilometer * kilometer
kilometer3 = kilometer * kilometer2

micrometer = 1.e-6 * meter
nanometer = 1.e-9 * meter
angstrom = 1.e-10 * meter
fermi = 1.e-15 * meter

nm = nanometer
mum = micrometer

micron = micrometer
micron2 = micrometer * micrometer
micron3 = micron2 * micrometer

barn = 1.e-28 * meter2
millibarn = 1.e-3 * barn
microbarn = 1.e-6 * barn
nanobarn = 1.e-9 * barn
picobarn = 1.e-12 * barn

# symbols
mm = millimeter
mm2 = millimeter2
mm3 = millimeter3

cm = centimeter
cm2 = centimeter2
cm3 = centimeter3

m = meter
m2 = meter2
m3 = meter3

km = kilometer
km2 = kilometer2
km3 = kilometer3

ft = 30.48 * cm

# Angle

radian = 1.
milliradian = 1.e-3 * radian
degree = (3.14159265358979323846/180.0) * radian

steradian = 1.

# symbols
rad = radian
mrad = milliradian
sr = steradian
deg = degree

# Time [T]

nanosecond = 1.
second = 1.e+9 * nanosecond
millisecond = 1.e-3 * second
microsecond = 1.e-6 * second
picosecond = 1.e-12 * second
femtosecond = 1.e-15 * second
year = 3.1536e+7 * second
day = 864e2 * second
minute = 60 * second
hour = 60 * minute

s = second
ms = millisecond
ps = picosecond
fs = femtosecond
mus = microsecond
ns = nanosecond

hertz = 1./second
kilohertz = 1.e+3 * hertz
megahertz = 1.e+6 * hertz
gigahertz = 1.e+6 * hertz

MHZ = megahertz
kHZ = kilohertz
kHz = kHZ
GHZ = gigahertz

# Electric charge [Q]

e = 1. # electron charge
e_SI = -1.60217733e-19 # electron charge in coulomb
coulomb = e/e_SI # coulomb = 6.24150 e+18 * e

# Energy [E]

megaelectronvolt = 1.
electronvolt = 1.e-6 * megaelectronvolt
milielectronvolt = 1.e-3 * electronvolt
kiloelectronvolt = 1.e-3 * megaelectronvolt
gigaelectronvolt = 1.e+3 * megaelectronvolt
teraelectronvolt = 1.e+6 * megaelectronvolt
petaelectronvolt = 1.e+9 * megaelectronvolt

meV = milielectronvolt
eV = electronvolt
keV = kiloelectronvolt
MeV = megaelectronvolt
GeV = gigaelectronvolt
TeV = teraelectronvolt
PeV = petaelectronvolt

eV2 = eV*eV

joule = electronvolt/e_SI # joule = 6.24150 e+12 * MeV
J     = joule
milijoule = 1e-3 * joule
microjoule = 1e-6 * joule
nanojoule = 1e-9 * joule
picojoule = 1e-12 * joule
femtojoule = 1e-15 * joule
mJ  = milijoule
muJ = microjoule
nJ  = nanojoule
pJ  = picojoule
fJ  = femtojoule

# Mass [E][T^2][L^-2]

kilogram = joule * second * second / meter2
gram = 1.e-3 * kilogram
milligram = 1.e-3 * gram
ton = 1.e+3 * kilogram
kiloton = 1.e+3 * ton

# symbols
kg = kilogram
g = gram
mg = milligram

# Power [E][T^-1]

watt = joule/second # watt = 6.24150 e+3 * MeV/ns
W    = watt
milliwatt = 1E-3 * watt
microwatt = 1E-6 * watt
mW = milliwatt
muW = microwatt

# Force [E][L^-1]

newton = joule/meter  # newton = 6.24150 e+9 * MeV/mm

# Pressure [E][L^-3]

hep_pascal = newton / m2 # pascal = 6.24150 e+3 * MeV/mm3
pascal = hep_pascal
Pa = pascal
kPa = 1000 * Pa
MPa = 1e+6 * Pa
GPa = 1e+9 * Pa
bar = 100000 * pascal # bar = 6.24150 e+8 * MeV/mm3
milibar = 1e-3 * bar

atmosphere = 101325 * pascal # atm = 6.32420 e+8 * MeV/mm3

denier = gram / (9000 * meter)

# Electric current [Q][T^-1]

ampere = coulomb/second # ampere = 6.24150 e+9 * e/ns
milliampere = 1.e-3 * ampere
microampere = 1.e-6 * ampere
nanoampere = 1.e-9 * ampere
mA = milliampere
muA = microampere
nA = nanoampere

# Electric potential [E][Q^-1]

megavolt = megaelectronvolt / e
kilovolt = 1.e-3 * megavolt
volt = 1.e-6 * megavolt
millivolt = 1.e-3 * volt

V = volt
mV = millivolt
kV = kilovolt
MV = megavolt

# Electric resistance [E][T][Q^-2]

ohm = volt / ampere # ohm = 1.60217e-16*(MeV/e)/(e/ns)

# Electric capacitance [Q^2][E^-1]

farad = coulomb / volt # farad = 6.24150e+24 * e/Megavolt
millifarad = 1.e-3 * farad
microfarad = 1.e-6 * farad
nanofarad = 1.e-9 * farad
picofarad = 1.e-12 * farad

nF = nanofarad
pF = picofarad

# Magnetic Flux [T][E][Q^-1]

weber = volt * second # weber = 1000*megavolt*ns

# Magnetic Field [T][E][Q^-1][L^-2]

tesla = volt*second / meter2 # tesla = 0.001*megavolt*ns/mm2

gauss = 1.e-4 * tesla
kilogauss = 1.e-1 * tesla

# Inductance [T^2][E][Q^-2]

henry = weber / ampere # henry = 1.60217e-7*MeV*(ns/e)**2

# Temperature

kelvin = 1
K = kelvin

# Amount of substance

mole = 1
mol = mole
milimole    = 1E-3 * mole
micromole   = 1E-6 * mole
nanomole    = 1E-9 * mole
picomole    = 1E-12 * mole

# Activity [T^-1]

becquerel = 1 / second

curie = 3.7e+10 * becquerel

Bq = becquerel
mBq = 1e-3 * becquerel
muBq = 1e-6 * becquerel
kBq =  1e+3 * becquerel
MBq =  1e+6 * becquerel
cks = Bq/keV
U238ppb = Bq / 81
Th232ppb = Bq / 246

# Absorbed dose [L^2][T^-2]

gray = joule / kilogram

# Luminous intensity [I]

candela = 1

# Luminous flux [I]

lumen = candela * steradian

# Illuminance [I][L^-2]

lux = lumen / meter2

# Miscellaneous

perCent = 1e-2
perThousand = 1e-3
perMillion = 1e-6

pes = 1
adc = 1

def celsius(tKelvin):
    return tKelvin - 273.15