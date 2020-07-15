'''
Detector constants
'''

lArDensity = 1.38 #g/cm^3
eField      = 0.50 #V/sm

'''
Unit Conversions
'''

GeVToMeV       = 1e3 #MeV
MeVToElectrons = 4.237e+04
msTous         = 10e3 # us   

'''
PHYSICAL_PARAMS
'''
MeVToElectrons = 4.237e+04
alpha          = 0.847
beta           = 0.2061

'''
TPC_PARAMS
'''
vdrift =  0.153812 # u.cm / u.us,
lifetime =  10e3 # u.us,
tpcBorders = ((-150, 150), (-150, 150), (-150, 150)) # u.cm,
tpcZStart = -150
timeInterval =  (0, 3000)
longDiff =  6.2e-6 # u.cm * u.cm / u.us,
tranDiff =  16.3e-6 # u.cm

