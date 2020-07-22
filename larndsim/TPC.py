import numpy as np

"""
TPC_PARAMS
"""
vdrift = 0.153812 # cm / us,
lifetime = 10e3 # us,
tpcBorders = ((-150, 150), (-150, 150), (-150, 150)) # cm,
tpcZStart = -150 # cm
timeInterval = (0, 3000) # us
longDiff = 6.2e-6 # cm * cm / us,
tranDiff = 16.3e-6 # cm


n_pixels = 1000

x_start = tpcBorders[0][0]
x_end = tpcBorders[0][1]
x_length = x_end - x_start

y_start = tpcBorders[1][0]
y_end = tpcBorders[1][1]
y_length = y_end - y_start

t_start = timeInterval[0]
t_end = timeInterval[1]
t_length = t_end - t_start

x_sampling = x_length/n_pixels/4
y_sampling = y_length/n_pixels/4
t_sampling = 1
z_sampling = t_sampling * vdrift

anode_x = np.linspace(x_start, x_end, int(x_length/x_sampling))
anode_y = np.linspace(y_start, y_end, int(y_length/y_sampling))
anode_t = np.linspace(t_start, t_end, int(t_length/t_sampling))

x_pixel_size = x_length / n_pixels
y_pixel_size = y_length / n_pixels

x_pixel_range = np.linspace(0, x_pixel_size, int(x_pixel_size/x_sampling))
y_pixel_range = np.linspace(0, y_pixel_size, int(y_pixel_size/y_sampling))
