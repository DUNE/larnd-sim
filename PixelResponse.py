import numpy as np
import h5py
import pandas as pd
import math
import consts
import skimage.draw
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt

#TODO : Define TPC borders in a better place with global access.
binx = np.arange(consts.tpcBorders[0][0], consts.tpcBorders[0][1], consts.pixel_x_size)
biny = np.arange(consts.tpcBorders[1][0], consts.tpcBorders[1][1], consts.pixel_y_size)
bint = np.arange(consts.timeInterval[0],  consts.timeInterval[1],  consts.time_sampling)

class Pixel:
    def __init__(self, id, charge, current):
        self.id = id
        self.charge = charge
        self.current = current


def currentResponse(t, A=1, B=5, t0=0):
    '''Simulates the current response of a pixel over the give time range t'''
    distance = 1
    result = np.heaviside(-t + t0, 0.5) * A * np.exp((1 + distance) * (t - t0) / B)
    result = np.nan_to_num(result)

    return result


def get_bin(track, pos):
    ''' pos: takes 'start' or 'end'
    For a given track position, it returns the bin number of the TPC grid 
    where it falls'''
    return (np.digitize(track[f'x_{pos}'], binx),
           np.digitize(track[f'y_{pos}'], biny),
           np.digitize(track[f't_{pos}'], bint))


def get_slice_range(line, nPad = 4):
    '''
    Line contains all the pixels which are hit by the track segment.
    For a given value of nPads to be added as neighbor pixels 
    that will be impacted by diffusion of charge, it returns the
    relevant ranges of the TPC grid.
    '''
    x_range = range(min(line[0])-nPad, max(line[0])+nPad+1)
    y_range = range(min(line[1])-nPad, max(line[1])+nPad+1)
    t_range = range(min(line[2])-nPad, max(line[2])+nPad+1)
    return x_range, y_range, t_range



def get_data_hits(track):
    '''
    For each track, return a list of Pixel objects (id, charge, current reponse).
    '''


    #Gets all the pixels impacted by the track based on start and end position
    trk_line = skimage.draw.line_nd(get_bin(track,'start'),
                                    get_bin(track,'end'),
                                    endpoint=True)
    

    #Creates the TPC grid and fills the impacted pixels by proportional number of electrons
    grid = np.zeros((len(binx), len(biny), len(bint)))
    grid[trk_line] = track['NElectrons']/len(trk_line[0])


    #Slice the grid into a smaller array that would be impacted by diffusion
    x_range, y_range, t_range = get_slice_range(trk_line)   
    slice = grid[np.ix_(x_range, y_range, t_range)]
    
    diffusedSlice = scipy.ndimage.gaussian_filter(slice,
                                          sigma=(track['tranDiff']*1000,
                                                track['tranDiff']*1000,
                                                track['longDiff']*1000))
    

    #Get the list of pixels (id wrt to whole TPC grid) which contain diffused charge  
    grid[np.ix_(x_range, y_range, t_range)] = diffusedSlice
    activePixels = np.nonzero(grid)
    activePixID  = set([(x,y) for x,y in zip(activePixels[0], activePixels[1])])
    activePixID  = list(sorted(activePixID))  #Not required but I prefer sorted lists

    #For each pixel with charge calculate current response and store as pixel objects
    DataHits = []
    for pix in activePixID:
        charge  = grid[pix[0]][pix[1]]
        current   = currentResponse(bint, t0=150)
        response  = scipy.signal.fftconvolve(current, charge , mode='same')
        DataHits.append(Pixel(pix, charge, response))
        
    return DataHits


def main():

    inFile  = h5py.File('TrackSegmentDataSet.h5', 'r')

    tracks                  = pd.DataFrame()

    #TODO: there exists a way to read h5 into dataframe in a single line. Can't make it work atm.
    tracks['run']           = np.array(inFile['RunID'])
    tracks['subrun']        = np.array(inFile['SubrunID'])
    tracks['spill']         = np.array(inFile['SpillID'])
    tracks['interactionID'] = np.array(inFile['InteractionID'])
    tracks['trackID']       = np.array(inFile['TrackID'])
    tracks['pdg']           = np.array(inFile['PDG'])
    tracks['x_start']       = np.array(inFile['StartXPos'])
    tracks['x_end']         = np.array(inFile['EndXPos'])
    tracks['y_start']       = np.array(inFile['StartYPos'])
    tracks['y_end']         = np.array(inFile['EndYPos'])
    tracks['z_start']       = np.array(inFile['StartZPos'])
    tracks['z_end']         = np.array(inFile['EndZPos'])
    tracks['NElectrons']    = np.array(inFile['IonizationElectrons'])
    tracks['longDiff']      = np.array(inFile['LongitudnalDiffusion'])
    tracks['tranDiff']      = np.array(inFile['TransverseDiffusion'])
    tracks['t_start']       = 100
    tracks['t_end']         = 100

    track = tracks.iloc[0]
    dataHits = get_data_hits(track)

    for hit in dataHits:
        print(hit.id)

    '''
    #To make plots: 
    #hit = dataHits[0]
    for hit in dataHits:
        fig, ax = plt.subplots(1, 1)
        ax.plot(bint, hit.charge,label='Charge')
        ax.plot(bint, hit.current,label='Current')
        ax.set_xlim(50, 150)
        ax.set_title("Pixel (%i,%i)" % (hit.id[0], hit.id[1]))
        ax.set_xlabel("Time [$\mu$s]")
        ax.legend()
    '''

if __name__ == "__main__":
    main()
