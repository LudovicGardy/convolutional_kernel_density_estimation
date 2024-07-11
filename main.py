'''
Creation date: 2018, August
Author: L. Gardy
E-mail: ludovic.gardy@cnrs.fr
Encoding: UTF-8

Related publication
--------------------
Title: Automatic detection of epileptic spikes in intracerebral eeg with convolutional kernel density estimation.
Authors: Gardy L., Barbeau E.J., Hurter, C.
4th International Conference on Human Computer Interaction Theory and Applications, pages 101–109. 
SCITEPRESS-Science and Technology Publications
DOI: https://doi.org/10.5220/0008877601010109
'''

from astropy.convolution import convolve, Gaussian2DKernel
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy.stats as st
import numpy as np
import json
import os

def random_signal_simulation(x = np.linspace(1, 10)):
    """
    Returns a random timeseries.

    Parameters
    ----------
    x: numpy 1D array
        Timestamps in the signal
        Will also be used as input to define y values
    """

    fx = np.sin(x) + np.random.normal(scale=0.1, size=len(x))

    plt.plot(x, fx)
    plt.tight_layout()
    plt.show()

    return x, fx

def create_gaussian_kernel(kernlen:int=21, nsig:int=3):
    """
    Returns a gaussian kernel (2D array).

    Parameters
    ----------
    kernlen: int
        Used for kernel configuration

    nsig: int
        Used for kernel configuration
    """

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    
    return kernel

def create_custom_kernel():
    """
    Returns a custom kernel (2D array).

    Parameters
    ----------
    kernlen: int
        Used for kernel configuration

    nsig: int
        Used for kernel configuration
    """

    kernel = np.array([[0, 1, 0],
                       [0, -4, 0],
                       [0, 1, 0]])       
    
    return(kernel)

def load_timeseries(timeseries_folderpath:str, timeserie_filename:str, sep:str= ";"):
    '''
    Returns a timeseries from the text file where it is stored.

    Parameters
    ----------
    timeseries_folderpath: str
        Folder name where the timeseries is located

    timeserie_filename: str
        File name and its extension (ex: mysignal_006.txt)

    sep: str
        Semilicon by default
    '''

    file_path = os.path.join(timeseries_folderpath, timeserie_filename)
    with open(file_path, 'r') as f:
        row = f.readline().strip()
    timeseries = np.array(row.split(sep), dtype=np.float64)

    plt.plot(timeseries)
    plt.tight_layout()
    plt.show()

    return timeseries

def from_1D_to_2D(timeseries:np.array, bandwidth:int, resolution:int):
    '''
    Returns an image (2D numpy array).

    Parameters
    ----------
    timeseries: list or numpy 1D array
        This signal will be transformed into an image

    bandwidth: int
        Size of the filter
    '''

    miny, maxy = (min(timeseries) * 1), (max(timeseries) * 1)
    X_range = np.linspace(miny, maxy, resolution)[:, np.newaxis]

    image_2D = []
    for _val in timeseries:
        _val_reshaped = np.array(_val[np.newaxis]).reshape(-1,1)
        kde = KernelDensity(kernel="tophat", bandwidth=bandwidth).fit(_val_reshaped)
        log_dens = np.exp(kde.score_samples(X_range))
        image_2D.append(log_dens)

    image_2D = np.rot90(np.array(image_2D))

    return image_2D

def convolve_2D_image(image_2D:np.array, convolution:str="gaussian custom"):
    '''
    Returns a convolved image (2D numpy array). 
    Kernels that will be used are either predefined or 
    defined using the function defined above.

    Parameters
    ----------
    image_2D: numpy 2D array
        This image will be convolved

    convolution: str
        Type of convolution user would like to apply between:
            - gaussian astropy
            - gaussian custom
            - custom
    '''

    if convolution == "gaussian astropy":
        image_2D_convolved = convolve(image_2D, Gaussian2DKernel(x_stddev=2))
    elif convolution == "gaussian custom":
        k = create_gaussian_kernel()
        image_2D_convolved = ndimage.convolve(image_2D, k, mode='nearest')         
    elif convolution == "custom":
        k = create_custom_kernel()
        image_2D_convolved = ndimage.convolve(image_2D, k, mode='nearest')    
    else:
        image_2D_convolved = convolve(image_2D, Gaussian2DKernel(x_stddev=2))

    return(image_2D_convolved)

def plot_result(signal:np.array, image_2D:np.array, image_2D_convolved:np.array, fig_name:str):
    '''
    Display a 3-rows figure.

    Parameters
    ----------
    signal: list or numpy 1D array
        The raw signal

    image_2D: numpy 2D array
        The imaged 1D signal

    image_2D_convolved: numpy 2D array
        The convolved imaged 1D signal
    '''

    fig, ax = plt.subplots(nrows = 3)
    fig.suptitle(fig_name)
    ax[0].plot(signal)
    ax[1].imshow(image_2D, aspect = "auto")
    ax[2].imshow(image_2D_convolved, aspect = "auto")

    for axis in ax:
        axis.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    ### Init parameters (root is the path to the folder you have downloaded)
    exemple_event_to_plot = 5

    ### Get a timeseries filepath (look in the folder you have downloaded)
    timeseries_folderpath =  r"test_events_database/events_signal_data"
    timeserie_filename = f"event_{exemple_event_to_plot}.txt"

    ### Load a timeseries from the sample data provided with this program (1D)
    signal = load_timeseries(timeseries_folderpath, timeserie_filename) # or,
    #signal = random_signal_simulation()

    ### Get the timeseries info
    json_dict = json.load(open(r"test_events_database/events_info.json"))
    sfreq = json_dict["events_info"][exemple_event_to_plot]["sampling_frequency"]

    ### Convert it to a 2D signal
    image_2D = from_1D_to_2D(signal, bandwidth=1, resolution=150)

    ### Convolve the 2D signal
    image_2D_convolved = convolve_2D_image(image_2D, convolution = "gaussian custom")

    ### Plot result
    fig_name = "Epileptic spike (signal duration: 400 ms) \n\n[1] raw [2] imaged [3] convoluted"
    plot_result(signal, image_2D, image_2D_convolved, fig_name)
