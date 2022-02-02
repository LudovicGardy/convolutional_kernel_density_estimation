"""
Creation date: 2019, March 20
Author: L. Gardy
E-mail: ludovic.gardy@cnrs.fr
Encoding: UTF-8

Related publication
--------------------
Title: Automatic Detection of Epileptic Spikes in Intracerebral EEG with Convolutional Kernel Density Estimation.
Authors: L. Gardy, E.J. Barbeau, C. Hurter.
Conference journal: Proceedings of the 15th International Joint Conference on Computer Vision,
Imaging and Computer Graphics Theory and Applications - Volume 2:
HUCAPP, ISBN 978-989-758-402-2, ISSN 2184-4321, pages 101-109.
DOI: https://doi.org/10.5220/0008877601010109
"""

import numpy as np
import os
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian2DKernel
from scipy import ndimage
import scipy.stats as st
import json

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
    plt.show()

    return(x, fx)

def create_gaussian_kernel(kernlen=21, nsig=3):
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

    return(kernel)

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

def load_timeseries(timeseries_folderpath, timeserie_filename, sep = ";"):
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

    ### Load & read data
    with open( os.path.join(timeseries_folderpath, timeserie_filename) , 'r+') as f:
        row = f.readlines()

    ### Raw signal
    timeseries = []
    [timeseries.append(float(i)) for i in row[0].split(sep)]
    timeseries = np.array(timeseries).astype('float64')

    plt.plot(timeseries)
    plt.show()

    return(timeseries)

def from_1D_to_2D(timeseries, bandwidth = 3):
    '''
    Returns an image (2D numpy array).

    Parameters
    ----------
    timeseries: list or numpy 1D array
        This signal will be transformed into an image

    bandwidth: int
        Size of the filter
    '''

    image_2D = []
    miny = (min(timeseries) * 1)
    maxy = (max(timeseries) * 1)
    X_range = np.linspace( miny, maxy, 150)[:, np.newaxis]

    for _val in timeseries:
        _val_reshaped = np.array(_val[np.newaxis]).reshape(-1,1)
        kde = KernelDensity(kernel="tophat", bandwidth=bandwidth).fit(_val_reshaped)
        log_dens = np.exp(kde.score_samples(X_range))
        image_2D.append(log_dens)

    image_2D = np.rot90(np.array(image_2D))

    return(image_2D)

def convolve_2D_image(image_2D, convolution = "gaussian cutstom"):
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
        image_2D_convolved = ndimage.filters.convolve(image_2D, k, mode='nearest')
    elif convolution == "custom":
        k = create_custom_kernel()
        image_2D_convolved = ndimage.filters.convolve(image_2D, k, mode='nearest')
    else:
        image_2D_convolved = convolve(image_2D, Gaussian2DKernel(x_stddev=2))

    return(image_2D_convolved)

def pot_result(signal, image_2D, image_2D_convolved, fig_name):
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

    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)

    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)

    ax[2].get_xaxis().set_visible(False)
    ax[2].get_yaxis().set_visible(False)
    ax[2].set_xticks([])

    plt.show()

if __name__ == "__main__":

    ### Init parameters (root is the path to the folder you have downloaded)
    root = r"F:\GardyL\Python\CKDE"
    event_num = 5

    ### Get a timeseries filepath (look in the folder you have downloaded)
    timeseries_folderpath =  os.path.join(root, "test_events_database\events_signal_data")
    timeserie_filename = f"event_{event_num}.txt"

    ### Load a timeseries from the sample data provided with this program (1D)
    signal = load_timeseries(timeseries_folderpath, timeserie_filename) # or,
    #signal = random_signal_simulation()

    ### Get the timeseries info
    json_dict = json.load(open(os.path.join(root,"test_events_database\events_info.json")))
    sfreq = json_dict["events_info"][event_num]["sampling_frequency"]

    ### Convert it to a 2D signal
    image_2D = from_1D_to_2D(signal, bandwidth = 1)

    ### Convolve the 2D signal
    image_2D_convolved = convolve_2D_image(image_2D, convolution = "gaussian custom")

    ### Plot result
    fig_name = "Epileptic spike (signal duration: 400 ms) \n\n[1] raw [2] imaged [3] convoluted"
    pot_result(signal, image_2D, image_2D_convolved, fig_name)
