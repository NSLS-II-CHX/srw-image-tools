import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

import h5py
from pyCHX.chx_xpcs_xsvs_jupyter_V1 import *

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions


def save_hdf5(data, filename='data.h5', dataset='dataset'):
    ''' Access BlueSky HDF5 binary data from CHX measurement.

    :param data: HDF5 binary data from CHX measurement.
    :param filename='data.h5': HDF5 filename.
    :param dataset='dataset': Creates dataset type. Default is dataset.

    :return: string status of dataset creation
    '''
    h5f = h5py.File(filename, 'w')
    r = h5f.create_dataset(dataset, data=data)
    status = '{} created: {}'.format(r, os.path.abspath(filename))
    h5f.close()
    return status


def plot_profile_horiz(data, uid, y_crd=1200, dpi=80, clim=(0, 200),
                       cmap='afmhot', line_color='deepskyblue',
                       linestyles=None):
    ''' Show plot of intensity versus horizontal position.

    :param data: HDF5 binary data from CHX measurement.
    :param uid: unique ID automatically assigned to a CHX measurement.
    :param y_crd=1200: add a horizontal line across the axis at a given
                       location on the image.
    :param dpi=80: dpi (dots per inch) for output image.
    :param clim=(0, 200): sets the color limits of the current image.
    :param cmap='afmhot': color map (https://matplotlib.org/examples/color/colormaps_reference.html)
    :param line_color='red': color of line that will show the cut location.
    '''
    print("\n\nHorizontal cut at row " + str(y_crd))

    # What size does the figure need to be in inches to fit the image?
    height, width = data.shape
    figsize = width / float(dpi), height / float(dpi)

    # Plot image with cut line
    fig = plt.figure(figsize=figsize)
    for d in data:
        plt.imshow(d, clim=clim, cmap=cmap)
        plt.axhline(y_crd, color=line_color)
        plt.show()
    fig.savefig(str(uid) + "_profile_horiz_image.tif", bbox_inches='tight')

    # Plot intensity plot
    fig2 = plt.figure(figsize=(15.0, 6.0))
    warnings.filterwarnings("ignore")
    for i, d in enumerate(data):
        linestyle = '-' if not linestyles else linestyles[i]
        plt.plot(np.log10(d[y_crd, :]), label=uid[i],
                 linestyle=linestyle, linewidth=1)

    plt.xlabel('Transverse Position (pixel)')
    plt.ylabel('Intensity')
    plt.grid()
    plt.legend()
    plt.show()
    fig2.savefig(str(uid) + "_profile_horiz_intensity.tif",
                 bbox_inches='tight')


def plot_profile_vert(data, uid, x_crd=1100, dpi=80, clim=(0, 200),
                      cmap='afmhot', line_color='deepskyblue',
                      linestyles=None):
    ''' Show plot of intensity versus vertical position.

    :param data: HDF5 binary data from CHX measurement.
    :param uid: unique ID automatically assigned to a CHX measurement.
    :param x_crd=1100: add a vertical line across the axis at a given
                       location on the image.
    :param dpi=80: dpi (dots per inch) for output image.
    :param clim=(0, 200): sets the color limits of the current image.
    :param cmap='afmhot': color map (https://matplotlib.org/examples/color/colormaps_reference.html)
    :param line_color='red': color of line that will show the cut location.
    :param linestyles=None: custom linestyles
    '''
    print("\n\nVertical cut at column " + str(x_crd))

    # What size does the figure need to be in inches to fit the image?
    height, width = data.shape
    figsize = width / float(dpi), height / float(dpi)

    # Plot image with cut line
    fig = plt.figure(figsize=figsize)
    for d in data:
        plt.imshow(d, clim=clim, cmap=cmap)
        plt.axvline(x_crd, color=line_color)
        plt.show()
    fig.savefig(str(uid) + "_profile_vert_image.tif", bbox_inches='tight')

    # Plot intensity plot
    fig2 = plt.figure(figsize=(15.0, 6.0))
    warnings.filterwarnings("ignore")
    for i, d in enumerate(data):
        linestyle = '-' if not linestyles else linestyles[i]
        plt.plot(np.log10(d[:, x_crd]), label=uid[i],
                 linestyle=linestyle, linewidth=1)

    plt.xlabel('Transverse Position (pixel)')
    plt.ylabel('Intensity')
    plt.grid()
    plt.legend()
    plt.show()
    fig2.savefig(str(uid) + "_profile_vert_intensity.tif", bbox_inches='tight')


def display_image_in_actual_size(img, uid, dpi=80, eiger_size_per_pixel=0.075,
                                 clim=(0, 100), cmap='gist_stern'):
    ''' Display CHX Eiger image in full size and save the image as a TIFF with dual pixel and mm axis.

    :param im: eiger detector image.
    :param uid: unique ID automatically assigned to a CHX measurement.
    :param dpi=80: dpi (dots per inch) for output image.
    :param eiger_size_per_pixel=0.075: eiger camera has 75 um per pixel.
    :param cmap='gist_stern': color map (https://matplotlib.org/examples/color/colormaps_reference.html)
    :param clim: sets the color limits of the current image.
    '''
    img_data = img
    height, width = img_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes
    # up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Title
    plt.title("UID: " + str(uid), fontsize=30)

    # Set up pixel axis
    ax.axis('on')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Pixels', fontsize=20)
    plt.ylabel('Pixels', fontsize=20)

    # Display the image.
    ax.imshow(img_data, cmap=cmap, clim=clim)

    # Set up microm axis
    ax1 = ax.twiny()  # Create a twin Axes sharing the yaxis
    ax2 = ax.twinx()  # Create a twin Axes sharing the xaxis

    # Decide the ticklabel position in the new axis,
    # then convert them to the position in the old axis
    newlabelX = range(0, int(eiger_size_per_pixel*width), 5)  # labels of the ticklabels: the position in the new x-axis
    newlabelY = range(0, int(eiger_size_per_pixel*height), 5)  # labels of the ticklabels: the position in the new x-axis
    pixel2micronX = lambda width: width*eiger_size_per_pixel  # convert function X: from pixels to microns
    pixel2micronY = lambda height: height*eiger_size_per_pixel  # convert function Y: from pixels to microns
    newposX = [pixel2micronX(width) for width in newlabelX]  # position of the xticklabels in the old x-axis
    newposY = [pixel2micronY(height) for height in newlabelY]  # position of the yticklabels in the old y-axis

    ax1.set_xticks(newposX)
    ax1.set_xticklabels(newlabelY)
    ax1.tick_params(axis="x", labelsize=20)

    ax2.set_yticks(newposY)
    ax2.set_yticklabels(newlabelY)
    ax2.tick_params(axis="y", labelsize=20)

    ax1.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
    ax2.yaxis.set_ticks_position('left')  # set the position of the second y-axis to left

    ax1.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
    ax2.yaxis.set_label_position('left')  # set the position of the second y-axis to left

    ax1.spines['bottom'].set_position(('outward', 80))
    ax2.spines['left'].set_position(('outward', 130))

    ax1.set_xlabel('mm', fontsize=20)
    ax2.set_ylabel('mm', fontsize=20)

    plt.draw()
    plt.show()

    # Save plot
    fig.savefig(str(uid) + ".tif", bbox_inches='tight')


def display_cropped_image(img, uid, x1=900, x2=1650, y1=750, y2=1400, dpi=80,
                          eiger_size_per_pixel=0.075, clim=(0, 100),
                          cmap='gist_stern'):
    '''Display CHX eiger image cropped to user specifications and save the
       image as a TIFF with dual pixel and mm axis.

    :param im: eiger detector image.
    :param uid: unique ID automatically assigned to a CHX measurement.
    :param x1: x-axis stating location (columns).
    :param x2: x-axis final location (columns).
    :param y1: y-axis stating location (rows).
    :param y2: y-axis final location (rows).
    :param dpi=80: dpi (dots per inch) for output image.
    :param eiger_size_per_pixel=0.075: eiger camera has 75 um per pixel.
    :param cmap='gist_stern': color map (https://matplotlib.org/examples/color/colormaps_reference.html)
    :param clim: sets the color limits of the current image.
    '''
    img_data = img
    height, width = img_data.shape
    croppedBoxHeight, croppedBoxWidth = x2-x1, y2-y1
    pixelTickMarkStepSize = 100

    # Crop the image from the center of the image
    img_data = img[x1:x2, y1:y2]

    # What size does the figure need to be in inches to fit the image?
    figsize = croppedBoxWidth / float(dpi), croppedBoxHeight / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Title
    plt.title("UID: " + str(uid) + "  Cropped: " + str(croppedBoxWidth) + "x" + str(croppedBoxHeight), fontsize=15)

    # Set up pixel axis
    ax.axis('on')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Pixels', fontsize=15)
    plt.ylabel('Pixels', fontsize=15)
    ax.set_xticks(range(0, croppedBoxHeight+pixelTickMarkStepSize, pixelTickMarkStepSize))
    ax.set_yticks(range(0, croppedBoxWidth+pixelTickMarkStepSize, pixelTickMarkStepSize))
    ax.set_xticklabels(range(x1, x2+pixelTickMarkStepSize, pixelTickMarkStepSize))
    ax.set_yticklabels(range(y1, y2+pixelTickMarkStepSize, pixelTickMarkStepSize))

    # Set up microm axis
    ax1 = ax.twiny()  # Create a twin Axes sharing the yaxis
    ax2 = ax.twinx()  # Create a twin Axes sharing the xaxis

    # Decide the ticklabel position in the new micron axis,
    # then convert them to the position in the old axis so that the pixel and micron axis line up properly
    # Scale to start at Zero; labels of the ticklabels: the position in the new axis
    newlabelX = range(0, int(eiger_size_per_pixel*croppedBoxHeight), 10)
    newlabelY = range(0, int(eiger_size_per_pixel*croppedBoxWidth), 10)
    pixel2micronX = lambda croppedBoxHeight: croppedBoxHeight*eiger_size_per_pixel  # convert function X: from pixels to microns
    pixel2micronY = lambda croppedBoxWidth: croppedBoxWidth*eiger_size_per_pixel  # convert function Y: from pixels to microns
    newposX = [pixel2micronX(croppedBoxHeight) for croppedBoxHeight in newlabelX]  # position of the xticklabels in the old x-axis
    newposY = [pixel2micronY(croppedBoxWidth) for croppedBoxWidth in newlabelY]  # position of the yticklabels in the old y-axis

    ax1.set_xticks(newposX)
    ax1.set_xticklabels(newlabelX)
    ax1.tick_params(axis="x", labelsize=15)

    ax2.set_yticks(newposY)
    ax2.set_yticklabels(newlabelY)
    ax2.tick_params(axis="y", labelsize=15)

    ax1.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
    ax2.yaxis.set_ticks_position('left')  # set the position of the second y-axis to left

    ax1.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
    ax2.yaxis.set_label_position('left')  # set the position of the second y-axis to left

    ax1.spines['bottom'].set_position(('outward', 50))
    ax2.spines['left'].set_position(('outward', 80))

    ax1.set_xlabel('mm', fontsize=15)
    ax2.set_ylabel('mm', fontsize=15)

    # Display the image.
    ax.imshow(img_data, cmap=cmap, clim=clim)
    plt.savefig(str(uid) + "_cropped.tif", bbox_inches='tight')
    plt.show()

    # Save plot
    fig.savefig(str(uid) + "_cropped.tif", bbox_inches='tight')


def plot_eiger_for_srw(uid, det='eiger4m_single_image', cmap='afmhot',
                       clim=(0, 100), mean=False, frame_num=0, grid=False):
    '''Display CHX eiger image: fullsize, cropped to user specifications, and with horizontal and
        vertical cuts, and save the plots and images as a TIFFs.

    :param uid: unique ID automatically assigned to a CHX measurement.
    :param det='eiger4m_single_image': which eiger dector.
    :param cmap='gist_stern': color map (https://matplotlib.org/examples/color/colormaps_reference.html)
    :param clim=(0, 200): sets the color limits of the current image.
    :param mean=False: mean of combined images along axis 0.
    :param frame_num=0: which image to use.
    :param grid=False: grid on the image.
    '''
    plt.figure()
    h = db[uid]
    print(h.fields())

    # Get image data
    imgs = h.data('eiger4m_single_image')
    imgs = list(imgs)
    print(np.shape(imgs))

    # Mean of images
    if not mean:
        d = imgs[frame_num][0]
    else:
        d = np.mean(imgs[0][:], axis=0)
    print('min: {}, max: {}'.format(d.min(), d.max()))

    # Print full sized eiger image
    display_image_in_actual_size(d, uid, clim=clim, cmap=cmap)

    # Print cropped Image
    display_cropped_image(d, uid, clim=clim, cmap=cmap)

    # Show plot of intensity versus horizontal position, and plot of intensity versus vertical position.
    plot_profile_horiz(imgs[frame_num], uid)
    plot_profile_vert(imgs[frame_num], uid)

    np.savetxt(str(uid) + ".dat", d)
    if grid:
        plt.grid()
    return imgs
