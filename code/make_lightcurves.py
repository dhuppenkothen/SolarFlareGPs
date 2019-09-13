import glob
import argparse

import numpy as np
import pandas as pd

import astropy.io.fits as fits

from write_wget_fermigbm import read_catalogue

def read_fermi_gbm_file(filename, tstart, duration, emin=10, emax=30, add_seconds=10):
    """
    Read a CTIME file from Fermi/GBM.

    Parameters
    ----------
    filename : str
        A Fermi/GBM CTIME file name + path

    tstart : float
        The start time for the burst in Fermi Mission Elapsed Time (MET)

    duration : float 
        The duration of the flare in seconds

    emin, emax: float, float
        The minimum and maximum energies to use in keV

    add_seconds : float
        The number of seconds to add before and after the burst for a 
        baseline for the bakground.

    Returns
    -------
    tnew, cnew : np.array, np.array
        Time and counts arrays. Counts are summed over channels


    """
    hdulist = fits.open(filename)
 
    # get out channel-to-energy conversion
    channel = hdulist[1].data.field("CHANNEL")
    e_min_channel = hdulist[1].data.field("E_MIN")
    e_max_channel = hdulist[1].data.field("E_MAX")

    # only the channels between emin and emax are interesting
    valid_channels = channel[(e_min_channel >= emin) & (e_max_channel <= emax)]

    # get out the time and counts arrays
    time = hdulist[2].data.field("TIME")
    counts = hdulist[2].data.field("COUNTS")

    valid_counts = counts[:, valid_channels]
    valid_counts = np.sum(valid_counts, axis=1)

    # start and end times
    tend = tstart + duration
    ts = tstart - add_seconds
    te = tend + add_seconds

    min_ind = time.searchsorted(ts)
    max_ind = time.searchsorted(te)

    tnew = time[min_ind:max_ind]
    cnew = valid_counts[min_ind:max_ind]

    return tnew, cnew

def convert_hours_seconds(time):
    """
    Convert a HH:MM:SS string into 
    seconds since midnight.

    Parameters
    ----------
    time : str
        A time in HH:MM:SS format

    Returns
    -------
    tnew : float 
        seconds since midnight

    """
    hours = float(time[:2])*3600.0
    minutes = float(time[3:5])*60.0
    seconds = float(time[6:])
    tnew = hours + minutes + seconds
    return tnew

def rebin_data(x, y, dx_new, yerr=None, method='sum', dx=None):
    """Rebin some data to an arbitrary new data resolution. Either sum
    the data points in the new bins or average them.

    Parameters
    ----------
    x: iterable
        The dependent variable with some resolution ``dx_old = x[1]-x[0]``

    y: iterable
        The independent variable to be binned

    dx_new: float
        The new resolution of the dependent variable ``x``

    Other parameters
    ----------------
    yerr: iterable, optional
        The uncertainties of ``y``, to be propagated during binning.

    method: {``sum`` | ``average`` | ``mean``}, optional, default ``sum``
        The method to be used in binning. Either sum the samples ``y`` in
        each new bin of ``x``, or take the arithmetic mean.

    dx: float
        The old resolution (otherwise, calculated from median diff)

    Returns
    -------
    xbin: numpy.ndarray
        The midpoints of the new bins in ``x``

    ybin: numpy.ndarray
        The binned quantity ``y``

    ybin_err: numpy.ndarray
        The uncertainties of the binned values of ``y``.

    step_size: float
        The size of the binning step
    """

    y = np.asarray(y)
    yerr = np.zeros_like(y)

    #dx_old = assign_value_if_none(dx, np.median(np.diff(x)))
    dx_old = np.diff(x)

    if np.any(dx_new < dx_old):
        raise ValueError("New frequency resolution must be larger than "
                         "old frequency resolution.")


    # left and right bin edges
    # assumes that the points given in `x` correspond to 
    # the left bin edges
    xedges = np.hstack([x, x[-1]+np.diff(x)[-1]])

    # new regularly binned resolution
    xbin = np.arange(xedges[0], xedges[-1]+dx_old[-1], dx_new)

    output, outputerr, step_size = [], [], []

    for i in range(xbin.shape[0]-1):
        total = 0
        total_err = 0

        nn = 0

        xmin = xbin[i]
        xmax = xbin[i+1]
        min_ind = xedges.searchsorted(xmin)
        max_ind = xedges.searchsorted(xmax)

        total += np.sum(y[min_ind:max_ind-1])
        total_err += np.sum(yerr[min_ind:max_ind-1])
        nn += len(y[min_ind:max_ind-1])

        prev_dx = xedges[min_ind] - xedges[min_ind-1]
        prev_frac = (xedges[min_ind] - xmin)/prev_dx
        total += y[min_ind-1]*prev_frac
        total_err += yerr[min_ind-1]*prev_frac
        nn += prev_frac

        if xmax <= xedges[-1]:
            dx_post = xedges[max_ind] - xedges[max_ind-1]
            post_frac = (xmax-xedges[max_ind-1])/dx_post
            total += y[max_ind-1]*post_frac
            total_err += yerr[max_ind-1]*post_frac
            nn += prev_frac


        output.append(total)
        outputerr.append(total_err)
        step_size.append(nn)

    output = np.asarray(output)
    outputerr = np.asarray(outputerr)

    if method in ['mean', 'avg', 'average', 'arithmetic mean']:
        ybin = output / np.float(step_size)
        ybinerr = outputerr / np.sqrt(np.float(step_size))

    elif method == "sum":
        ybin = output
        ybinerr = outputerr

    else:
        raise ValueError("Method for summing or averaging not recognized. "
                         "Please enter either 'sum' or 'mean'.")

    tseg = x[-1] - x[0] + dx_old

    #if (tseg / dx_new % 1) > 0:
    #    ybin = ybin[:-1]
    #    ybinerr = ybinerr[:-1]

    #new_x0 = (x[0] - (0.5 * dx_old)) + (0.5 * dx_new)
    #xbin = np.arange(ybin.shape[0]) * dx_new + new_x0

    return xbin[:-1], ybin, ybinerr, step_size


def make_lightcurve(date, tstart, duration, detecs, emin=10, emax=30, add_seconds=20, resolution=1.0,
                    datadir="./"):

    """
    For a flare, extract a light curve and save to a file.

    Parameters
    ----------
    date : str
        The date of the flare, in `YYMMDD` format.

    tstart : str 
        The start time of the flare, in HH:MM:SS format.

    duration : float 
        The duration of the flare in seconds

    detecs : iterable of str
        The Fermi/GBM detectors that pointed towards the sun

    emin, emax : float, float
        The minimum and maximum energy ranges

    add_seconds : float
        The number of seconds to add before and after the flare

    resolution : float
        The time resolution of the output light curve. Must be larger than the 
        native resolution of the time series, because this supports only binning, 
        not interpolation

    """
    d1 = detecs[0]

    f = glob.glob(datadir+"glg_ctime_%s_%s_*.pha"%(d1, date))
    print(datadir) 
    hdulist = fits.open(f[0])
    
    obs_start_met = hdulist[0].header["TSTART"]
    obs_end_met = hdulist[0].header["TSTOP"]
    
    # date plus start time of the observation, should be close to 
    # midnight, but not quite midnight
    date_obs = hdulist[0].header["DATE-OBS"]
    
    if int(date[-2:]) > int(date_obs.split("T")[0].split("-")[-1]):
        neg = True
    else: 
        neg = False
    
    # just the observation start time in HH:MM:SS
    obs_start = date_obs.split("T")[1]
    
    # convert to seconds since midnight:
    if neg:
        obs_start_sec = 86400 - convert_hours_seconds(obs_start)
    else:
        obs_start_sec = convert_hours_seconds(obs_start)
    
    hdulist.close()
    
    # start time in seconds since midnight
    tstart = convert_hours_seconds(tstart)
    
    # tstart in MET is the time since observation start, i.e. the 
    # seconds since midnight minus the seconds between midnight and the 
    # start of the observation plus the MET of the start of the observation
    tstart_met = tstart + obs_start_met + obs_start_sec
    
    
    # loop over all detectors and get out the light curve for each for 
    # the given start time and duration
    c_all = []

    # TODO: FIX THIS FOR LOOP!!!!
    for d in detecs[1:]:
        f = glob.glob(datadir+"glg_ctime_%s_%s_*.pha"%(d, date))
        tnew, cnew = read_fermi_gbm_file(f[0], tstart_met, duration, emin=emin, emax=emax, add_seconds=add_seconds)
        c_all.append(cnew)

    c_all = np.array(c_all)

    # sum over all detectors
    csum = np.sum(c_all, axis=0)

    # compute length of each time bin
    tdiff = np.diff(tnew)
    # add the last time bin
    tdiff = np.hstack([tdiff, tdiff[-1]])

    # compute the count rate in counts/s
    countrate = csum/tdiff

    # bin to the new time resolution
    tbin, cbin, cbinerr, step_size = rebin_data(tnew, csum, resolution, yerr=None, method='sum')

    cbinrate = cbin/resolution
    # sort of hacky uncertainty on the countrate
    cerr = np.sqrt(cbinrate)

    outfile = "%s%s_%f_%ito%ikeV_lc.dat"%(datadir, date, tstart_met, int(emin), int(emax))

    np.savetxt(outfile, np.array([tbin, cbinrate, cerr]).T)

    return


def write_all_lightcurves(catfile, emin=10, emax=30, add_seconds=20, resolution=1.0, datadir="./"):
    """
    Take a catalogue file and write out light curves for all bursts in that 
    catalogue.

    Parameters
    ----------
    catfile : str
        The path and file name for the catalogue file

    emin, emax : float
        Minimum and maximum energy to consider, in keV

    add_seconds: int
        The number of seconds to add on either end of the Fermi/GBM burst start/end times
        Note: we should add some time on either side, because Fermi/GBM is not designed 
        for solar flares and tends to miss the start/end of the flare

    resolution : float
        The time resolution of the output light curve. Must be > 0.256 seconds, which 
        is the resolution of Fermi/GBM CTIME data.

    """
    catdf = read_catalogue(datadir+catfile)

    nflares = len(catdf.index)

    for i in catdf.index:
        print("I am on flare %i out of %i."%(i, nflares))
        date = catdf.loc[i, "date"]
        tstart = catdf.loc[i, "tstart"]
        duration = catdf.loc[i, "duration"]
        detecs = [catdf.loc[i, "det1"],catdf.loc[i, "det2"],catdf.loc[i, "det3"],catdf.loc[i, "det4"]]

        make_lightcurve(date, tstart, duration, detecs, emin=emin, emax=emax, 
                        add_seconds=add_seconds, resolution=resolution, datadir=datadir)
   
    return


# what to do if called from command line
if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(description="Making light curves out of Fermi/GBM files.")

    parser.add_argument("-f", "--filename", action="store", dest="filename", required=True, help="File name with catalogue")
    parser.add_argument("--emin", action="store", dest="emin", default=10, required=False, help="Lower boundary on energy channels.")
    parser.add_argument("--emax", action="store", dest="emax", default=30, required=False, help="Upper boundary on energy channels.")
    parser.add_argument("-s", "--seconds", action="store", dest="add_seconds", default=20, required=False, help="number of seconds to add before and after flare.")
    parser.add_argument("-r", "--resolution", action="store", dest="resolution", default=1.0, required=False, help="Time resolution of the output light curve.")
    parser.add_argument("-d", "--dir", action="store", dest="datadir", default="./", required=False, help="Directory with the data files.")
 
    clargs = parser.parse_args()

    filename = clargs.filename
    emin = np.float(clargs.emin)
    emax = np.float(clargs.emax)
    add_seconds = np.float(clargs.add_seconds)
    resolution = np.float(clargs.resolution)
    datadir = clargs.datadir

    write_all_lightcurves(filename, emin, emax, add_seconds, resolution, datadir)
