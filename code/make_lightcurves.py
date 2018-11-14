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
    hdulist = fits.open(f[0])

    obs_start_met = hdulist[0].header["TSTART"]

    # date plus start time of the observation, should be close to 
    # midnight, but not quite midnight
    date_obs = hdulist[0].header["DATE-OBS"]

    # just the observation start time in HH:MM:SS
    obs_start = date_obs.split("T")[1]

    # convert to seconds since midnight:
    obs_start_sec = convert_hours_seconds(obs_start)

    hdulist.close()

    # start time in seconds since midnight
    tstart = convert_hours_seconds(tstart) 

    # tstart in MET is the time since observation start, i.e. the 
    # seconds since midnight minus the seconds between midnight and the 
    # start of the observation plus the MET of the start of the observation
    tstart_met = tstart + obs_start_met - obs_start_sec

    # loop over all detectors and get out the light curve for each for 
    # the given start time and duration
    c_all = []
    for d in detecs:
        f = glob.glob("glg_ctime_%s_%s_*.pha"%(d, date))
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

    # sort of hacky uncertainty on the countrate
    cerr = np.sqrt(countrate)

    outfile = "%s_%f_lc.dat"%(date, tstart_met)

    np.savetxt(outfile, np.array([tnew, countrate, cerr]).T)

    return


def write_all_lightcurves(catfile, emin=10, emax=30, add_seconds=20, resolution=1.0):

    catdf = read_catalogue(catfile)

    nflares = len(catdf.index)

    for i in catdf.index:
        print("I am on flare %i out of %i."%(i, nflares))
        date = catdf.loc[i, "date"]
        tstart = catdf.loc[i, "tstart"]
        duration = catdf.loc[i, "duration"]
        detecs = [catdf.loc[i, "det1"],catdf.loc[i, "det2"],catdf.loc[i, "det3"],catdf.loc[i, "det4"]]

        make_lightcurve(date, tstart, duration, detecs, emin=emin, emax=emax, add_seconds=add_seconds, resolution=resolution)
   
    return


# what to do if called from command line
if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(description="Making light curves out of Fermi/GBM files.")

    parser.add_argument("-f", "--filename", action="store_true", dest="filename", required=True, help="File name with catalogue")
    parser.add_argument("--emin", action="store_true", dest="emin", default=10, required=False, help="Lower boundary on energy channels.")
    parser.add_argument("--emax", action="store_true", dest="emax", default=30, required=False, help="Upper boundary on energy channels.")
    parser.add_argument("-s", "--seconds", action="store_true", dest="add_seconds", default=20, required=False, help="number of seconds to add before and after flare.")
    parser.add_argument("-r", "--resolution", action="store_true", dest="resolution", default=1.0, required=False, help="Time resolution of the output light curve.")
   
    clargs = parser.parse_args()

    filename = clargs.filename
    emin = np.float(clargs.emin)
    emax = np.float(clargs.emax)
    add_seconds = np.float(clargs.add_seconds)
    resolution = np.float(clargs.resolution)

    write_all_lightcurves(filename, emin, emax, add_seconds, resolution)
