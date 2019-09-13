import argparse

import numpy as np
import pandas as pd

def read_catalogue(filename, datapath="./"):
    """ 
    Read in the Fermi/GBM solar flare catalogue and return 
    entries in a useful pandas DataFrame format.

    Parameters
    ----------
    filename : str
        Name of the Fermi/GBM catalogue file

    datapath : str
        path to the directory with the Fermi/GBM catalogue file

    Returns
    -------
    catdf : pandas.DataFrame
        A pandas.DataFrame with the catalogue data

    """

    # read in the catalogue
    catfile = open(datapath + filename, "r")
    catlines = catfile.readlines()
    catfile.close()
    

    catdict = {"date":[], "tstart":[], "hour":[], "minutes":[], "seconds":[],
               "duration":[], "peakcounts":[], "totalcounts":[], 
               "det1": [], "det2":[], "det3":[], "det4":[], "trigname":[], "rhessi":[]} 
    # run through all lines except for the first one
    # which contains the headers    
    for i,c in enumerate(catlines):
        #i += 1
        # split by spaces
        # need to do this because there are variable spaces
        # between columns and a variable number of columns
        csplit = c.split(" ")
        cfilter = list(filter(None, csplit))
        # get the date of the observation
        catdict["date"].append(cfilter[0][:6])
        tstart = cfilter[2]
        catdict["tstart"].append(tstart)
        # get the hours and days
        catdict["hour"].append(int(tstart[:2]))
        catdict["minutes"].append(int(tstart[3:5]))
        catdict["seconds"].append(int(tstart[6:8]))
        catdict["duration"].append(int(cfilter[5]))
        catdict["peakcounts"].append(int(cfilter[6]))
        catdict["totalcounts"].append(int(cfilter[7]))
 
        catdict["det1"].append(cfilter[8])
        catdict["det2"].append(cfilter[9])
        catdict["det3"].append(cfilter[10])
        catdict["det4"].append(cfilter[11])
 
        if len(cfilter) == 13:
            if cfilter[12][:2] == "bn":
                catdict["trigname"].append(cfilter[12][:-2])
                catdict["rhessi"].append("")
            elif cfilter[12] == "\n":
                catdict["trigname"].append("")
                catdict["rhessi"].append("")
            else:
                catdict["rhessi"].append(cfilter[12][:-2])
                catdict["trigname"].append("")
 
        elif len(cfilter) == 14:
                catdict["trigname"].append(cfilter[12])
                catdict["rhessi"].append(cfilter[13][:-2])
 
        else:
                catdict["trigname"].append("")
                catdict["rhessi"].append("")
 
    catdf = pd.DataFrame(catdict)
    return catdf
      
def make_wget_list(catdf, filename="fermigbmcat_wget.dat", datapath="./"):
    """
    Take the Fermi/GBM catalogue and make a list of wget commands that 
    will download the relevant files for all bursts. Then save those wget 
    commands in an .sh file that one can run from the command line.

    Parameters
    ---------
    catdf : pandas.DataFrame
        A pandas.DataFrame with the Fermi/GBM solar flare catalogue data

    filename : str
        The filename of the output file to store the wget commands in

    datapath : str
        The path to the directory where the file with the wget commands 
        will be stored

    """


    date = np.array(catdf["date"])
    unique_date = np.unique(date)
    detecs = np.array([[d1, d2, d3, d4] for d1, d2, d3, d4 in zip(catdf["det1"], catdf["det2"], catdf["det3"], catdf["det4"])])

    f = open(datapath + filename, "w")

    for d in unique_date:
 
        year = d[:2]
        month = d[2:4]
        day = d[4:]
    
        b_detecs = []
        for i,t in enumerate(date):
           if d in t:
               det = detecs[i]
               b_detecs.extend(det)
    
        b_detecs = np.unique(np.array(b_detecs))
 
        for b in b_detecs:
            f.write("wget heasarc.gsfc.nasa.gov:/fermi/data/gbm/daily/20%s/%s/%s/current/*ctime*%s* -P %s \n"%(year, month, day, b, datapath))
    f.close()

    return

if __name__ == "__main__":

    # set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Download a bunch of Fermi/GBM light curves.")

    parser.add_argument("-c", "--catfile", action="store", dest="catfile", required=True, help="The filename of the catalogue file,")
    parser.add_argument("-o", "--outfile", action="store", dest="outfile", required=False, default="fermigbmcat_wget.dat", help="The name for the output file")
    parser.add_argument("-p", "--path", action="store", dest="datapath", required=False, default="./", help="The path to the data directory.")

    # parse the arguments and store in variables
    clargs = parser.parse_args()
    catfile = clargs.catfile
    outfile = clargs.outfile
    datapath = clargs.datapath

    # read out the catalogue from file
    catdf = read_catalogue(catfile, datapath)

    # set up the wget commands for downloading the data
    make_wget_list(catdf, outfile, datapath) 
