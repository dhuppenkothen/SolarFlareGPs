import argparse

import numpy as np
import pandas as pd

def read_catalogue(filename, datapath="./"):
    # read in the catalogue
    catfile = open(datapath + filename, "r")
    catlines = catfile.readlines()
    catfile.close()
    

    catdict = {"date":[], "tstart":[], "hour":[], "minutes":[], "seconds":[],
               "duration":[], "peakcounts":[], "totalcounts":[], 
               "det1": [], "det2":[], "det3":[], "det4":[], "trigname":[], "rhessi":[]} 
    # run through all lines except for the first one
    # which contains the headers    
    for i,c in enumerate(catlines[1:]):
        i += 1
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

    date = np.array(catdf["date"])
    unique_date = np.unique(date)
    detecs = np.array([[d1, d2, d3, d4] for d1, d2, d3, d4 in zip(catdf["det1"], catdf["det2"], catdf["det3"], catdf["det4"])])

    f = open(datapath + filename, "w")

    for d in unique_date:
        if d[:2] == "10":
    
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
                f.write("wget heasarc.gsfc.nasa.gov:/fermi/data/gbm/daily/20%s/%s/%s/current/*ctime*%s* \n"%(year, month, day, b))
        else:
            continue 
    f.close()

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a bunch of Fermi/GBM light curves.")

    parser.add_argument("-c", "--catfile", action="store", dest="catfile", required=True, help="The filename of the catalogue file,")
    parser.add_argument("-o", "--outfile", action="store", dest="outfile", required=False, default="fermigbmcat_wget.dat", help="The name for the output file")
    parser.add_argument("-p", "--path", action="store", dest="datapath", required=False, default="./", help="The path to the data directory.")

    clargs = parser.parse_args()
    catfile = clargs.catfile
    outfile = clargs.outfile
    datapath = clargs.datapath

    catdf = read_catalogue(catfile, datapath)
    make_wget_list(catdf, outfile, datapath) 
