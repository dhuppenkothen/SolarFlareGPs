#use this block of code to load helper functions from one directory up
def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname('__file__'), fpath))


import numpy as np
import scipy as sp
from scipy import signal
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import celerite as ce
import os
from celerite.solver import LinAlgError
import dynesty
from dynesty import utils as dyfunc
import sys

load_src("QPP_Funcs", "../notebooks/QPP_Funcs.py")
import QPP_Funcs as qpp



SHO_prior_bounds  = [(np.log(1), np.log(1e9)),(np.log(2), np.log(20)), (-10, 0)]
CTSModel_prior_bounds  = [(np.log(1), np.log(1e7)), (np.log(1), np.log(1e4)), (np.log(1), np.log(1e7)), (-10, 10)]
RealTerm_prior_bounds  = [(-20,20), (-20,10)]

bound_vec = SHO_prior_bounds  + CTSModel_prior_bounds
bound_vec2 = RealTerm_prior_bounds + CTSModel_prior_bounds
prior_transform = qpp.make_prior_transform(bound_vec)
prior_transform2 = qpp.make_prior_transform(bound_vec2)


if __name__=='__main__':
    datelabel, s0, q = sys.argv[1:]
    s0 = float(s0)
    q = float(q)
    qpolabel = "simulated_burst_s0" + str(s0) + "_Q" + str(q)
    fname = "/scratch/ci411/Data/Simulating/" + datelabel + "/" + qpolabel + '/'
    if os.path.exists(fname):
        print("Exists at: " + fname)
        sys.exit()
    print("Running S: " + str(s0) + "\tQ: "+ str(q)  +"\nSaving at: " + fname + '\n')
    print(s0*q)
    qpoparams = [s0, q, -4]
    realparams = [-.13, -1.4] 
    modelparams = [11.33844804, 6.92311406, 6.85207764, np.log(1000)]
    trueparams = qpoparams + modelparams
    ndim = len(trueparams)
    ndim2 = len(bound_vec2)
    model = qpp.CTSModel_prior(log_A = modelparams[0], log_tau1 = modelparams[1], log_tau2 = modelparams[2], log_bkg = modelparams[3])
    kernel1 = qpp.SHOTerm_Prior(log_S0 = qpoparams[0], log_Q = qpoparams[1], log_omega0 = qpoparams[2])
    #kernel2 = qpp.RealTerm_Prior(log_a = realparams[0], log_c = realparams[1])
    kernel = kernel1

    t = np.linspace(0,4000,2000)
    I = qpp.simulate(t, model, kernel, noisy = True)



    A_guess, t1_guess, t2_guess = qpp.initguess(t,I)
    model = qpp.CTSModel_prior(log_A = np.log(A_guess), log_tau1 = np.log(t1_guess), log_tau2 = np.log(t2_guess), log_bkg = np.log(1000))
    kernel1 = qpp.SHOTerm_Prior(log_S0 = np.log(3), log_Q = np.log(3), log_omega0 = -6) #write guesser for kernel parameters
    kernel2 = qpp.RealTerm_Prior(log_a=0., log_c=0.) #write guesser for kernel parameters
    kernel = kernel1
    gp = ce.GP(kernel, mean=model, fit_mean=True)
    gp.compute(t, np.sqrt(I))
    initparams = gp.get_parameter_vector()

    gp2 = ce.GP(kernel2, mean=model, fit_mean=True)
    gp2.compute(t, np.sqrt(I))
    initparams2 = gp.get_parameter_vector()

    loglike = qpp.make_loglike(I, gp)
    loglike2 = qpp.make_loglike(I, gp2)

    soln = qpp.optimize_gp(gp, I)
    gp.set_parameter_vector(soln.x)
    figopt = qpp.plot_gp(t, I, np.sqrt(I), gp, model, predict=True, label = "Optimized fit", flat=True)

    soln2 = qpp.optimize_gp(gp2, I)
    gp2.set_parameter_vector(soln2.x)
    figopt2 = qpp.plot_gp(t, I, np.sqrt(I), gp2, model, predict=True, label = "Optimized fit", flat=True)

    sampler =  dynesty.DynamicNestedSampler(loglike, prior_transform, ndim, sample="rwalk", nlive=1000)
    sampler2 =  dynesty.DynamicNestedSampler(loglike2, prior_transform2, ndim2, sample="rwalk", nlive=1000)

    print "Sampling1 ..."
    sampler.run_nested()
    res = sampler.results
    bayesfac = res.logz[-1:]
    samples, weights = res.samples, np.exp(res.logwt-res.logz[-1])
    chain = dyfunc.resample_equal(samples, weights)

    print "Sampling2 ..."
    sampler2.run_nested()
    res2 = sampler2.results
    bayesfac2 = res2.logz[-1:]
    samples2, weights2 = res2.samples, np.exp(res2.logwt-res2.logz[-1])
    chain2 = dyfunc.resample_equal(samples2, weights2)


    #figsam = qpp.plot_chain(chain, labels = gp.get_parameter_names(), burstid = qpolabel, flat=True)

    try:
        figoptsam = qpp.plot_gp(t, I, np.sqrt(I), gp, model, chain=chain, burstid = qpolabel, predict=True, flat=True)
    except LinAlgError:
        figoptsam = qpp.plot_gp(t, I, np.sqrt(I), gp, model, chain=chain, burstid = qpolabel, predict=True, flat=True)

    try:
        figoptsam2 = qpp.plot_gp(t, I, np.sqrt(I), gp2, model, chain=chain2, burstid = qpolabel, predict=True, flat=True)
    except LinAlgError:
        figoptsam2 = qpp.plot_gp(t, I, np.sqrt(I), gp2, model, chain=chain2, burstid = qpolabel, predict=True, flat=True)

        
    figcorner, maxparams = qpp.plot_corner(chain, labels = gp.get_parameter_names(), truevals = trueparams, burstid = qpolabel, flat=True)
    figcorner2, maxparams2 = qpp.plot_corner(chain2, labels = gp2.get_parameter_names(), burstid = qpolabel, flat=True)
    bayesresult = bayesfac-bayesfac2

    header = "Rednoise Kernel Params: \t"+str(realparams) + "\nQPO Params: \t" + str(qpoparams) + "\nEnvelope Params \t" + str(modelparams) + "\nTrue Parameter Vector: \t" + str(trueparams) + "\nMaxParams: \t" + str(maxparams) + "\nBayes Factor: \t" + str(bayesresult)

    if not os.path.exists(fname):
        os.makedirs(fname)
        
        qpp.store_flare(fname+"data", header, t, I, soln.x, chain, res)
        figoptsam.savefig(fname+"lc_plot.png")
        figcorner.savefig(fname+"chain_corner.png")
        
        qpp.store_flare(fname+"data2", header, t, I, soln2.x, chain2, res2)
        figoptsam2.savefig(fname+"lc_plot2.png")
        figcorner2.savefig(fname+"chain_corner2.png")
        
        
        textheader = open(fname+"header.txt",'w')
        textheader.write(header)
        textheader.close()
    else:
        print "directory exists"
