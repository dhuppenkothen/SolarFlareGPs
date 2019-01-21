import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import celerite as ce
import emcee as mc
import corner
import h5py
import os
import dynesty
import astropy.io
from celerite.modeling import Model
from scipy.optimize import minimize, curve_fit
from astropy.io import fits
from numpy import linalg
from celerite.solver import LinAlgError
from dynesty import utils as dyfunc
from IPython.display import clear_output

SHO_prior_bounds  = [(np.log(0), np.log(1e9)),(np.log(0), np.log(20)), (-10, 0)]
CTSModel_prior_bounds  = [(np.log(1), np.log(1e7)), (np.log(1), np.log(1e4)), (np.log(1), np.log(1e7)), (-10, 10)]
RealTerm_prior_bounds  = [(-20,20), (-20,10)]

class RealTerm_Prior(ce.terms.RealTerm):
    name = "RealTerm_Prior"
    def log_prior(self):
        prob_a = 1.
        prob_c = 1.


#defining model class
class CTSModel_prior(Model):
    name="CTSModel_prior"
    parameter_names = ("log_A", "log_tau1", "log_tau2", "log_bkg")
 
    def get_value(self, t):
        A = np.exp(self.log_A)
        tau1 = np.exp(self.log_tau1)
        tau2 = np.exp(self.log_tau2)
        lam = np.exp(np.sqrt(2*np.exp(tau1/tau2)))
        bkg = np.exp(self.log_bkg)
        return A*lam*np.exp((-tau1/t)-(t/tau2)) + bkg
    
    #the gradient terms were manually calculated
    def compute_gradient(self, t):
        A = np.exp(self.log_A)
        tau1 = np.exp(self.log_tau1)
        tau2 = np.exp(self.log_tau2)
        lam = np.exp(np.sqrt(2*np.exp(tau1/tau2)))
        dA = (1./A) * self.get_value(t)
        dtau1 = ((1/(np.sqrt(2*tau1*tau2))) - (1/t)) * self.get_value(t)
        dtau2 = ((t/(tau2**2))-(tau1/((tau2**2)*np.sqrt(2*tau1/tau2)))) * self.get_value(t)
        return np.array([dA, dtau1, dtau2])
        

    #defining our somewhat naive prior, a simple tophat distribution for each parameter
    #SUBJECT TO CHANGE!!!

    def log_prior(self):
        probA = 1.
        probtau1 = 1.
        probtau2 = 1.
        probbkg = 1.
        prior = CTSModel_prior_bounds 
        if not (self.log_A>prior[0][0] and self.log_A<prior[0][1]): 
            probA = 0.
        if not ((self.log_tau1>prior[1][0] and self.log_tau1<prior[1][1])):
            probtau1 = 0.
        if not ((self.log_tau2>prior[2][0] and self.log_tau2<prior[2][1])):
            probtau2 = 0.
        if not ((self.log_bkg>prior[3][0] and self.log_bkg<prior[3][1])):
            probbkg = 0.
        return np.log(probA * probtau1 * probtau2 * probbkg * np.e)

#QPO term being redefined for our inclusion of priors
class SHOTerm_Prior(ce.terms.SHOTerm):
    name = "SHOTerm_Prior"

    def log_prior(self):
        prob_S0 = 1.
        prob_Q = 1.
        prob_omega0 = 1.
        prior = SHO_prior_bounds 
        
        #again, using simple (naive) tophat distributions
        if not ((self.log_S0 > prior[0][0]) and (self.log_S0 < prior[0][1])):
            prob_S0 = 0.
        if not (self.log_Q > prior[1][0] and self.log_Q < prior[1][1]):
            prob_Q = 0.
        if not (self.log_omega0 > prior[2][0] and self.log_omega0 < prior[2][1]):
            prob_omega0 = 0.
        return np.log(prob_S0*prob_Q*prob_omega0 * np.e)
    

#Rednoise term being redefined for our inclusion of priors
class RealTerm_Prior(ce.terms.RealTerm):
    name = "RealTerm_Prior"
    def log_prior(self):
        prob_a = 1.
        prob_c = 1.
        prior = RealTerm_prior_bounds 
        #again, using simple (naive) tophat distributions
        if not (self.log_a > prior[0][0] and self.log_a < prior[0][1]):
            prob_a = 0.
        if not (self.log_c > prior[1][0] and self.log_c < prior[1][1]):
            prob_c = 0.
        return np.log(prob_a*prob_c * np.e)

#produces a sample from the GP prior, optional poisson nose   
def simulate(x, model, kernel, noisy = False):
    K = kernel.get_value(x[:, None] - x[None, :])
    y = np.abs(np.random.multivariate_normal(model.get_value(x), K))
    if (noisy==True):
        y = np.random.poisson(y)
    return np.abs(y)


#for dynesty, positive log likelihood function
def log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return gp.log_likelihood(y)

#defining fitting functions for our GP (neg-log-like for minimization optimization)
def neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)

#similarly produces the gradient of our fit-function
def grad_neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.grad_log_likelihood(y)[1]

#defining absolute probability fit function for sampling purposes
def neg_log_prob(params, y, gp):
    return -log_probability(params, y, gp)

#coarse fitting function for initialization of our model pre-optimizer
def initguess (x, y):
    A = .8*max(y)
    t1 = int(x[-1:]*1./7)
    t2 = int(x[-1:]*1./5)
    return A, t1, t2

#optimizes hyperparameters to fit the envelope shape
def optimize_gp(gp, y, verbose = False):
    if verbose:
        print("Initial log-likelihood: {0}".format(gp.log_likelihood(y)))
    bounds = gp.get_parameter_bounds()
    initial_params = gp.get_parameter_vector()
    soln = minimize(neg_log_prob, initial_params, method="L-BFGS-B", args=(y, gp))
    #soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like, method="L-BFGS-B", args=(y, gp))
    if verbose:
        print("Final log-likelihood: {0}".format(-soln.fun))
        print("Optimized log-parameters: " + str(soln.x))
    return soln

#produces a normally distributed collection of parameters to initialize sampler
def samplepdf(params, scale=1):
    return np.random.normal(loc=params, scale = np.sqrt(np.abs(params))*scale)

#computes the log-probability for sampling, with case solutions for nans and infs
def log_probability(params, y, gp):
    gp.set_parameter_vector(params)
    lp = gp.log_prior()
    try:
        ll = gp.log_likelihood(y)
    except ce.solver.LinAlgError as err:
        ll = np.nan
    except RuntimeError:
        ll = np.nan

    result = ll + lp
    if not (np.isfinite(lp)):
        return -np.inf
    if np.isnan(ll)==True:
        return -np.inf
    return result


#runs GP through emcee sampler, including burnin period
def sample_gp(paramstart, y, gp, nwalkers = 100, nsteps = 2000, burnin = 500, verbose = False):
    sampler = mc.EnsembleSampler(nwalkers, len(paramstart), log_probability, args=(y, gp))
    if verbose:
        print("Burning in...")
    p0 = paramstart + 1e-8 * np.random.randn(nwalkers, len(paramstart))
    p0, lp, _ = sampler.run_mcmc(p0, burnin)
    if verbose:
        print("Sampling...")
    sampler.reset()
    sampler.run_mcmc(p0, nsteps)
    if verbose:
        print("Done!")
    return sampler

def make_prior_transform(prior_bounds):
    def prior(prior_vec):
        prior_transformed = np.empty(len(prior_vec))
        for i in range(len(prior_vec)):
            prior_transformed[i] = (prior_bounds[i][1] - prior_bounds[i][0]) * prior_vec[i] + prior_bounds[i][0]
        return prior_transformed
    return prior

def make_loglike(y, gp):
    def loglike(params):
        gp.set_parameter_vector(params)
        return gp.log_likelihood(y)
    return loglike

#sample GP but with Dynamic Sampling (Dynesty)
def dynesty_sample_gp(ndim, loglike, prior_transform, nlive = 250):
    sampler =  dynesty.DynamicNestedSampler(loglike, prior_transform, ndim,   nlive=nlive)
    print ("Sampling ...")
    sampler.run_nested()
    res = sampler.results
    samples, weights = res.samples, np.exp(res.logwt-res.logz[-1])
    chain = dyfunc.resample_equal(samples, weights)
    return chain
    

#plots time series of chain, given the chain directly extracted from emcee sampler object
def plot_chain(chain,  labels = None, burstid=None):
    flat_samples = chain[:,:,:].reshape((-1,len(chain[0,0])))
    meanparams = np.mean(flat_samples, axis=0)
    nwalkers = len(chain)
    nsteps = len(chain[0])
    ndim = len(chain[0][0])
    fig, axarr = plt.subplots(ndim, sharex=True, figsize = (10,10))
    if burstid!= None:
        plt.suptitle("Chain Time Series for Burst " + str(burstid))
    xline = np.linspace(0,nsteps)
    for j in range(ndim):
        for i in range(nwalkers):
            axarr[j].plot(np.arange(nsteps), chain[i,:,j], 'k-', alpha=1./np.log(nwalkers))
        meanvals = meanparams[j] * np.ones(50)
        axarr[j].plot(xline, meanvals, 'r--')
        if (labels != None):
            axarr[j].set_title(labels[j])
    return fig

#similarly plots corner of chain from chain object extracted from sampler object
def plot_corner(chain, labels = None, truevals = None, burstid = None, flat = False):
    if labels==None:
        labels = str(np.arange(dim))
    if flat==False:
        flat_samples = chain[:,:, :].reshape((-1,dim))
    else:
        flat_samples = chain
    dim = len(flat_samples[0])
    
    maxparams = np.empty(dim)
    bounds = []
    for i in range(dim):
        hist, bin_edges = np.histogram(flat_samples[:,i], bins = 50)
        maxindex = np.argmax(hist)
        maxparams[i] = np.average([bin_edges[maxindex], bin_edges[maxindex+1]])
        bounds.append((np.average(flat_samples[:,i])-3*np.std(flat_samples[:,i]),np.average(flat_samples[:,i])+3*np.std(flat_samples[:,i])))
    if truevals == None:
        truevals = maxparams
    fig = corner.corner(flat_samples, bins=50, labels = labels, truths = truevals, range=bounds)
    if burstid!= None:
        plt.suptitle("Chain Corner Plot for Burst " + str(burstid))
    return fig, maxparams

#the multifunctional plot method for plotting lightcurves
def plot_gp(x, y, yerr, gp, model, label = "Prediction", predict=False, chain=[], burstid = None, flat = False):
    init_params = gp.get_parameter_vector()
    fig = plt.figure()
    
    if burstid!= None:
        plt.suptitle("Burst and fit for Burst " + str(burstid))
    #plots lightcurve with error bars (pretty useless given the density of data but oh well)
    plt.errorbar(x, y, yerr=yerr, fmt='k.', alpha = 0.05, label = "Data")
    
    if (predict == True):
        #plots model prediction, given current GP parameters
        ytest = gp.mean.get_value(x)
        plt.plot(x, ytest, 'r--', label = label)
    if(len(chain)>1):
        #plots posterior samples
        labeled = False
        if flat == False:
            nsteps = len(chain[0])
            nwalkers = len(chain)
        else:
            nsteps = len(chain)
        for i in range(5):
            if flat == False:
                params = chain[np.random.randint(nwalkers),nsteps-1]
            else:
                params = chain[np.random.randint(nsteps-1)]
            gp.set_parameter_vector(params)
            model.set_parameter_vector(params[-len(gp.mean.get_parameter_vector()):])
            ymc = gp.sample_conditional(y,x)
            if not np.isnan(ymc).any():  
                if labeled == False:
                    plt.plot(x, ymc, 'm-', label = "Posterior Samples")
                    labeled = True
                else: 
                    plt.plot(x, ymc, 'm-')
    plt.legend()
    return fig

#loads data from GOES format, can return header
def load_data(datadir, burstid, returnhead=False):
    f = datadir+'go'+str(burstid)+'.fits'
    hdulist = fits.open(f)
    data = hdulist[2].data
    time = data.field('TIME')
    flux = np.sum(data.field('FLUX'), axis=2)[0]
    time = time[0]-time[0,0]
    if returnhead==True:
        return time, flux, hdulist[0].header
    return time, flux

#locates flare and returns flare + relevant local data (determined by buffer size)
def trim_data(time, flux, cutoff = .05, buffer_ratio = 0.2):
    maxin = np.where(flux == np.max(flux))[0][0]
    threshold = (np.max(flux)-np.min(flux))*cutoff + np.min(flux) #threshold determines when the flare "ends"
    pre = maxin - (np.where(flux[:maxin] < threshold)[0])[-1:][0]
    post = (np.where(flux[maxin:] < threshold)[0])[0]
    length = pre+post
    buffer = int(length*buffer_ratio)
    pre += buffer
    post += buffer
    
    if maxin<pre:
        pre=maxin-1
    if len(time)-maxin<post:
        post = len(time)-maxin -1
    time_trim = time[(maxin-pre):(maxin+post)]
    if (len(time_trim)==0):
        print("\nNo time at index: " + str(maxin) + "\nPre: " + str(pre) + "\tPost: " + str(post))
    time_trim = time_trim-time_trim[0]
    flux_trim = flux[(maxin-pre):(maxin+post)]
    return time_trim, flux_trim

#stores flare using h5py
def store_flare(fname, header, t, I, optparams, chain, results):
    f = h5py.File(fname, "w")
    if not isinstance(header, str):
        stringhead = header.tostring()
    else:
        stringhead = header
    
    chain_dset = f.create_dataset("chain", chain.shape, dtype = chain.dtype, data=chain)
    results_dset = f.create_dataset("results", (100000,), dtype=h5py.special_dtype(vlen=str))
    results_dset[0] = str(results)
    optparams_dset = f.create_dataset("optparams", optparams.shape, dtype = optparams.dtype, data=optparams)
    header_dset = f.create_dataset("header", (100,), dtype=h5py.special_dtype(vlen=str))
    header_dset[0] = stringhead
    flare = np.array((t,I))
    flare_dset = f.create_dataset("flare", flare.shape, dtype = flare.dtype, data=flare)
    f.close()
    print("Stored flare at " + fname)

#loads flare using h5py
def load_flare(fname, astroheader=False):
    f = h5py.File(fname,"r")
    head_load = f['header']
    chain_load = f['chain']
    results_load = f['results']
    flare_load = f['flare']
    optparams_load = f['optparams']
    if astroheader==True:
        return astropy.io.fits.Header.fromstring(head_load[0]), flare_load[0,:], flare_load[1,:], optparams_load, chain_load, results_load[0]
    else:
        return head_load[0], flare_load[0,:], flare_load[1,:], optparams_load, chain_load, results_load[0]
    f.close()

#essentially a clone of plot_gp but takes a subplot as an input to more easily plot multiple lightcurves in the same figure
def plot_gp_subplot(ax, x, y, yerr, gp, model, label = "Prediction", predict_color=None, chain=[0], burstid = None):
    init_params = gp.get_parameter_vector()
    if burstid!= None:
        ax.set_title("Burst and fit for Burst " + str(burstid))
    ax.errorbar(x, y, yerr=yerr, fmt='k.', alpha = 0.05, label = "Data")
    
    if (predict_color != None):
        ytest = gp.mean.get_value(x)
        ax.plot(x, ytest, predict_color, label = label)
       

#plots posterior samples in subplots                    
def plot_post_subplot(ax, x, y, yerr, gp, params, fmt = 'g-', label = None, alpha = 1):
    gp.set_parameter_vector(params)
    gp.compute(x, yerr)
    ymc = gp.sample_conditional(y,x)
    ax.plot(x, ymc, 'm-', fmt, label = label, alpha = alpha)

def analyze_flare(loc, t, I, qpolabel = 'unnamed flare', trueparams = None):
    fname = loc + qpolabel + '/'
    if os.path.exists(fname):
            print("Exists at: " + fname)
            return

    bound_vec1 = SHO_prior_bounds  + CTSModel_prior_bounds
    bound_vec2 = RealTerm_prior_bounds + CTSModel_prior_bounds

    prior_transform1 = make_prior_transform(bound_vec1)
    prior_transform2 = make_prior_transform(bound_vec2)

    qpoparams = [np.log(3), np.log(3), 3]
    realparams = [0., 0.] 
    modelparams = [11.33844804, 6.92311406, 6.85207764, np.log(1000)]

    ndim1 = len(bound_vec1)
    ndim2 = len(bound_vec2)
            
    kernel1 = SHOTerm_Prior(log_S0 = qpoparams[0], log_Q = qpoparams[1], log_omega0 = qpoparams[2])
    kernel2 = RealTerm_Prior(log_a = realparams[0], log_c = realparams[1])

    A_guess, t1_guess, t2_guess = initguess(t,I)
    model = CTSModel_prior(log_A = np.log(A_guess), log_tau1 = np.log(t1_guess), log_tau2 = np.log(t2_guess), log_bkg = np.log(1000))
  
    gp1 = ce.GP(kernel1, mean=model, fit_mean=True)
    gp1.compute(t, np.sqrt(I))
    initparams1 = gp1.get_parameter_vector()
    
    gp2 = ce.GP(kernel2, mean=model, fit_mean=True)
    gp2.compute(t, np.sqrt(I))
    initparams2 = gp2.get_parameter_vector()

    loglike1 = make_loglike(I, gp1)
    loglike2 = make_loglike(I, gp2)

    soln1 = optimize_gp(gp1, I)
    gp1.set_parameter_vector(soln1.x)
    figopt1 = plot_gp(t, I, np.sqrt(I), gp1, model, predict=True, label = "Optimized fit QPO", flat=True)

    soln2 = optimize_gp(gp2, I)
    gp2.set_parameter_vector(soln2.x)
    figopt2 = plot_gp(t, I, np.sqrt(I), gp2, model, predict=True, label = "Optimized fit No QPO", flat=True)
    
    sampler1 =  dynesty.DynamicNestedSampler(loglike1, prior_transform1, ndim1, bound="multi", sample="rwalk", nlive=1000)
    sampler2 =  dynesty.DynamicNestedSampler(loglike2, prior_transform2, ndim2, bound="multi", sample="rwalk", nlive=1000)


    print ("Sampling QPO ...")
    sampler1.run_nested()
    res1 = sampler1.results
    bayesfac1 = res1.logz[-1:]
    samples1, weights1 = res1.samples, np.exp(res1.logwt-res1.logz[-1])
    chain1 = dyfunc.resample_equal(samples1, weights1)

    print ("Sampling No QPO ...")
    sampler2.run_nested()
    res2 = sampler2.results
    bayesfac2 = res2.logz[-1:]
    samples2, weights2 = res2.samples, np.exp(res2.logwt-res2.logz[-1])
    chain2 = dyfunc.resample_equal(samples2, weights2)
    


    #figsam = plot_chain(chain, labels = gp.get_parameter_names(), burstid = qpolabel, flat=True)

    try:
        figoptsam1 = plot_gp(t, I, np.sqrt(I), gp1, model, chain=chain1, burstid = qpolabel, predict=True, flat=True)
    except LinAlgError:
        figoptsam1 = plot_gp(t, I, np.sqrt(I), gp1, model, chain=chain1, burstid = qpolabel, predict=True, flat=True)

    try:
        figoptsam2 = plot_gp(t, I, np.sqrt(I), gp2, model, chain=chain2, burstid = qpolabel, predict=True, flat=True)
    except LinAlgError:
        figoptsam2 = plot_gp(t, I, np.sqrt(I), gp2, model, chain=chain2, burstid = qpolabel, predict=True, flat=True)

        
        
    figcorner1, maxparams1 = plot_corner(chain1, labels = gp1.get_parameter_names(), burstid = qpolabel, flat=True)
    figcorner2, maxparams2 = plot_corner(chain2, labels = gp2.get_parameter_names(), burstid = qpolabel, flat=True)
    bayesresult = bayesfac1-bayesfac2

    header1 = "QPO Params: \t" + str(qpoparams) + "\nEnvelope Params \t" + str(modelparams) + "\nTrue Parameter Vector: \t" + "\nBayes Factor: \t" + str(bayesresult)
    header2 = "Rednoise Kernel Params: \t"+str(realparams) +  "\nEnvelope Params \t" + str(modelparams) + "\nTrue Parameter Vector: \t" + "\nBayes Factor: \t" + str(bayesresult)


    if trueparams!=None:
        header += '\nTrue Params: ' + str(trueparams)


    if not os.path.exists(fname):
        os.makedirs(fname)
        
        store_flare(fname+"data1", header, t, I, soln1.x, chain1, res1)
        figoptsam1.savefig(fname+"lc_plot1.png")
        figcorner1.savefig(fname+"chain_corner1.png")
        
        store_flare(fname+"data2", header, t, I, soln2.x, chain2, res2)
        figoptsam2.savefig(fname+"lc_plot2.png")
        figcorner2.savefig(fname+"chain_corner2.png")
        
        
        textheader = open(fname+"header.txt",'w')
        textheader.write(header)
        textheader.close()
    else:
        print ("directory exists")
     
