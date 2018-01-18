import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import celerite as ce
import emcee as mc
import corner
from celerite.modeling import Model
from scipy.optimize import minimize, curve_fit
from astropy.io import fits


#no way of setting prior in constructor? simply redefine log_prior?
class CTSModel_prior(Model):
    name="CTSModel_prior"
    parameter_names = ("log_A", "log_tau1", "log_tau2")
 
    def get_value(self, t):
        A = np.exp(self.log_A)
        tau1 = np.exp(self.log_tau1)
        tau2 = np.exp(self.log_tau2)
        lam = np.exp(np.sqrt(2*np.exp(tau1/tau2)))
        return A*lam*np.exp((-tau1/t)-(t/tau2))
    
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

    def log_prior(self):
        probA = 1.
        probtau1 = 1.
        probtau2 = 1.
        if not (self.log_A>1 and self.log_A<25): 
            probA = 0.
        if not ((self.log_tau1>1 and self.log_tau1<15)):
            probtau1 = 0.
        if not ((self.log_tau2>1 and self.log_tau2<15)):
            probtau2 = 0.
        return np.log(probA * probtau1 * probtau2 * np.e)


class SHOTerm_Prior(ce.terms.SHOTerm):
    name = "SHOTerm_Prior"

    def log_prior(self):
        prob_S0 = 1.
        prob_Q = 1.
        prob_omega0 = 1.
        
        #again, using simple (naive) tophat distributions
        if not ((self.log_S0 > -20) and (self.log_S0 < 10)):
            prob_S0 = 0.
        if not (self.log_Q > -20 and self.log_Q < 20):
            prob_Q = 0.
        if not (self.log_omega0 > -20 and self.log_omega0 < 20):
            prob_omega0 = 0.
        return np.log(prob_S0*prob_Q*prob_omega0 * np.e)
    


class RealTerm_Prior(ce.terms.RealTerm):
    name = "RealTerm_Prior"
    def log_prior(self):
        prob_a = 1.
        prob_c = 1.
     
        #again, using simple (naive) tophat distributions
        if not (self.log_a > -20):# and self.log_a < 20):
            prob_a = 0.
        if not (self.log_c > -20 and self.log_c < 10):
            prob_c = 0.
        return np.log(prob_a*prob_c * np.e)

    
def simulate(x, model, kernel, noisy = False):
    K = kernel.get_value(x[:, None] - x[None, :])
    y = np.abs(np.random.multivariate_normal(model.get_value(x), K))
    if (noisy==True):
        y += np.random.poisson(y)
    return np.abs(y)


#defining fitting functions for our GP
def neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    #print params
    return -gp.log_likelihood(y)

def grad_neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.grad_log_likelihood(y)[1]

#defining fitting functions for our GP
def neg_log_prob(params, y, gp):
    return -log_probability(params, y, gp)

def initguess (x, y):
    A = max(y)
    t1 = max(x) * (1./5.)
    t2 = max(x) * (2./5.)
    return A, t1, t2

def optimize_gp(gp, y):
    print("Initial log-likelihood: {0}".format(gp.log_likelihood(y)))
    bounds = gp.get_parameter_bounds()
    initial_params = gp.get_parameter_vector()
    soln = minimize(neg_log_prob, initial_params, method="L-BFGS-B", args=(y, gp))
    #soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like, method="L-BFGS-B", args=(y, gp))
    print("Final log-likelihood: {0}".format(-soln.fun))
    print ("Optimized log-parameters: " + str(soln.x))
    return soln

def samplepdf(params, scale=1):
    return np.random.normal(loc=params, scale = np.sqrt(np.abs(params))*scale)

def log_probability(params, y, gp):
    gp.set_parameter_vector(params)
    lp = gp.log_prior()
    try:
        ll = gp.log_likelihood(y)
    except numpy.linalg.linalg.LinAlgError:
        print "Recurse"
        return log_probability(params, y, gp)
    except RuntimeError:
        ll = np.nan

    result = ll + lp
    if not (np.isfinite(lp)):
        return -np.inf
    if np.isnan(ll)==True:
        return -np.inf
    return result

def sample_gp(paramstart, y, gp, nwalkers = 100, nsteps = 2000, burnin = 500):
    sampler = mc.EnsembleSampler(nwalkers, len(paramstart), log_probability, args=(y, gp))
    print "Burning in..."
    p0 = paramstart + 1e-8 * np.random.randn(nwalkers, len(paramstart))
    p0, lp, _ = sampler.run_mcmc(p0, burnin)
    print "Sampling..."
    sampler.reset()
    sampler.run_mcmc(p0, nsteps)
    print "Done!"
    return sampler

def plot_chain(chain,  labels = None):
    flat_samples = chain[:,:,:].reshape((-1,len(chain[0,0])))
    meanparams = np.mean(flat_samples, axis=0)
    nwalkers = len(chain)
    nsteps = len(chain[0])
    ndim = len(chain[0][0])
    fig, axarr = plt.subplots(ndim, sharex=True, figsize = (10,10))
    xline = np.linspace(0,nsteps)
    for j in range(ndim):
        for i in range(nwalkers):
            axarr[j].plot(np.arange(nsteps), chain[i,:,j], 'k-', alpha=1./np.log(nwalkers))
        meanvals = meanparams[j] * np.ones(50)
        axarr[j].plot(xline, meanvals, 'r--')
        if (labels != None):
            axarr[j].set_title(labels[j])
    return fig

def plot_corner(chain, labels = None):
    dim = len(chain[0,0])
    if labels==None:
        labels = str(np.arange(dim))
    flat_samples = chain[:,:, :].reshape((-1,dim))

    maxparams = np.empty(dim)
    for i in range(dim):
        hist, bin_edges = np.histogram(flat_samples[:,i], bins = 50)
        maxindex = np.argmax(hist)

        maxparams[i] = np.average([bin_edges[maxindex], bin_edges[maxindex+1]])

    fig = corner.corner(flat_samples, bins=50, labels = labels, truths = maxparams)
    return fig, maxparams

def plot_gp(x, y, yerr, gp, model, label = "Prediction", predict=False, chain=[0]):
    init_params = gp.get_parameter_vector()
    fig = plt.figure()
    plt.errorbar(x, y, yerr=yerr, fmt='k.', alpha = 0.05, label = "Data")
    
    if (predict == True):
        ytest = gp.mean.get_value(x)
        plt.plot(x, ytest, 'r-', label = label)
       
    if not(len(chain) == 1):
        labeled = False
        nsteps = len(chain[0])
        nwalkers = len(chain)
        for i in range(nwalkers/10):
            params = chain[np.random.randint(nwalkers),nsteps-1]
            gp.set_parameter_vector(params)
            model.set_parameter_vector(params[-3:])
            ymc = gp.sample_conditional(y,x)
            aval = 1./nwalkers
            #ymc, ymcvar = gp.predict(y, x, return_var=True)
            #ymcstd = np.sqrt(ymcvar)
            if not np.isnan(ymc).any():  
                if labeled == False:
                    plt.plot(x, ymc, 'm-', alpha = aval, label = "Posterior Predictions")
                    #plt.fill_between(x, ymc+ymcstd, ymc-ymcstd, color='g', alpha=0.1, edgecolor='none')
                    labeled = True
                else: 
                    plt.plot(x, ymc, 'm-', alpha = aval)
                    #plt.fill_between(x, ymc+ymcstd, ymc-ymcstd, color='g', alpha=0.1, edgecolor='none')
    plt.legend()
    return fig

def load_data(datadir, burstid):
    f = datadir+'go'+str(burstid)+'.fits'
    hdulist = fits.open(f)
    data = hdulist[2].data
    time = data.field('TIME')
    flux = np.sum(data.field('FLUX'), axis=2)[0]
    time = time[0]-time[0,0]
    return time, flux

def trim_data(time, flux, pre=500, post=1500):
    maxin = np.where(flux == np.max(flux))[0][0]
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
