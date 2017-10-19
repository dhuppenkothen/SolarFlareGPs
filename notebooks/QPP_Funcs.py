import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import celerite as ce
import emcee as mc
import corner
from celerite.modeling import Model
from scipy.optimize import minimize, curve_fit


#no way of setting prior in constructor? simply redefine log_prior?
class CTSModel_prior(Model):
    name="CTSModel_prior"
    parameter_names = ("log_A", "log_tau1", "log_tau2")
 
    def get_value(self, t):
        lam = np.exp(np.sqrt(2*np.exp(self.log_tau1-self.log_tau2)))
        return np.exp(self.log_A)*lam*np.exp((-np.exp(self.log_tau1)/t)-(t/np.exp(self.log_tau2)))
    
    #the gradient terms were manually calculated
    def compute_gradient(self, t):
        lam = np.exp(np.sqrt(2*np.exp(self.log_tau1-self.log_tau2)))
        dA = (1./np.exp(self.log_A)) * self.get_value(t)
        dtau1 = ((1/(np.exp(self.log_tau2) * np.log(lam))) - (1/t)) * self.get_value(t)
        dtau2 = ((t/(np.exp(self.log_tau2)**2)) - (np.exp(self.log_tau1)/((np.exp(self.log_tau2)**2) * np.log(lam)))) * self.get_value(t)
        return np.array([dA, dtau1, dtau2])
        

    #defining our somewhat naive prior, a simple tophat distribution for each parameter

    def log_prior(self):
        probA = 1.
        probtau1 = 1.
        probtau2 = 1.
        T=2000.
        if not (self.log_A>0 and self.log_A<25):
            probA = 0.
        if not ((self.log_tau1>0 and self.log_tau1<15)):
            probtau1 = 0.
        if not ((self.log_tau2>0 and self.log_tau2<15)):
            probtau2 = 0.
        return np.log(probA * probtau1 * probtau2 * np.e)


class SHOTerm_Prior(ce.terms.SHOTerm):
    name = "SHOTerm_Prior"

    def log_prior(self):
        prob_S0 = 1.
        prob_Q = 1.
        prob_omega0 = 1.
        
        #again, using simple (naive) tophat distributions
        if not ((self.log_S0 > -10) and (self.log_S0 < 50)):
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
        if not (self.log_a > -20 and self.log_a < 10):
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

#use poisson noise, with rate paremeter
#np.random.poisson(model(t))
#


#defining fitting functions for our GP
def neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)

def grad_neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.grad_log_likelihood(y)[1]

def initguess (x, y):
    A = max(y)
    t1 = max(x) * (1./3.)
    t2 = max(x) * (2./3.)
    return A, t1, t2

def optimize_gp(gp, y):
    print("Initial log-likelihood: {0}".format(gp.log_likelihood(y)))
    bounds = gp.get_parameter_bounds()
    initial_params = gp.get_parameter_vector()
    soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like, method="L-BFGS-B", bounds=bounds, args=(y, gp))
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

def plot_chain(chain):
    flat_samples = chain[:,200:, :].reshape((-1,len(chain[0,0])))
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
    return fig

def plot_corner(chain, labels = None):
    dim = len(chain[0,0])
    if labels==None:
        labels = str(np.arange(dim))
    flat_samples = chain[:,200:, :].reshape((-1,dim))

    maxparams = np.empty(dim)
    for i in range(dim):
        hist, bin_edges = np.histogram(flat_samples[:,i], bins = 50)
        maxindex = np.argmax(hist)

        maxparams[i] = np.average([bin_edges[maxindex], bin_edges[maxindex+1]])

    fig = corner.corner(flat_samples, bins=50, labels = labels, truths = maxparams,  range = np.ones(dim))
    return fig, maxparams

def plot_gp(x, y, yerr, gp, model, soln=0, chain=[0]):
    init_params = gp.get_parameter_vector()
    fig = plt.figure()
    plt.errorbar(x, y, yerr=yerr, fmt='k.', alpha = 0.05, label = "Data")
    if not(soln == 0):
        ytest, yvar = gp.predict(y, x, return_var=True)
        ystd = np.sqrt(yvar)
        plt.plot(x, ytest, 'r--', label = "Optimized Prediction")
        plt.fill_between(x, ytest+ystd, ytest-ystd, color='r', alpha=0.3, edgecolor='none')

    if not(len(chain) == 1):
        labeled = False
        nsteps = len(chain[0])
        nwalkers = len(chain)
        for i in range(nsteps/10):
            params = chain[np.random.randint(nwalkers),np.random.randint(100,nsteps)]
            gp.set_parameter_vector(params)
            model.set_parameter_vector(params[-3:])
            ymc, ymcvar = gp.predict(y, x, return_var=True)
            ymcstd = np.sqrt(ymcvar)
            if not np.isnan(ymc).any():  
                if labeled == False:
                    plt.plot(x, ymc, 'm-', alpha = 0.1, label = "Posterior Predictions")
                    plt.fill_between(x, ymc+ymcstd, ymc-ymcstd, color='g', alpha=0.1, edgecolor='none')
                    labeled = True
                else: 
                    plt.plot(x, ymc, 'm-', alpha = 0.1)
                    plt.fill_between(x, ymc+ymcstd, ymc-ymcstd, color='g', alpha=0.1, edgecolor='none')
    plt.legend()
    return fig