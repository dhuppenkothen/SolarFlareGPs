import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import celerite as ce
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
        if not (self.log_A>np.log(1e4) and self.log_A<np.log(3.5e7)):
            probA = 0.
        if not ((self.log_tau1>np.log(1) and self.log_tau1<np.log(T))):
            probtau1 = 0.
        if not ((self.log_tau2>np.log(1) and self.log_tau2<np.log(T))):
            probtau2 = 0.
        return np.log(probA * probtau1 * probtau2 * np.e)

class RealTerm_Prior(ce.terms.RealTerm):
    name = "RealTerm_Prior"
    def log_prior(self):
        prob_a = 1.
        prob_c = 1.
        
        #again, using simple (naive) tophat distributions
        if not ((self.log_a > -1e5) and (self.log_a < np.log(1e6))):
            prob_a = 0.
        if not (self.log_c > np.log(1./1000) and self.log_c < np.log(100)):
            prob_c = 0.
        return np.log(prob_a*prob_c * np.e)

def simulate(x, yerr, model, kernel):
    #generates a covariance matrix and then data using the multivariate normal distribution
    #could this be where the error is????
    K = kernel.get_value(x[:, None] - x[None, :])
    K[np.diag_indices(len(x))] += yerr**2
    y = np.random.multivariate_normal(model.get_value(x), K)
    return np.abs(y)

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
    print("Initial log-likelihood: {0}".format(gp.log_likelihood(ysim)))
    bounds = gp.get_parameter_bounds()
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

def sample_gp(paramstart, y, gp, nwalkers = 100, nsteps = 1500):
    start = [samplepdf(paramstartq,1e-10) for i in range(nwalkers)]
    print "Picking start..."
    for i in range(nwalkers):
        attempt = 0
        while(log_probability(start[i], ysimq, gpq)==-np.inf):
            attempt += 1
            start[i] = samplepdf(paramstartq, 1)
    print "Sampling..."
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(y, gp))
    sampler.run_mcmc(start, nsteps)
    return sampler.chain

def plot_gp(x, y, yerr, gp, model, soln=np.nan, chain=np.nan):
    init_params = gp.get_parameter_vector()
    fig = plt.figure()
    plt.errorbar(x, y, yerr=yerr, fmt='k.', label = "Data")
    if(np.isnan(soln)!):
        ytest, yvar = gp.predict(y, x, return_var=True)
        ystd = np.sqrt(yvar)
        plt.plot(x, ytest, 'r--', label = "Optimized Prediction")
        plt.fill_between(x, ytest+ystd, ytest-ystd, color='r', alpha=0.3, edgecolor='none')
        plt.plot(x, np.abs(ytestq-modelq.get_value(x)), 'g-', label = "Optimized Residual")

    if(np.isnan(chain)!):
        lamaled = False
        nsteps = len(chain[0])
        nwalkers = len(chain)
        for i in range(nsteps/10):
        params = chain[np.random.randint(nwalkers),np.random.randint(100,nsteps)]
            gp.set_parameter_vector(params)
            model.set_parameter_vector(params[3:])
            ymc, ymcvar = gp.predict(y, x, return_var=True)
            ymcstd = np.sqrt(ymcvar)
            gpnoisemc = ymc - model.get_value(x)
            if not np.isnan(ymc).any():  
                if labeled == False:
                    plt.plot(x, ymc, 'g-', alpha = 0.1, label = "Posterior Predictions")
                    plt.fill_between(x, ymc+ymcstd, ymc-ymcstd, color='g', alpha=0.1, edgecolor='none')
                    plt.plot(x, gpnoisemc, 'c-', label = "GP Prediction", alpha=0.1)
                    labeled = True
                    else: 
                        plt.plot(x, ymc, 'g-', alpha = 0.1)
                        plt.fill_between(x, ymc+ymcstd, ymc-ymcstd, color='g', alpha=0.1, edgecolor='none'
                        plt.plot(x, gpnoisemc, 'c-', alpha = 0.1)
    return fig