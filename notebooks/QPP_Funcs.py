import numpy as np
from celerite.modeling import Model

def expf(t, a, b):
    return np.exp(a*(t-b))

def linef(t, x1, y1, x2, y2):
    m = (y2-y1)/(x2-x1)
    b = y2 - (m*x2)
    return ((m*t)+b)

def PW(t, params):
    result = np.empty(len(t))
    for i in range(len(t)):
        if(t[i]<params[0]):
            result[i] = expf(t[i], params[2], params[3])
        elif(params[0]<=t[i] and t[i]<=params[1]):
            result[i] = linef(t[i], params[0], expf(params[0], params[2], params[3]), params[1], expf(params[1], params[4], params[5]))
        elif(params[1]<t[i]):
            result[i] = expf(t[i], params[4], params[5])
    return result

def DExp(t, params):
    result = np.empty(len(t))
    for i in range(len(t)):
        if(t[i]<params[0]):
            result[i] = expf(t[i], params[1], params[2])
        elif(params[0]<t[i]):
            result[i] = expf(t[i], params[3], params[4])
    return result

def CTS(t, params):
    lam = np.exp(np.sqrt(2*(params[1]/params[2])))
    return params[0] * lam * np.exp((-params[1]/t)-(t/params[2]))

def simulate(N, params, model, yerrsize=20000, bounds=(0,2000)):
    x = (np.random.rand(N) *(bounds[1]-bounds[0])) + bounds[0]
    y = model(x, params)
    yerr = (np.random.rand(N) * 2 * yerrsize) - yerrsize
    y += yerr
    return x, y

class PWModel(Model):
    parameter_names = ("xl", "xr", "al", "bl", "ar", "br")
    
    def get_value(self, t):
        result = np.empty(len(t))
        for i in range(len(t)): #had to tweak this to accept t-arrays, may affect speed...
            if(t[i]<self.xl):
                result[i] = expf(t[i], self.al, self.bl)
            elif(self.xl<=t[i] and t[i]<=self.xr):
                result[i] = linef(t[i], self.xl, expf(self.xl, self.al, self.bl), self.xr, expf(self.xr, self.ar, self.br))
            elif(self.xr<t[i]):
                result[i] = expf(t[i], self.ar, self.br)
        return result
    
    #the gradient terms were manually calculated
    def compute_gradient(self, t):
        yl = np.exp(self.al*(self.xl-self.bl))
        yr = np.exp(self.ar*(self.xr-self.br))
        result = np.empty([len(t), 6])
        result2 = np.empty([6, len(t)])
        for i in range(len(t)):
            ylt = np.exp(self.al*(t[i]-self.bl))
            yrt = np.exp(self.ar*(t[i]-self.br))
            if(t[i]<self.xl):
                dxl = 0.
                dxr = 0.
                dal = (t[i]-self.bl) * ylt
                dbl = -1* self.al * ylt
                dar = 0.
                dbr = 0.
                result[i] = np.array([dxl, dxr, dal, dbl, dar, dbr])
                result2[:,i] = result[i]

            elif(self.xl<=t[i] and t[i]<=self.xr):
                term = (t[i]-self.xr)
                dxl = ((term)/((self.xr-self.xl)**2)) * ((yr-yl) - (self.al * yl * (self.xr-self.xl)))
                dxr = (((term)/((self.xr-self.xl)**2)) * ((self.ar * yr * (self.xr-self.xl))-(yr-yl))) - ((yr-yl)/(self.xr-self.xl)) + (self.ar * yr)
                dal = ((term)/(self.xr-self.xl)) * (yl * (self.bl-self.xl))
                dbl = ((term)/(self.xr-self.xl)) *(self.al * yl)
                dar = (((term)/(self.xr-self.xl))+1) * ((self.xr-self.br)*yr)
                dbr = (((term)/(self.xr-self.xl))+1) * (-1*(self.ar*yr))
                result[i] = np.array([dxl, dxr, dal, dbl, dar, dbr])
                result2[:,i] = result[i]
        

            elif(self.xr<t[i]):
                dxl = 0.
                dxr = 0.
                dal = 0.
                dbl = 0.
                dar = (t[i]-self.br) * yrt
                dbr = -1 * self.ar * yrt
                result[i] = np.array([dxl, dxr, dal, dbl, dar, dbr])
                result2[:,i] = result[i]

        return result2

class DExpModel(Model):
    parameter_names = ("xc", "al", "bl", "ar", "br")
    
    def get_value(self, t):
        result = np.empty(len(t))
        for i in range(len(t)): #had to tweak this to accept t-arrays, may affect speed...
            if(t[i]<self.xc):
                result[i] = expf(t[i], self.al, self.bl)
            elif(self.xc<t[i]):
                result[i] = expf(t[i], self.ar, self.br)
        return result
    
    #the gradient terms were manually calculated
    def compute_gradient(self, t):
        result = np.empty([len(t), 5])
        result2 = np.empty([5, len(t)])
        for i in range(len(t)):
            ylt = np.exp(self.al*(t[i]-self.bl))
            yrt = np.exp(self.ar*(t[i]-self.br))
            
            if(t[i]<self.xc):
                dxc = 0
                dal = (t[i]-self.bl) * ylt
                dbl = -1* self.al * ylt
                dar = 0.
                dbr = 0.
                result[i] = np.array([dxc, dal, dbl, dar, dbr])
                result2[:,i] = result[i]

            elif(self.xc<=t[i]):
                dxc = 0.
                dal = 0.
                dbl = 0.
                dar = (t[i]-self.br) * yrt
                dbr = -1 * self.ar * yrt
                result[i] = np.array([dxc, dal, dbl, dar, dbr])
                result2[:,i] = result[i]
        return result2
    
class CTSModel(Model):
    parameter_names = ("A", "tau1", "tau2")
    def get_value(self, t):
        lam = np.exp(np.sqrt(2*(self.tau1/self.tau2)))
        return self.A*lam*np.exp((-self.tau1/t)-(t/self.tau2))
    #the gradient terms were manually calculated
    def compute_gradient(self, t):
        lam = np.exp(np.sqrt(2*(self.tau1/self.tau2)))
        dA = (1./self.A) * self.get_value(t)
        dtau1 = ((1/(self.tau2 * np.log(lam))) - (1/t)) * self.get_value(t)
        dtau2 = ((t/(self.tau2**2)) - (self.tau1/((self.tau2**2) * np.log(lam)))) * self.get_value(t)
        return np.array([dA, dtau1, dtau2])


