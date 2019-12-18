# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

try:
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
except:
    pass
import numpy as np
from ..util.general import reshape


class function2d:
    '''
    This is a benchmark of bi-dimensional functions interesting to optimize. 

    '''
    
    def plot(self):
        bounds = self.bounds
        x1 = np.linspace(bounds[0][0], bounds[0][1], 100)
        x2 = np.linspace(bounds[1][0], bounds[1][1], 100)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.hstack((X1.reshape(100*100,1),X2.reshape(100*100,1)))
        Y = self.f(X)
        
        #fig = plt.figure()
        #ax = fig.gca(projection='3d')
        #ax.plot_surface(X1, X2, Y.reshape((100,100)), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        #ax.zaxis.set_major_locator(LinearLocator(10))
        #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        #ax.set_title(self.name)    
            
        plt.figure()    
        plt.contourf(X1, X2, Y.reshape((100,100)),100)
        if (len(self.min)>1):    
            plt.plot(np.array(self.min)[:,0], np.array(self.min)[:,1], 'w.', markersize=20, label=u'Observations')
        else:
            plt.plot(self.min[0][0], self.min[0][1], 'w.', markersize=20, label=u'Observations')
        plt.colorbar()
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title(self.name)
        plt.show()


class rosenbrock(function2d):
    '''
    Cosines function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds is  None: self.bounds = [(-0.5,3),(-1.5,2)]
        else: self.bounds = bounds
        self.min = [(0, 0)]
        self.fmin = 0
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.name = 'Rosenbrock'

    def f(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            fval = 100*(X[:,1]-X[:,0]**2)**2 + (X[:,0]-1)**2
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return -fval.reshape(n,1) + noise


class beale(function2d):
    '''
    Cosines function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds is  None: self.bounds = [(-1,1),(-1,1)]
        else: self.bounds = bounds
        self.min = [(0, 0)]
        self.fmin = 0
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.name = 'Beale'

    def f(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            fval = 100*(X[:,1]-X[:,0]**2)**2 + (X[:,0]-1)**2
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return -fval.reshape(n,1) + noise


class dropwave(function2d):
    '''
    Cosines function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds is  None: self.bounds = [(-1,1),(-1,1)]
        else: self.bounds = bounds
        self.min = [(0, 0)]
        self.fmin = 0
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.name = 'dropwave'

    def f(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            fval = - (1+np.cos(12*np.sqrt(X[:,0]**2+X[:,1]**2))) / (0.5*(X[:,0]**2+X[:,1]**2)+2) 
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return -fval.reshape(n,1) + noise


class cosines(function2d):
    '''
    Cosines function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds is  None: self.bounds = [(0,1),(0,1)]
        else: self.bounds = bounds
        self.min = [(0.31426205,  0.30249864)]
        self.fmin = -1.59622468
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.name = 'Cosines'

    def f(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            u = 1.6*X[:,0]-0.5
            v = 1.6*X[:,1]-0.5
            fval = 1-(u**2 + v**2 - 0.3*np.cos(3*np.pi*u) - 0.3*np.cos(3*np.pi*v) )
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return -fval.reshape(n,1) + noise

class simpletime(function2d):
    '''
    simpletime function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,a= 0.5,b=0.,sd=None):
        self.input_dim = 2
        if bounds is  None: self.bounds = [(0,1),(0,1)]
        else: self.bounds = bounds
        if a==None: self.a = 0.5
        else: self.a = a           
        if b==None: self.b = 0.
        else: self.b = b
        self.min = [(0., 0.)]
        self.fmin = None
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.name = 'simplefunc'

    def f(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        def fval(x1, x2, a=self.a, b=self.b):
            y = 1. + a*x1 + b*x2
            return y
        mini = min(fval(self.bounds[0][0],self.bounds[1][0]),fval(self.bounds[0][0],self.bounds[1][1]),fval(self.bounds[0][1],self.bounds[1][0]),fval(self.bounds[0][1],self.bounds[1][1]))
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:,0]
            x2 = X[:,1]
            val = fval(x1,x2) - mini +1.
            if self.sd == 0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return val.reshape(n,1) + noise


class branin(function2d):
    '''
    Branin function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,a=None,b=None,c=None,r=None,s=None,t=None,sd=None, normalized=False):
        self.normalized = normalized
        self.input_dim = 2
        if bounds is  None:
            if normalized == True:
                self.bounds = [(0,1),(0,1)]
            else:
                self.bounds = [(-5,10),(1,15)]
        else: self.bounds = bounds
        if a==None: self.a = 1
        else: self.a = a           
        if b==None: self.b = 5.1/(4*np.pi**2)
        else: self.b = b
        if c==None: self.c = 5/np.pi
        else: self.c = c
        if r==None: self.r = 6
        else: self.r = r
        if s==None: self.s = 10 
        else: self.s = s
        if t==None: self.t = 1/(8*np.pi)
        else: self.t = t    
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.min = [(-np.pi,12.275),(np.pi,2.275),(9.42478,2.475)] 
        self.fmin = 0.397887
        self.name = 'Branin'
    
    def f(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim: 
            return 'Wrong input dimension' 
        elif self.normalized == False:
            x1 = X[:,0]
            x2 = X[:,1]
            fval = self.a * (x2 - self.b*x1**2 + self.c*x1 - self.r)**2 + self.s*(1-self.t)*np.cos(x1) + self.s 
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return fval.reshape(n,1)+ 1. + noise
        else:
            _bounds = [(-5,10),(1,15)]
            x1 = (_bounds[0][1]-_bounds[0][0])*(X[:,0]/(self.bounds[0][1]-self.bounds[0][0])) + _bounds[0][0]
            x2 = (_bounds[1][1]-_bounds[1][0])*(X[:,1]/(self.bounds[1][1]-self.bounds[1][0])) + _bounds[1][0]
            _fval = self.a * (x2 - self.b*x1**2 + self.c*x1 - self.r)**2 + self.s*(1-self.t)*np.cos(x1) + self.s 
            fval = ((_fval- self.fmin)/270.)
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return fval.reshape(n,1) + noise

    
    def ftime(self,X,mini=100., scale=100):
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim: 
            return 'Wrong input dimension'  
        else:
            x1 = X[:,0]
            x2 = X[:,1]
            fval = self.a * (x2 - self.b*x1**2 + self.c*x1 - self.r)**2 + self.s*(1-self.t)*np.cos(x1) + self.s 
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return (fval.reshape(n,1)+ mini + noise)/scale


class goldstein(function2d):
    '''
    Goldstein function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None, normalized = False):
        self.normalized = normalized
        self.input_dim = 2
        if bounds is  None:
            if normalized == True:
                self.bounds = [(0,1),(0,1)]
            else:
                self.bounds = [(-2,2),(-2,2)]
        else: self.bounds = bounds
        self.min = [(0,-1)]
        self.fmin = 3
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.name = 'Goldstein'

    def f(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        elif self.normalized == False:
            x1 = X[:,0]
            x2 = X[:,1]
            fact1a = (x1 + x2 + 1)**2
            fact1b = 19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2
            fact1 = 1 + fact1a*fact1b
            fact2a = (2*x1 - 3*x2)**2
            fact2b = 18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2
            fact2 = 30 + fact2a*fact2b
            fval = fact1*fact2
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return fval.reshape(n,1) + noise
        else: 
            _bounds = [(-2,2),(-2,2)]
            x1 = (_bounds[0][1]-_bounds[0][0])*(X[:,0]/(self.bounds[0][1]-self.bounds[0][0])) + _bounds[0][0]
            x2 = (_bounds[1][1]-_bounds[1][0])*(X[:,1]/(self.bounds[1][1]-self.bounds[1][0])) + _bounds[1][0]
            fact1a = (x1 + x2 + 1)**2
            fact1b = 19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2
            fact1 = 1 + fact1a*fact1b
            fact2a = (2*x1 - 3*x2)**2
            fact2b = 18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2
            fact2 = 30 + fact2a*fact2b
            _fval = fact1*fact2
            fval = ((_fval-3)/1e+6)
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return fval.reshape(n,1) + noise  




class sixhumpcamel(function2d):
    '''
    Six hump camel function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None, normalized = False):
        self.normalized = normalized
        self.input_dim = 2
        if bounds is  None:
            if normalized == True:
                self.bounds = [(0,1),(0,1)]
            else:
                self.bounds = [(-2,2),(-1,1)]
        else: self.bounds = bounds
        self.min = [(0.0898,-0.7126),(-0.0898,0.7126)]
        self.fmin = -1.0316
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.name = 'Six-hump camel'

    def f(self,x):
        x = reshape(x,self.input_dim)
        n = x.shape[0]
        if x.shape[1] != self.input_dim:
            return 'wrong input dimension'
        elif self.normalized == False:
            x1 = x[:,0]
            x2 = x[:,1]
            term1 = (4-2.1*x1**2+(x1**4)/3) * x1**2
            term2 = x1*x2
            term3 = (-4+4*x2**2) * x2**2
            fval = term1 + term2 + term3
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return fval.reshape(n,1) + noise
        else:
            _bounds = [(-2,2),(-1,1)]
            x1 = (_bounds[0][1]-_bounds[0][0])*(x[:,0]/(self.bounds[0][1]-self.bounds[0][0])) + _bounds[0][0]
            x2 = (_bounds[1][1]-_bounds[1][0])*(x[:,1]/(self.bounds[1][1]-self.bounds[1][0])) + _bounds[1][0]
            term1 = (4-2.1*x1**2+(x1**4)/3) * x1**2
            term2 = x1*x2
            term3 = (-4+4*x2**2) * x2**2
            _fval = term1 + term2 + term3
            fval =(_fval - self.fmin)/7.
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return fval.reshape(n,1) + noise 



class mccormick(function2d):
    '''
    Mccormick function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds is  None: self.bounds = [(-1.5,4),(-3,4)]
        else: self.bounds = bounds
        self.min = [(-0.54719,-1.54719)]
        self.fmin = -1.9133
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.name = 'Mccormick'

    def f(self,x):
        x = reshape(x,self.input_dim)
        n = x.shape[0]
        if x.shape[1] != self.input_dim:
            return 'wrong input dimension'
        else:
            x1 = x[:,0]
            x2 = x[:,1]
            term1 = np.sin(x1 + x2)
            term2 = (x1 - x2)**2
            term3 = -1.5*x1
            term4 = 2.5*x2
            fval = term1 + term2 + term3 + term4 + 1
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return fval.reshape(n,1) + noise


class powers(function2d):
    '''
    Powers function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds is  None: self.bounds = [(-1,1),(-1,1)]
        else: self.bounds = bounds
        self.min = [(0,0)]
        self.fmin = 0
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.name = 'Sum of Powers'

    def f(self,x):
        x = reshape(x,self.input_dim)
        n = x.shape[0]
        if x.shape[1] != self.input_dim:
            return 'wrong input dimension'
        else:
            x1 = x[:,0]
            x2 = x[:,1]
            fval = abs(x1)**2 + abs(x2)**3
            fval = abs(x1) + abs(x2)
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return fval.reshape(n,1) + noise

class l1(function2d):
    '''
    Powers function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds is  None: self.bounds = [(-1,1),(-1,1)]
        else: self.bounds = bounds
        self.min = [(0,0)]
        self.fmin = 0
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.name = 'Sum of Powers'

    def f(self,x):
        x = reshape(x,self.input_dim)
        n = x.shape[0]
        if x.shape[1] != self.input_dim:
            return 'wrong input dimension'
        else:
            x1 = x[:,0]
            x2 = x[:,1]
            #fval = abs(x1)**2 + abs(x2)**3
            fval = abs(x1-0.5) + abs(x2+0.2)
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return fval.reshape(n,1) + noise

class eggholder:
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds is  None: self.bounds = [(-512,512),(-512,512)]
        else: self.bounds = bounds
        self.min = [(512,404.2319)]
        self.fmin = -959.6407
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.name = 'Egg-holder'

    def f(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:,0]
            x2 = X[:,1]
            fval = -(x2+47) * np.sin(np.sqrt(abs(x2+x1/2+47))) + -x1 * np.sin(np.sqrt(abs(x1-(x2+47))))
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return fval.reshape(n,1) + noise












