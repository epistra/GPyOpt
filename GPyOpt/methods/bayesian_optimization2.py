import GPy
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.special import erfc
from scipy.stats import norm
import matplotlib.pyplot as plt
import scipy
import random
from pylab import plot, xlabel, ylabel, title, grid
import matplotlib.pyplot as plt

# this will be replaced by a multidimensional lattice
from ..util.general import samples_multimensional_uniform, multigrid, reshape, ellipse 
from ..plotting.plots_bo import plot_acquisition, plot_convergence

from .acquisition import AcquisitionEI 

class BO(object):
        def __init__(self, acquisition_func, bounds=None, optimize_model=None,Nrandom =None):
		if bounds==None: 
			print 'Box contrainst are needed. Please insert box constrains'	
		else:
			self.bounds = bounds
			self.input_dim = len(self.bounds)		
                self.acquisition_func = acquisition_func
                if optimize_model == None: self.optimize_model = True
		else: self.optimize_model = optimize_model
		self.Ngrid = 5
		if Nrandom ==None: self.Nrandom = 3*self.input_dim
		else: self.Nrandom = Nrandom  # number or samples of initial random exploration
	
 
        def _init_model(self, X, Y):
                pass
                
	def start_optimization(self, f=None, H=None , X=None, Y=None):
		if f==None: print 'Function to optimize is requiered'
		else: self.f = f
		if H == None: H=0
		if X==None or Y == None:
			self.X = samples_multimensional_uniform(self.bounds,self.Nrandom)
			self.Y = f(self.X)
		else:
			self.X = X
			self.Y = Y
		print self.X
		print self.Y 
		self._init_model(self.X,self.Y)
                self.acquisition_func.model = self.model
		self._update_model()
		prediction = self.model.predict(self.X)
		self.m_in_min = prediction[0]
		self.s_in_min = prediction[1] 
			 
#		k=1
#		while k<=H:
#			# add new data point in the minumum)
 #                       self.X = np.vstack((self.X,self.suggested_sample))
#                        self.Y = np.vstack((self.Y,self.f(np.array([self.suggested_sample]))))
#			pred_min = self.model.predict(reshape(self.suggested_sample,self.input_dim))
#			self.m_in_min = np.vstack((self.m_in_min,pred_min[0]))
#			self.s_in_min = np.vstack((self.s_in_min,pred_min[1]))
#	      		self._update_model()
#			k +=1
			  
		self.optimization_started = True
		return self.continue_optimization(H)
	
	def continue_optimization(self,H):
		if self.optimization_started:
			k=1
			while k<=H:
				self.X = np.vstack((self.X,self.suggested_sample))
				self.Y = np.vstack((self.Y,self.f(np.array([self.suggested_sample]))))
				pred_min = self.model.predict(reshape(self.suggested_sample,self.input_dim))
				self.m_in_min = np.vstack((self.m_in_min,pred_min[0]))
				self.s_in_min = np.vstack((self.s_in_min,pred_min[1]))
				self._update_model()				
				k +=1
			return self.suggested_sample

		else: print 'Optimization not initiated: Use .start_optimization and provide a function to optimize'
		
#	def get_moments(self,x):
#		x = reshape(x,self.input_dim)
#		fmin = min(self.model.predict(self.X)[0])
#		m, s = self.model.predict(x)
#		return (m, s, fmin)

	def optimize_acquisition(self):
                # combines initial grid search with local optimzation starting on the minimum of the grid
                grid = multigrid(self.bounds,self.Ngrid)
                pred_grid = self.acquisition_func.acquisition_function(grid)
                x0 =  grid[np.argmin(pred_grid)]
                res = scipy.optimize.minimize(self.acquisition_func.acquisition_function, x0=np.array(x0), method='SLSQP',bounds=self.bounds)
                return res.x

	def _update_model(self):
                # Update X and Y in the model
                self.model.X = GPy.core.parameterization.ObsAr(self.X)
                self.model.Y = GPy.core.parameterization.ObsAr(self.Y)
                if self.optimize_model:
                        self.model.optimize_restarts(num_restarts = 5)
                        self.model.optimize()
		self.suggested_sample = self.optimize_acquisition()

	def plot_acquisition(self):
		return plot_acquisition(self.bounds,self.input_dim,self.model,self.X,self.Y,self.acquisition_func.acquisition_function)

	def plot_convergence(self):
		return plot_convergence(self.X,self.m_in_min,self.s_in_min)



####
####
####


class BayesianOptimizationEI(BO):
	def __init__(self, bounds=None, kernel=None, optimize_model=None, acquisition_par=None, invertsign=None, Nrandom = None):
		self.input_dim = len(bounds)
		if bounds==None: 
			raise 'Box contrainst are needed. Please insert box constrains'	
		if kernel is None: 
			self.kernel = GPy.kern.RBF(self.input_dim, variance=.1, lengthscale=.1)
		else: 
			self.kernel = kernel
		acq = AcquisitionEI(acquisition_par, invertsign)
		super(BayesianOptimizationEI ,self ).__init__(acq, bounds, optimize_model,Nrandom)
        
	def _init_model(self, X, Y):
		self.model = GPy.models.GPRegression(X,Y,kernel=self.kernel)





#class BayesianOptimizationEI(BO):
#	'''
#	Bayesian Optimization with EI
#	'''
#	def __init__(self, bounds=None, kernel=None, optimize_model=None, acquisition_par=None, invertsign=None, Nrandom = None):
#		if bounds==None: 
#			print 'Box contrainst are needed. Please insert box constrains'	
#		else:
#			self.bounds = bounds
#			self.input_dim = len(self.bounds)		
#		if acquisition_par == None: self.acquisition_par = 0.01
#		else: self.acquisition_par = acquisition_par 		
#		if kernel is None: self.kernel = GPy.kern.RBF(self.input_dim, variance=.1, lengthscale=.1)
#		else: self.kernel = kernel
#		if optimize_model == None: self.optimize_model = True
#		else: self.optimize_model = optimize_model		
#		if invertsign == None: self.sign = 1		
#		else: self.sign = -1	
#		if Nrandom ==None: self.Nrandom = 3*self.input_dim
#		else: self.Nrandom = Nrandom  # number or samples of initial random exploration 
#		self.Ngrid = 5
#		self.acquisition_function = self.expected_improvement
#
 #	def expected_improvement(self,x):
#		m, s, fmin = self.get_moments(x) 	
#		u = ((1+self.acquisition_par)*fmin-m)/s	
#		phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
#		Phi = 0.5 * erfc(-u / np.sqrt(2))	
#		f_acqu = self.sign * (((1+self.acquisition_par)*fmin-m) * Phi + s * phi)
#		return -f_acqu  # note: returns negative value for posterior minimization (but we plot +f_acqu)
#
#	def d_expected_improvement(self,x):
#		m, s, fmin = self.get_moments(x)
#		u = ((1+self.acquisition_par)*fmin-m)/s	
#		phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
#		Phi = 0.5 * erfc(-u / np.sqrt(2))	
#		dmdx, dsdx = self.model.predictive_gradients(x)
#		df_acqu =  self.sign* (-dmdx * Phi  + dsdx * phi)
#		return -df_acqu
#

#class BayesianOptimizationMPI(BO):
#	'''
#	Bayesian Optimization with MPI
#	'''
#	def __init__(self, bounds=None, kernel=None, optimize_model=None, acquisition_par=None, invertsign=None, Nrandom = None):
#		if bounds==None: 
#			print 'Box contrainst are needed. Please insert box constrains'	
#		else:
#			self.bounds = bounds
#			self.input_dim = len(self.bounds)		
#		if acquisition_par == None: self.acquisition_par = 0.01
#		else: self.acquisition_par = acquisition_par 		
#		if kernel is None: self.kernel = GPy.kern.RBF(self.input_dim, variance=.1, lengthscale=.1)
#		else: self.kernel = kernel
#		if optimize_model == None: self.optimize_model = True
#		else: self.optimize_model = optimize_model		
#		if invertsign == None: self.sign = 1		
#		else: self.sign = -1	
#		if Nrandom ==None: self.Nrandom = 3*self.input_dim
#		else: self.Nrandom = Nrandom  # number or samples of initial random exploration 
#		self.Ngrid = 5
#		self.acquisition_function = self.maximum_probability_improvement	
#		
#	def maximum_probability_improvement(self,x):   
#		m, s, fmin = self.get_moments(x) 
#		u = ((1+self.acquisition_par)*fmin-m)/s
#		Phi = 0.5 * erfc(-u / np.sqrt(2))
#		f_acqu =  self.sign*Phi
#		return -f_acqu # note: returns negative value for posterior minimization (but we plot +f_acqu)

#	def d_maximum_probability_improvement(self,x):
#		m, s, fmin = self.get_moments(x)
#		u = ((1+self.acquisition_par)*fmin-m)/s	
#		phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
#		Phi = 0.5 * erfc(-u / np.sqrt(2))	
#		dmdx, dsdx = self.model.predictive_gradients(x)
#		df_acqu =  self.sign* ((Phi/s)* (dmdx + dsdx + z))
#		return -df_acqu 
#
#lass BayesianOptimizationUCB(BO):
#	'''
#	Bayesian Optimization with EI
#	'''
#	def __init__(self, bounds=None, kernel=None, optimize_model=None, acquisition_par=None, invertsign=None, Nrandom = None):
#		if bounds==None: 
#			print 'Box contrainst are needed. Please insert box constrains'	
#		else:
#			self.bounds = bounds
#			self.input_dim = len(self.bounds)		
#		if acquisition_par == None: self.acquisition_par = 2
#		else: self.acquisition_par = acquisition_par 		
#		if kernel is None: self.kernel = GPy.kern.RBF(self.input_dim, variance=.1, lengthscale=.1)
#		else: self.kernel = kernel
#		if optimize_model == None: self.optimize_model = True
#		else: self.optimize_model = optimize_model		
#		if invertsign == None: self.sign = 1		
#		else: self.sign = -1	
#		if Nrandom ==None: self.Nrandom = 3*self.input_dim
#		else: self.Nrandom = Nrandom  # number or samples of initial random exploration 
#		self.Ngrid = 5
#		self.acquisition_function = self.upper_confidence_bound
#
#	def upper_confidence_bound(self,x):	
#		m, s, fmin = self.get_moments(x)
#		f_acqu = self.sign*(-m - self.sign* self.acquisition_par * s)
#		return -f_acqu  # note: returns negative value for posterior minimization (but we plot +f_acqu)
#			
#	def d_upper_confidence_bound(self,x):
#		x = reashape(x,self.input_dim)
#		dmdx, dsdx = self.model.predictive_gradients(x)		
#		df_acqu = self.sign*(-dmdx - self.sign* self.acquisition_par * dsdx) 
#		return -df_acqu




















