""" link_functions.py

Link function classes which include inverses and derivatives."""
from __future__ import print_function

import numpy as np
from scipy.special import erf, erfinv

### Base class
class LinkFunction(object):

	def __init__(self):
		self.name = "base class -- do not use directly."

	def __call__(self, x):
		raise NotImplementedError("Subclasses should implement this function!")

	def inverse(self, x):
		raise NotImplementedError("Subclasses should implement this function!")

	def derivative(self, x):
		raise NotImplementedError("Subclasses should implement this function!")

	def regularize(self, Y):

		""" Function returns a copy of Y where values of link(Y) = NaN are 
		exchanged with well defined values. In most cases, this should just 
		return a direct copy. """

		return np.copy(Y)

### Sub-classes
class IdentityLink(LinkFunction):	

	def __init__(self):
		self.name = "identity"
	
	def __call__(self, x):
		return x
	
	def inverse(self, x):
		return x
	
	def derivative(self, x):
		if isinstance(x,np.ndarray):
			return np.ones(x.shape)
		else:
			return 1.

class LogLink(LinkFunction):
	
	def __init__(self):
		self.name = "log"
	
	def __call__(self, x):
		return np.log(x)
	
	def inverse(self, x):
		return np.exp(x)
	
	def derivative(self, x):
		return 1./x

	def regularize(self, Y):
		mu = np.copy(Y)
		mu[mu == 0.] = 1.
		return mu

class LogitLink(LinkFunction):
	
	def __init__(self):
		self.name = "logit"
	
	def __call__(self, x):
		return np.log(x/(1.-x))
	
	def inverse(self, x):
		return 1./(1. + np.exp(-x))
	
	def derivative(self, x):
		return 1./(x*(1.-x))

class ProbitLink(LinkFunction):
	
	def __init__(self):
		self.name = "probit"

	def __call__(self, x):
		return np.sqrt(2.)*erfinv(2.*x - 1.)

	def inverse(self, x):
		return 0.5*(erf(x/np.sqrt(2.)) + 1.)

	def derivative(self, x):
		w = self.__call__(x)
		return np.sqrt(2.*np.pi)*np.exp(0.5*(w**2))


if __name__ == "__main__":

	link = IdentityLink()

	print(link(5))
	print(link.inverse(6))
	print(link.derivative(34.3))