""" noise_models.py

Distribution classes for the GLM, which basically just contain likelihood functions
and some of the np functionality."""
from __future__ import print_function

import numpy as np
from scipy.special import gamma

### Base class template
###############################################################################################################
class Noise(object):

	def __init__(self):
		self.name = "base class -- do not use directly."

	def a(self, phi):
		raise NotImplementedError("Subclasses should implement this function!")

	def b(self, theta):
		raise NotImplementedError("Subclasses should implement this function!")

	def c(self, y, phi):
		raise NotImplementedError("Subclasses should implement this function!")

	def mu(self, theta):
		raise NotImplementedError("Subclasses should implement this function!")

	def var_function(self, mu):
		raise NotImplementedError("Subclasses should implement this function!")

	def log_likelihood(self, y, theta, phi):
		return (y*theta - self.b(theta))/self.a(phi) + self.c(y,phi)

##### Implemented subclasses
###############################################################################################################
class GaussianNoise(Noise):

	def __init__(self):
		Noise.__init__(self)
		self.name = "gaussian"

	def a(self, phi):
		return phi

	def b(self,theta):
		return 0.5*(theta**2)

	def c(self, y, phi):
		return -0.5*((y**2)/phi + np.log(2.*np.pi*phi))

	def var_function(self, mu):
		return 1.

	def mu(self, theta):
		return theta

class PoissonNoise(Noise):

	def __init__(self, over_dispersion=1.):
		Noise.__init__(self)
		self.name = "poisson"
		self.sig2 = over_dispersion

	def a(self, phi):
		return 1.

	def b(self, theta):
		return np.exp(theta)

	def c(self, y, phi):
		return -np.log(gamma(y+1.))

	def var_function(self, mu):
		return self.sig2*mu

	def mu(self, theta):
		return np.exp(theta)

class BinomialNoise(Noise):

	def __init__(self):
		Noise.__init__(self)
		self.name = "binomial"

	def a(self, phi):
		return phi

	def b(self, theta):
		return np.log(1. + np.exp(theta))

	def c(self, y, phi):

		## Unraveling the binomial coefficient in
		## np.log(C(n, n*y)) with phi = 1./n
		term1 = np.log(gamma(1. + 1./phi))
		term2 = np.log(gamma(1. + y/phi))
		term3 = np.log(gamma(2. - y/phi))
		return term1 - term2 - term3

	def mu(self, theta):
		return np.exp(theta)/(1. + np.exp(theta))

	def var_function(self, mu):
		return mu*(1. - mu)


if __name__ == "__main__":


	noise_model = GaussianNoise()

	print(noise_model.a(5.))
	print(noise_model.log_likelihood(y=5., theta=4.5, phi=1.))
