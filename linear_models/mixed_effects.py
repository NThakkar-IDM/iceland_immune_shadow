""" mixed_effects.py

A mixed effects model implementation, based on the documentation for lmer and on "Estimation in generalized lienar models with 
random effects" by Schall, 1991."""
from __future__ import print_function

import numpy as np
from six.moves import cPickle as pickle

## For interaction with the user
import warnings

## Link functions and noise models
from link_functions import *
from noise_models import *

## Some extras
from scipy.linalg import block_diag

#########################################################################################################
## Mixed effects class
#########################################################################################################
class MixedEffectsModel(object):

	def __init__(self, noise_model, link_function):

		## Store the model structure
		self.noise_model = noise_model
		self.link = link_function

		## Create an internal boolean to mark the model
		## as fit or not.
		self._fit = False


	def __call__(self, X, U, offset=0.):
		
		""" Evaluation on data set X, expected to be with shape=(num_data_points,num_features). Offset must be provided if not used 
		already, and U is a list of numpy arrays specifying random effects. """

		assert self._fit, "The model must be fit before it can be evaluated."

		## Not sure what the best thing to do here is...
		#random_effects = sum([np.dot(u, np.sqrt(v)*np.random.normal(loc=0., scale=1., size=(u.shape[1],))) for u, v in zip(U, self.var_i)])
		random_effects = np.dot(np.concatenate(U,axis=1), self.b)
		return self.link.inverse(np.dot(X,self.beta) + random_effects + offset)		


	def fit(self, X, U, Y, offset=None, tol=1.e-5, max_its=1000):

		""" Use penalized iterated reweighted least squares to fit the mixed effects model.

		Y: observation, numpy array with shape = (num_data_points, 1)
		X: fixed effects, numpy array with shape (num_data_points, num_fixed_effects)
		U: list of random effects, list of numpy arrays, length = num_random_effects, each array has shape =
			(num_data_points, num_groups_in_random_effect)
		offset: numpy array with shape = (num_data_points, 1)
		tol: 1 norm difference between subsequent estimates of beta
		max_its: maximum least squares iterations before giving up. """

		## Check for shape consistency
		assert Y.shape[0] == X.shape[0], "The number of data points between X and Y is inconsistent."
		for u in U:
			assert Y.shape[0] == u.shape[0], "The number of data points between U and Y is inconsistent."

		## Store the structural pieces of the model, the number of data points
		## number of random and fixed effects, and number of groups for each random
		## effect.
		self.num_data_points = Y.shape[0]
		try:
			self.num_fixed_effects = X.shape[1]
		except: 
			X = X.reshape((self.num_data_points,1))
			self.num_fixed_effects = 1
		self.num_random_effects = len(U)
		self.num_groups = np.array([u.shape[1] for u in U])
		assert self.num_data_points >= self.num_fixed_effects, "There must be at least as many data points as fixed effects."

		## Create the appropriate offset vector if needed.
		if offset is None:
			offset = np.zeros(Y.shape)
		else:
			assert Y.shape == offset.shape, "The offset must be a vector with length equal to the number of data points."

		## Initial guesses for the fixed effects. Here we assume
		## eta = link(mu), and we start by guessing mu is just the
		## data (regularized so link(mu) is defined).
		self.beta = np.zeros((self.num_fixed_effects,1))
		mu = self.link.regularize(Y)
		eta = self.link(mu) - offset
		
		## Create the initial weights based on the 
		## link function derivative and the variance of the
		## noise model.
		deta_dmu = self.link.derivative(mu)
		weights = 1./((deta_dmu**2)*self.noise_model.var_function(mu))

		## Initialize the variances, var_i of the 
		## random effects, and the associated covariance
		## matrix. We also create a "full" random effects arrray,
		## just U from above concantenated
		q = int(np.sum(self.num_groups))
		self.var_i = np.var(Y)*np.ones((self.num_random_effects,))
		D = np.eye(q)
		U_full = np.concatenate(U,axis=1)

		## Penalized iterated reweighted least squares (PIRLS) loop,
		## regressing the model with linearized g(mu).
		converged = False
		for i in range(max_its):

			## Linearized output
			z = eta + (Y - mu)*deta_dmu

			## Solve the extended Normal equations. These are the least
			## squares equations with weights and with zeros in the output
			## in order to incorporate an L2 penalty with the random effects
			## correlation structure.
			estimate, CCT_inv = ExtendedNormalEquations(z,X,U_full,weights,D)

			## Unpack the estimate
			beta1 = estimate[:self.num_fixed_effects]
			b1 = estimate[self.num_fixed_effects:]

			## Check for convergence
			delta_beta = np.linalg.norm(beta1-self.beta, ord=1)
			if delta_beta < tol:
				converged = True

			## Update mu, eta, weights
			self.beta = beta1
			self.b = b1
			eta = np.dot(X, self.beta)
			mu = self.link.inverse(eta + offset)
			deta_dmu = self.link.derivative(mu)
			weights = 1./((deta_dmu**2)*self.noise_model.var_function(mu))

			## And update var_i and D for the random effects
			T = np.split(np.diag(CCT_inv[-q:,-q:]), np.cumsum(self.num_groups[:-1]))
			b_i = np.split(self.b, np.cumsum(self.num_groups[:-1]))
			v_i = np.array(map(lambda x: x.sum(), T))/self.var_i
			self.var_i = np.array([np.sum(b**2) for b in b_i])/(self.num_groups - v_i)
			D = block_diag(*[v*np.eye(qi) for v,qi in zip(self.var_i,self.num_groups)])

			## All done if converged, otherwise 
			## we try again.
			if converged:
				break

		## Raise a warning if the calculation failed
		if not converged:
			warnings.warn("IRLS failed to converge after %i iterations!" % max_its)

		## Mark the model as fit
		self._fit = True

		## Compute the variance, and beta covariance matrix
		self.resid = Y - self.__call__(X,U,offset)
		self.dof = self.num_data_points - self.num_fixed_effects
		self.dof_effective = self.dof - np.sum(self.num_groups - v_i)
		self.var = (self.resid**2).sum(axis=0)/self.dof_effective
		self.beta_var = self.var*CCT_inv[:self.num_fixed_effects,:self.num_fixed_effects]

		## Store the data used to fit
		self.X_fit = X
		self.Y_fit = Y
		self.offset_fit = offset
		self.U_fit = U

	def summary(self, fixed_effects=None, random_effects=None):

		""" Function which summarizes the model. """

		## Header
		print("\nA linear mixed effects model with...")
		print("Noise model: "+self.noise_model.name.title())
		print("Link function: "+self.link.name.title()+"\n")

		## Has the model been fit? If so...
		if self._fit:

			print("Fixed effects:")
			if fixed_effects is None:
				fixed_effects = [str(i) for i in range(len(self.beta))]
			std = np.sqrt(np.diag(self.beta_var))
			for f, b, s in zip(fixed_effects, self.beta, std):
				print("Parameter "+f+" = %0.5f +\- %0.5f" % (b,s))

			print("\nRandom effects:")
			if random_effects is None:
				random_effects = [str(i) for i in range(len(self.var_i))]
			std = np.sqrt(self.var_i)
			for r, s in zip(random_effects, std):
				print("Random effect "+r+": standard deviation = %0.5f" % s)

		else:
			print("The model has not yet been fit to data.")


#########################################################################################################
## WLS implementation
#########################################################################################################
def ExtendedNormalEquations(Y,X,U,weights,D):

	""" Solve the weighted least squares problem with regularizing extention based on 
	U and coveraiance D."""

	## Get the dimensions
	num_data_points = X.shape[0]
	num_fixed_effects = X.shape[1]
	num_random_effects = U.shape[1]

	## Set up the weight matrix
	W = np.diag(weights.reshape(-1))

	## Construct the required normal equation matrix.
	CCT = np.zeros((num_fixed_effects + num_random_effects, num_fixed_effects + num_random_effects))
	CCT[:num_fixed_effects,:num_fixed_effects] = np.dot(X.T,np.dot(W,X))
	CCT[num_fixed_effects:,:num_fixed_effects] = np.dot(U.T,np.dot(W,X))
	CCT[:num_fixed_effects,num_fixed_effects:] = np.dot(X.T,np.dot(W,U))
	CCT[num_fixed_effects:,num_fixed_effects:] = np.diag(1./np.diag(D))

	## Compute the matrix inversion
	CCT_inv = np.linalg.inv(CCT)

	## Compute the resulting estimate
	rhs = np.dot(W,Y)
	rhs = np.concatenate([np.dot(X.T,rhs), np.dot(U.T,rhs)],axis=0)
	estimate = np.dot(CCT_inv, rhs).reshape((num_fixed_effects+num_random_effects,))
 	return estimate, CCT_inv




















