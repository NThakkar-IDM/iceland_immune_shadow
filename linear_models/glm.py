""" glm.py

This file contains the basic GLM model class, what should be called by the user. The model class
takes as input a link function (link_functions.py) and a noise model (noise_models.py)"""
from __future__ import print_function

import numpy as np
from six.moves import cPickle as pickle

## For interaction with the user
import warnings

## Link functions and noise models
from linear_models.link_functions import *
from linear_models.noise_models import *

#########################################################################################################
## Basic GLM class
#########################################################################################################
class GLM(object):

	def __init__(self, noise_model, link_function):

		## Store the GLM structure.
		self.noise_model = noise_model
		self.link = link_function

		## Initialize parameters
		#self.beta = 0.

		## Create an internal boolean to mark the model as
		## fit or not.
		self._fit = False

	def fit(self, X, Y, offset=None, tol=1.e-5, max_its=1000):

		"""Use iterated, reweighted least squares to fit the GLM to realization of the output
		Y (shape = (num_data_points, 1)) with covariates X (shape = (num_data_points, num_features).

		tol is the 1 norm difference between subsquent beta estimates in order to break
		max_its is the max number of iterations."""

		## Check for some consistency between shapes of X and Y
		assert Y.shape[0] == X.shape[0], "Number of data points between X and Y is inconsistent. "+ \
										 "They must have the same first dimension."
		assert len(Y.shape) == 1, "Only scalar response variables are supported!"

		## Initialize the algorithm, first by getting
		## the appropriate shapes, checking to make sure you have
		## more data than parameters.
		self.num_data_points = X.shape[0]
		try:
			self.num_features = X.shape[1]
		except:
			X = X.reshape((self.num_data_points,1))
			self.num_features = 1
		assert self.num_data_points >= self.num_features, "You need at least as many data points as features."

		## Check the offset vector if provided, otherwise 
		## it's just set to zero
		if offset is None:
			offset = np.zeros(Y.shape)
		else:
			assert Y.shape == offset.shape, "Offset must be vector with length equal to the number of data points."

		## Inital guesses for the algorithm. Here we assume 
		## eta = link(mu), and we start by guessing that mu is simply
		## the data, as per McCullagh's and Nelder's suggestion.
		self.beta = np.zeros((self.num_features,1))
		mu = self.link.regularize(Y)
		eta = self.link(mu) - offset
		deta_dmu = self.link.derivative(mu)
		weights = 1./((deta_dmu**2)*(self.noise_model.var_function(mu)))

		## Iterated reweighted least squares loop.
		## This is main loop for the algorithm, doing WLS (implemented below)
		## on successive linearizations of g(mu).
		for i in range(max_its):

			## Linearized output
			z = eta + (Y - mu)*deta_dmu

			## Compute regression via WLS
			beta1, beta_var, resid = WeightedLeastSquares(X,z,weights=weights)

			## Check for convergence based on the 1 norm
			## difference.
			delta_beta = np.linalg.norm(beta1 - self.beta, ord=1)
			if delta_beta < tol:
				self.beta = beta1
				break

			## If not yet converged, update mu and eta.
			self.beta = beta1
			eta = np.dot(X,self.beta)
			mu = self.link.inverse(eta + offset)
			deta_dmu = self.link.derivative(mu)
			weights = 1./((deta_dmu**2)*(self.noise_model.var_function(mu)))

		if i == (max_its-1):
			warnings.warn("IRLS failed to converge after %i iterations!" % max_its)

		## Store the final variance and mark the model
		## as fit.
		self._fit = True
		self.beta_var = beta_var

		## And compute the residual and
		## degrees of freedom
		Yhat = self.__call__(X,offset)
		self.resid = Y - Yhat
		self.dof = self.num_data_points - self.num_features

		## Storing the data the model was fit on.
		self.X_fit = X
		self.Y_fit = Y
		self.offset_fit = offset

	def __call__(self, X, offset=0.):

		""" Evaluation on data set X, expected to be with shape=(num_data_points,num_features). Offset must be 
		provided if not used already."""

		assert self._fit, "The model must be fit before it can be evaluated."

		return self.link.inverse(np.dot(X,self.beta) + offset)

	def resample(self, X, num_samples, offset=0.):

		""" Evaluation of the resampled output (sampling based on self.beta_var from fitting)."""

		assert self._fit, "The model must be fit before it can be resampled."

		## Resample the parameters.
		betas = np.random.multivariate_normal(mean=self.beta,cov=self.beta_var,size=(num_samples,))

		## Compute the samples of Yhat
		samples = self.link.inverse(np.dot(X,betas.T) + offset)
		return samples.T

	def save(self,fname):
		pickle.dump(self, open(fname,"wb"), protocol=pickle.HIGHEST_PROTOCOL)

	def summary(self, features=None):

		""" Function to summarize the parameters. """

		## Construct a summary header
		if self.link.name.startswith(("a","e","i","o","u")):
			header = "\nThis GLM has a "+self.noise_model.name.title()+" noise model with an "+self.link.name+" link function."
		else:
			header = "\nThis GLM has a "+self.noise_model.name.title()+" noise model with a "+self.link.name+" link function."

		## Print the summary.
		print(header)
		if self._fit:
			if features is None:
				features = [str(i) for i in range(len(self.beta))]
			std = np.sqrt(np.diag(self.beta_var))
			for f,b,s in zip(features,self.beta,std):
				if abs(b) <= 2.*s:
					print("Parameter "+f+" = %0.5f +\- %0.5f ****" % (b,s))
				else:
					print("Parameter "+f+" = %0.5f +\- %0.5f" % (b,s))
		else:
			print("The model has not yet been fit to data.")

def LoadGLM(fname):
	glm = pickle.load(open(fname,"rb"))
	return glm

#########################################################################################################
## WLS implementation
#########################################################################################################
def WeightedLeastSquares(X,Y,weights=None,verbose=False,standardize=False):

	""" Weighted LS, reduces to OLS when weights is None. This implementation computes
	the estimator and covariance matrix based on sample variance.
	NB: x is assumed to be an array with shape = (num_data points, num_features). """

	## Get the dimensions
	num_data_points = X.shape[0]
	try:
		num_features = X.shape[1]
	except:
		num_features = 1
		X = X.reshape((num_data_points,1))

	## Initialize weight matrix
	if weights is None:
		W = np.eye(num_data_points)
	else:
		W = np.diag(weights.reshape(-1))

	## Standardize the inputs and outputs to help with
	## stability of the matrix inversion. This is needed because
	## cumulative cases and births both get very large.
	if standardize:
		muY = Y.mean()
		sigY = Y.std()
		muX = X.mean(axis=0)
		sigX = X.std(axis=0)
		X = (X-muX)/sigX
		Y = (Y-muY)/sigY

	## Compute the required matrix inversion
	## i.e. inv(x.T*w*x), which comes from minimizing
	## the residual sum of squares (RSS) and solving for
	## the optimum coefficients. See eq. 3.6 in EST
	xTwx_inv = np.linalg.inv(np.dot(X.T,np.dot(W,X)))

	## Now use that matrix to compute the optimum coefficients
	## and their uncertainty.
	beta_hat = np.dot(xTwx_inv,np.dot(X.T,np.dot(W,Y)))

	## Compute the estimated variance in the data points
	residual = Y - np.dot(X,beta_hat)
	RSS = (residual)**2
	var = RSS.sum(axis=0)/(num_data_points - num_features)

	## Then the uncertainty (covariance matrix) is simply a 
	## reapplication of the inv(x.T*x):
	beta_var = var*xTwx_inv

	## Reshape the outputs
	beta_hat = beta_hat.reshape((num_features,))

	## Rescale back to old values
	if standardize:
		X = sigX*X + muX
		Y = sigY*Y + muY
		beta_hat = beta_hat*(sigY/sigX)
		sig = np.diag(sigY/sigX)
		beta_var = np.dot(sig,np.dot(beta_var,sig))
		residual = sigY*residual + muY - np.dot(muX,beta_hat)

	## Print summary if needed
	if verbose:
		for i in range(num_features):
			output = (i,beta_hat[i],2.*np.sqrt(beta_var[i,i]))
			print("Feature %i: coeff = %.4f +/- %.3f." % output)

	return beta_hat, beta_var, residual