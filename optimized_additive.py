""" optimized_additive.py

A script to solve the 1d optimization problem on the additive regression."""
from __future__ import print_function
import sys
sys.path.append("..\\")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Regression stuff
from linear_models.glm import *
from linear_models.data_tools import *

## Get the data processing function
from data_process import *

## Custom global matplotlib parameters
## see http://matplotlib.org/users/customizing.html for details.
plt.rcParams["font.size"] = 28.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams["xtick.labelsize"] = "medium"
plt.rcParams["ytick.labelsize"] = "medium"
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["axes.formatter.use_mathtext"] = True

#################################################################################################
### Basic transformations (with no age structure)
#################################################################################################
def AdditiveTransform(cases,time_delay):

	""" Perform the additive transformation described in the Mina SI. time_delay is the 
	number of quarters over which to sum. """

	if time_delay == 0:
		return cases

	## First step is to smooth the cases because, if incidence is
	## uniformly distributed within a given quarter, half of that incidence will,
	## on average contribute to the immunomodulation in the next quarter.
	transformed = 0.5*(cases + cases.shift(1))

	## Rolling sum for the additive transformation, which is fixed to
	## to be in units of quarters (unlike the gamma transformation.)
	transformed = transformed.rolling(time_delay).sum()
	return transformed.dropna()


#################################################################################################
### Optimization methods
#################################################################################################
def AdditiveOptimization(cases,deaths,pop,time_delays,drop_zeros=False):

	""" Function to loop over additive transformed regressions and compute R^2
	values for brute force optimization"""

	## Get the year bounds
	start_year = deaths.index[0]
	end_year = deaths.index[-1]

	## Initialize the line fitting model
	model = GLM(noise_model=GaussianNoise(),link_function=IdentityLink())

	## And get the correctly sampled pop
	re_pop = pop.resample("MS").interpolate().reindex(cases.index)

	## Finally, loop over delays, perform the fit,
	## and compute R2.
	R2s = np.zeros((len(time_delays,)))
	for i, time_delay in enumerate(time_delays):

		## Compute the transform
		transformed = AdditiveTransform(cases,time_delay)

		## Annualize the transformed series and convert to a 
		## rate.
		transformed = (AnnualizeCases(transformed)/12.).loc[start_year:end_year]
		transformed = 1000.*transformed/re_pop.reindex(transformed.index)

		## Drop zeros if needed
		if drop_zeros:
			transformed = transformed.replace(0.,np.nan).dropna()

		## Set up features and responses
		X = np.array([np.ones((len(transformed),)),transformed.values]).T
		Y = deaths.reindex(transformed.index).values

		## Fit the model
		model.fit(X,Y)

		## Compute R2
		R2s[i] = 1. - np.sum(model.resid**2)/np.sum((Y - Y.mean())**2)

	return R2s

if __name__ == "__main__":

	## Start year and end year
	start_year = "1945"
	end_year = "1974"

	## Decide if you want to drop years with no
	## immune suppression
	drop_zeros = False

	## What age groups do you want to considers?
	age_groups = ["1st year","1-4 years","5-9 years","10-14 years","15-19 years","20-24 years",
				  "25-29 years","30-34 years","35-39 years","40-44 years","45-49 years","50-54 years",
				  "55-59 years","60-64 years","65-69 years","70-74 years","75-79 years","80-84 years",
				  "85-89 years","90-94 years","95 years and older"]


	### Getting the data
	###########################################################################################
	cases = GetProcessedMeaslesData()
	mortality = GetProcessedMortalityData(age_groups=["1-4 years","5-9 years"]).groupby("year").apply(UpsampleMortality)
	#mortality = GetProcessedMortalityData(age_groups=age_groups).groupby("year").apply(UpsampleMortality)
	population = GetProcessedPopData()

	## Slice mortality to the appropriate years
	mortality = mortality.loc[start_year:end_year]

	## Extract the total deaths
	deaths = mortality.deaths

	## Convert to rates, 1 to 9 year olds for the deaths, 
	## all of the population for the annualized cases
	deaths = 1000.*deaths/(GetProcessedPopData(low=1,high=9).reindex(deaths.index))
	#deaths = 1000.*deaths/(GetProcessedPopData().reindex(deaths.index))

	### Test the optimization procedure on shuffled 
	### death data.
	_shuffle = False
	if _shuffle:

		## Copy deaths
		deaths_raw = deaths.copy()

		## Death trend
		time = np.arange(float(start_year),float(end_year)+1)
		X = np.array([np.ones((len(time),)),time]).T
		Y = deaths.values
		model = GLM(noise_model=GaussianNoise(),link_function=IdentityLink())
		model.fit(X,Y)

		## Detrend
		deaths = deaths - model(X)

		## Shuffle
		deaths = pd.Series(np.random.permutation(deaths.values),index=deaths.index)

		## Retrend
		deaths = deaths + model(X)


	### R^2 optimization
	###########################################################################################
	time_delays = np.arange(0.,120.,dtype=int)
	R2s = AdditiveOptimization(cases,deaths,population,time_delays,drop_zeros=drop_zeros)

	## Plot the R2s
	fig, axes = plt.subplots(figsize=(8,8))
	i_max = np.argmax(R2s)
	axes.axvline(time_delays[i_max],color="grey",ls="dashed")
	axes.plot(time_delays,R2s,color="k",lw=1,marker="o")
	axes.plot(time_delays[i_max],R2s[i_max],marker="o",ls="None",
			  markersize=15,markeredgecolor="C3",markerfacecolor="None")

	## Finish up
	axes.set(xlabel="Duration (months)",ylabel=r"R$^2$")#,ylim=(0.,0.9))
	axes.grid(color="grey",ls="-",alpha=0.25)
	axes.set_xticks([0.,50.,100.])
	plt.tight_layout()
	#plt.savefig("_plots\\iceland_optimization.pdf")


	### See the optimal transform and associated regression
	###########################################################################################
	## Construct measles per 1000 on a monthly basis by forward filling
	## the population
	population = population.resample("MS").interpolate().reindex(cases.index)
	case_density = (1000.*cases/population).loc[start_year:end_year]
	transformed = AdditiveTransform(cases,time_delays[i_max])
	transformed_density = (1000.*transformed/population.reindex(transformed.index)).loc[start_year:end_year]

	## Plot the immuno-suppresion curve and measles
	## incidence.
	fig, axes = plt.subplots(figsize=(10,8))
	axes.plot([],color="grey",lw=5,label=str(time_delays[i_max])+" month immune-suppresion prevalence")
	axes.fill_between(transformed_density.index,[0.]*len(transformed_density),transformed_density,
					  color="grey",alpha=0.25)
	axes.plot(case_density,color="k",label="Measles incidence")
	axes.set(ylabel="Cases per 1000")
	axes.set_ylim((0.,1.25*transformed_density.max()))
	axes.legend(loc=1)
	plt.tight_layout()
	
	## Annualize the transformed series and
	## convert to an annual rate
	transformed = AnnualizeCases(transformed)/12.
	transformed = (1000.*transformed/population.reindex(transformed.index)).loc[start_year:end_year]

	## Make sure the deaths align
	if drop_zeros:
		transformed = transformed.replace(0.,np.nan).dropna()
	deaths = deaths.reindex(transformed.index)

	## Create the model
	model = GLM(noise_model=GaussianNoise(),link_function=IdentityLink())

	## Prepare the data for fitting
	X = np.array([np.ones((len(transformed),)),transformed.values]).T
	Y = deaths.values
	model.fit(X,Y)

	## Resample the lines
	test_x = np.array([np.ones((500,)),np.linspace(0.,1.2*transformed.max(),500)]).T
	samples = model.resample(test_x,5000)
	low = np.percentile(samples,2.5,axis=0)
	high = np.percentile(samples,97.5,axis=0)
	mid = samples.mean(axis=0)

	## Compute the R^2 value
	SStot = np.sum((Y - Y.mean())**2)
	SSres = np.sum(model.resid**2)
	R2 = 1. - SSres/SStot

	## Plot it
	fig, axes = plt.subplots(figsize=(10,8))

	## First the data, with colors based on the year
	scatter = axes.scatter(transformed,deaths,marker="o",
						   c=deaths.index.year.tolist(),cmap="copper",
						   s=13**2)
	plt.colorbar(scatter,ticks=np.arange(int(start_year),int(end_year)+1,3))

	## Then the fit
	axes.fill_between(test_x[:,1],low,high,color="grey",alpha=0.2)
	axes.plot(test_x[:,1],mid,color="k")

	## and the details
	axes.set(xlabel="Cases (per 1k)",ylabel="Deaths (per 1k)")
	axes.text(0.2,0.9,r"R$^2$ = "+str(R2)[:4],fontsize=36,
			  horizontalalignment="center",verticalalignment="center",
			  transform=axes.transAxes)
	axes.set_xlim((0.9*transformed.min(),transformed.max()*1.1))
	axes.set_xticks(np.arange(10,80,10))
	plt.tight_layout()
	#plt.savefig("_plots\\optimal_transform_reg.pdf")

	if _shuffle:
		fig, axes = plt.subplots(figsize=(12,8))
		axes.plot(deaths_raw,marker="o",color="k",label="Actual deathrate")
		axes.plot(deaths,marker="o",color="C3",label="Shuffled deathrate")
		axes.legend()
		axes.set(ylabel="Deaths per 1k per year")


	plt.show()

