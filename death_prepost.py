""" This script is a different analysis, computing the death rate before and after
outbreaks to see if there's a statistically significant difference between them. """
from __future__ import print_function
import sys
sys.path.append("..\\")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Custom global matplotlib parameters
## see http://matplotlib.org/users/customizing.html for details.
plt.rcParams["font.size"] = 24.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["axes.formatter.use_mathtext"] = True

## Regression stuff
from linear_models.glm import *
from linear_models.data_tools import *

## Get the data processing function
from data_process import *


if __name__ == "__main__":

	## Start year and end year
	start_year = "1904"
	end_year = "1974"

	## What age groups do you want to considers?
	age_groups = ["1st year","1-4 years","5-9 years","10-14 years","15-19 years","20-24 years",
				  "25-29 years","30-34 years","35-39 years","40-44 years","45-49 years","50-54 years",
				  "55-59 years","60-64 years","65-69 years","70-74 years","75-79 years","80-84 years",
				  "85-89 years","90-94 years","95 years and older"]

	### Getting the data
	###########################################################################################
	cases = GetProcessedMeaslesData(fname="measles.csv")
	mortality = GetProcessedMortalityData(age_groups=["1-4 years","5-9 years"]).groupby("year").apply(UpsampleMortality)
	#mortality = GetProcessedMortalityData(age_groups=age_groups).groupby("year").apply(UpsampleMortality)
	population = GetProcessedPopData()

	## Slice everything to the appropriate years
	mortality = mortality.loc[start_year:end_year]

	## Extract the total deaths
	deaths = mortality.deaths

	## Convert to rates, 1 to 9 year olds for the deaths, 
	## all of the population for the annualized cases
	deaths = 1000.*deaths/(GetProcessedPopData(low=1,high=9).reindex(deaths.index))
	#deaths = 1000.*deaths/(GetProcessedPopData().reindex(deaths.index))

	_detrend = True
	if _detrend:
		time = np.arange(float(start_year),float(end_year)+1)		
		X = np.array([np.ones((len(time),)),time]).T
		Y = deaths.values
		model = GLM(noise_model=GaussianNoise(),link_function=IdentityLink())
		model.fit(X,Y)
		deaths = deaths - model(X)
		#deaths = deaths - deaths.min()

	### Getting death rates during and after outbreaks
	###########################################################################################
	annualized_cases = AnnualizeCases(cases)
	non_outbreak_deaths = deaths[annualized_cases == 0.].reindex(deaths.index).rename("no_measles_deaths")
	outbreak_deaths = deaths[annualized_cases != 0.].reindex(deaths.index).rename("measles_deaths")

	## Find the death rate in years before measles introduction
	pre_measles_dr = pd.concat([non_outbreak_deaths,outbreak_deaths.shift(-1)],axis=1).dropna()["no_measles_deaths"]
	pre_measles_dr.rename("before",inplace=True)

	## Compute the average time between outbreak years
	import_years = np.array(pre_measles_dr.index.year.tolist())
	post1945 = import_years[import_years >= 1945]
	print("Average (post 1945) time between outbreaks = {}".format((post1945[1:]-post1945[:-1]).mean()))

	
	## And in the years after
	post_measles_dr = pd.concat([non_outbreak_deaths,outbreak_deaths.shift(1)],axis=1).dropna()["no_measles_deaths"]
	post_measles_dr.rename("after",inplace=True)

	## Concatenate them
	pre_post = pd.concat([pre_measles_dr,post_measles_dr],axis=1)

	## And compute the average difference
	diff = (pre_post.after.shift(-1) - pre_post.before).dropna()
	#diff = ((pre_post.after.shift(-1) - pre_post.before)/pre_post.before).dropna()

	## Compute summary statistics of diff
	print("Average pre-post difference = {}".format(diff.mean()))
	print("Standard deviation = {}".format(diff.std()))
	print("This is based on {} measles importations.".format(len(diff)))

	## Plot a summary
	fig, axes = plt.subplots(figsize=(12,10))
	axes.grid(color="grey",alpha=0.2)
	axes.plot(deaths,marker="o",color="k",label="Death rate")
	axes.plot(pre_measles_dr,ls="None",marker="s",markersize=12,markeredgecolor="C0",
			  markerfacecolor="None",label="Before importation")
	axes.plot(post_measles_dr,ls="None",marker="o",markersize=12,markeredgecolor="C3",
			  markerfacecolor="None",label="After importation")
	axes.plot([],ls="dashed",c="k",label="Measles cases")
	if _detrend:
		axes.set(ylabel="Detrended deaths per 1k per year",ylim=(-15.,7.))
	else:
		axes.set(ylabel="Death rate")#,ylim=(0.,14.))
	axes2 = axes.twinx()
	axes2.plot(annualized_cases,ls="dashed",c="k")
	axes2.set(ylabel="Measles cases per year",ylim=(0.,16000))
	axes.legend()
	plt.tight_layout()
	#plt.savefig("_plots\\iceland_pre_post.pdf")
	plt.show()

