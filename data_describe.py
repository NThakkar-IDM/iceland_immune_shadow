""" Script to make a sample figure, showing transformed traces, 
the data, etc."""
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
plt.rcParams["font.serif"] = "Garamond"
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
	number of months over which to sum. """

	## First step is to smooth the cases because, if incidence is
	## uniformly distributed within a given quarter, half of that incidence will,
	## on average contribute to the immunomodulation in the next quarter.
	transformed = 0.5*(cases + cases.shift(1))

	## Rolling sum for the additive transformation, which is fixed to
	## to be in units of months (unlike the gamma transformation.)
	transformed = transformed.rolling(time_delay).sum()

	return transformed.dropna()

if __name__ == "__main__":

	## Start year and end year
	start_year = "1904"
	end_year = "1974"
	timedelay_1 = 48
	timedelay_2 = 27

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

	## Slice everything to the appropriate years
	mortality = mortality.loc[start_year:end_year]

	## Extract the total deaths
	deaths = mortality.deaths

	## Convert to rates, 1 to 9 year olds for the deaths, 
	## all of the population for the annualized cases
	deaths = 1000.*deaths/(GetProcessedPopData(low=1,high=9).reindex(deaths.index))
	#deaths = 1000.*deaths/(GetProcessedPopData().reindex(deaths.index))


	## Construct measles per 1000 on a monthly basis by forward filling
	## the population
	population = population.resample("MS").interpolate().reindex(cases.index)
	case_density = (1000.*cases/population).loc[start_year:end_year]

	## Compute two sample transformations
	transformed = AdditiveTransform(cases,timedelay_1)
	transformed_2 = AdditiveTransform(cases,timedelay_2)

	## Convert them to rates
	transformed = (1000.*transformed/population.reindex(transformed.index)).loc[start_year:end_year]
	transformed_2 = (1000.*transformed_2/population.reindex(transformed_2.index)).loc[start_year:end_year]

	### Getting death rates during and after outbreaks
	###########################################################################################
	annualized_cases = AnnualizeCases(cases)
	non_outbreak_deaths = deaths[annualized_cases == 0.].reindex(deaths.index).rename("no_measles_deaths")
	outbreak_deaths = deaths[annualized_cases != 0.].reindex(deaths.index).rename("measles_deaths")

	## Find the death rate in years before measles introduction
	pre_measles_dr = pd.concat([non_outbreak_deaths,outbreak_deaths.shift(-1)],axis=1).dropna()["no_measles_deaths"]
	pre_measles_dr.rename("before",inplace=True)

	## And in the years after
	post_measles_dr = pd.concat([non_outbreak_deaths,outbreak_deaths.shift(1)],axis=1).dropna()["no_measles_deaths"]
	post_measles_dr.rename("after",inplace=True)

	### Make the plot
	###########################################################################################
	fig, axes = plt.subplots(figsize=(12,10))
	axes.grid(color="grey",ls="-",alpha=0.2)

	## Data and model performance	
	axes.plot(deaths.loc["1915":],color="k",marker="o")
	#axes.plot(pre_measles_dr,ls="None",marker="s",markersize=15,markeredgecolor="C0",
	#		  markerfacecolor="None",markeredgewidth=2,label="Before importation")
	#axes.plot(post_measles_dr,ls="None",marker="o",markersize=15,markeredgecolor="C3",
	#		  markerfacecolor="None",markeredgewidth=2,label="After importation")

	## Set the labels, etc.
	axes.set_ylabel("All-cause mortality (per 1k per year)",color="k")
	#axes.set_ylim((-10.,11.2))
	#axes.set_yticks([0.,4.,8.])

	## And add the case data.
	axes2 = axes.twinx()
	axes2.fill_between(transformed.index,[0.]*len(transformed),transformed,
					  color="grey",alpha=0.35)
	axes2.fill_between(transformed_2.index,[0.]*len(transformed_2),transformed_2,
					  color="C3",alpha=0.35)
	axes2.plot(case_density,lw=2,color="k")
	axes2.set(xlabel="Time",ylabel="Cases (per 1k per month)")
	axes2.set_ylim((0.,1.25*transformed.max()))
	axes2.set_xlim(("1913",None))

	## Reorder them
	axes.set_zorder(axes2.get_zorder()+1)
	axes.patch.set_visible(False)
		
	## Set up the legend
	axes2.plot([],color="k",marker="o",label="Mortality (1-9 year-olds)")
	axes2.plot([],color="k",lw=2,label="Measles incidence")
	axes2.plot([],color="C3",lw=5,label=str(timedelay_2)+" month immune-suppression prevalence")
	axes2.plot([],color="grey",lw=5,label=str(timedelay_1)+" month immune-suppression prevalence")
	#axes2.plot([],ls="None",marker="s",markersize=15,markeredgecolor="C0",markerfacecolor="None",
	#			  markeredgewidth=2,label="Before importation")
	#axes2.plot([],ls="None",marker="o",markersize=15,markeredgecolor="C3",markerfacecolor="None",
	#			  markeredgewidth=2,label="After importation")
	axes2.legend()
	plt.tight_layout()
	#plt.savefig("_plots\\iceland_overview.pdf")
	

	### Make a smaller plot for publication purposes
	###########################################################################################
	fig, axes = plt.subplots(figsize=(10,8))
	axes.grid(color="grey",ls="-",alpha=0.2)

	## Data and model performance	
	axes.plot(deaths,color="xkcd:red wine",lw=1.5)

	## Set the labels, etc.
	axes.set_ylabel("All-cause mortality (per 1k per year)",color="k")
	axes.set_ylim((-2.,9.))
	axes.set_yticks(np.arange(0.,9.,2))

	## And add the case data.
	axes2 = axes.twinx()
	axes2.plot(case_density,lw=1,color="k")
	axes2.fill_between(case_density.index,[0.]*len(case_density),case_density.values,
					   color="grey",alpha=0.25)
	axes2.set(xlabel="Time",ylabel="Cases (per 1k per month)")
	axes2.set_ylim((0.,None))
	axes2.set_xlim(("1900",None))
		
	## Set up the legend
	axes2.plot([],color="xkcd:red wine",lw=1.5,label="Mortality (1-9 year-olds)")
	axes2.plot([],color="k",label="Measles incidence")
	axes2.legend()
	plt.tight_layout()
	#plt.savefig("_plots\\iceland_overview_v1.pdf")

	### Make a smaller plot for publication purposes
	###########################################################################################
	fig, axes = plt.subplots(figsize=(18,8))
	axes.grid(color="grey",ls="-",alpha=0.2)

	## Data and model performance	
	axes.plot(deaths,color="xkcd:red wine",lw=1.,marker="o")

	## Set the labels, etc.
	axes.set_ylabel("All-cause mortality (per 1k per year)",color="k")
	axes.set_ylim((-2.,13.5))
	axes.set_yticks(np.arange(0.,13.,2))

	## And add the case data.
	axes2 = axes.twinx()
	axes2.plot(case_density,lw=1,color="k")
	axes2.fill_between(case_density.index,[0.]*len(case_density),case_density.values,
					   color="grey",alpha=0.25)
	axes2.set(xlabel="Time",ylabel="Cases (per 1k per month)")
	axes2.set_ylim((0.,None))
	axes2.set_xlim(("1900",None))
		
	## Set up the legend
	axes2.plot([],color="xkcd:red wine",lw=1.,marker="o",label="All-cause mortality (1-9 year-olds)")
	axes2.plot([],color="k",label="Measles incidence")
	axes2.legend()
	plt.tight_layout()
	#plt.savefig("_plots\\iceland_overview_v2.pdf")
	plt.show()