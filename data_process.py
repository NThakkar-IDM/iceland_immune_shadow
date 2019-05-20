""" data_process.py
Processing the measles and mortality data from iceland."""
from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Custom global matplotlib parameters
## see http://matplotlib.org/users/customizing.html for details.
plt.rcParams["font.size"] = 20.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

################### Mortality data
##############################################################################################
def GetRawMortalityData(root="_data\\",fname="deaths.csv"):

	""" Read the csv file from 
	https://www.statice.is/statistics/population/births-and-deaths/deaths/ """

	## Get the data frame
	df = pd.read_csv(root+fname,skiprows=2,header=0)

	return df

def GetProcessedMortalityData(root="_data\\",fname="deaths.csv",age_groups=None):

	""" Use the above function to get the data and run some basic processing. """

	## Get the raw data
	df = GetRawMortalityData(root,fname)

	## Keep only the childhood deaths
	if age_groups is not None:
		df = df[df.Age.isin(age_groups)]

	## Keep only the total values
	df = df[["Year","Age"] + [c for c in df.columns if c.find("Total") != -1]]

	## Rename the columns
	df.columns = ["year","age","deaths","deaths_per_1k"]

	## Conver the year to a date time
	df.year = pd.to_datetime({"year":df.year,"month":1,"day":1})

	## Slice out years pre 1900
	df = df[df.year >= "1900-01-01"]

	## Reset the index as a multiindex
	df = df.set_index(["year","age"])

	return df

def UpsampleMortality(x):

	""" Function to be used on df.groupby("year").apply calls, which correctly
	handles the total deaths with a sum and the rates with a mean. """

	return pd.Series([x.deaths.sum(), x.deaths_per_1k.mean()],index=["deaths","deaths_per_1k"])

################### Population data
##############################################################################################
def GetRawPopData(root="_data\\",fname="population.csv"):

	""" Read the csv file from 
	https://statice.is/statistics/population/inhabitants/overview/ """

	## Get the data frame
	df = pd.read_csv(root+fname,skiprows=2,header=0)

	## Convert the age column to an integer column
	df["Age"] = df.Age.str.replace("Under 1 year", "0 year")
	df["Age"] = df.Age.str.extract("(\d+)",expand=False).astype(int)

	## Change it to a multiindex df
	df = df.set_index(["Sex","Age"])

	## And reshape it to have a third level based on the columns
	series = {}
	for t, sf in df.groupby(["Sex","Age"]):
		series[t] = sf.loc[t]
	df = pd.concat(series.values(),keys=series.keys())
	df.index.rename(["sex","age","year"],inplace=True)

	return df.sort_index(level=1)

def GetProcessedPopData(root="_data\\",fname="population.csv",low=0,high=2100):

	""" Use the above function to get the data and run some basic processing. """

	## Get the data frame
	df = GetRawPopData(root,fname)
	
	## Group by sex to get totals
	df = df.groupby(["age","year"]).sum()

	## Keep specified ages and then
	## sum over them.
	df = df.loc[low:high,:]
	df = df.groupby("year").sum()

	## Finally, convert the index to datetime and return
	## the relevent years
	df.index = pd.to_datetime({"year":df.index,"month":1,"day":1})
	df = df.loc["1900-01-01":]

	return df

################### Measles data
##############################################################################################
def GetRawMeaslesData(root="_data\\",fname="measles.csv"):

	""" Get the raw measles data for all of Iceland, transcribed from the appendix of
	Spatial Diffusion: An Historical Geography of Epidemics in an Island Community by 
	Cliff, Haggett, Ord, and Versey."""

	df = pd.read_csv(root+fname,skiprows=2,header=0)
	return df

def GetProcessedMeaslesData(root="_data\\",fname="measles.csv"):

	""" Do some basic processing on the output from the function above. """

	## Get the raw data
	df = GetRawMeaslesData(root,fname)

	## Make the time column a datetime column and
	## set it as the index
	df["time"] = pd.to_datetime(df.time)
	df = pd.Series(df.cases.values,index=df.time,name="cases")

	## Fill NaNs with zeros. Those are times in record
	## where no measles cases were reported.
	df.fillna(0., inplace=True)

	return df

def AnnualizeCases(cases):
	annual_cases = cases.groupby(lambda t: t.year).sum()
	annual_cases.index = pd.to_datetime({"year":annual_cases.index,"month":1,"day":1})
	return annual_cases

if __name__ == "__main__":

	## Get the data
	cases = GetProcessedMeaslesData()
	mortality = GetProcessedMortalityData(age_groups=["1-4 years","5-9 years"]).groupby("year").apply(UpsampleMortality)
	population = GetProcessedPopData()

	## Extract the total deaths
	deaths = mortality.deaths

	## Create an annualized case trace
	annual_cases = AnnualizeCases(cases)
	annual_cases = 1000.*annual_cases/(population.reindex(annual_cases.index))

	## Construct measles per 1000 on a monthly basis by forward filling
	## the population
	population = population.resample("MS").interpolate().reindex(cases.index).rename("total_pop")

	## Convert to rates
	deaths = (1000.*deaths/GetProcessedPopData(low=1,high=9)).rename("deaths_per_1k")
	cases = (1000.*cases/population).rename("cases_per_1k")
	
	## Plot some data
	fig, axes = plt.subplots(figsize=(12,8))
	axes2 = axes.twinx()
	axes2.plot(deaths,c="C3",ls="dashed",lw=2)
	axes.plot(cases,color="k",label="Measles cases per 1000")
	axes.fill_between(cases.index,[0.]*len(cases),cases.values,color="grey",alpha=0.75)
	axes.plot([],c="C3",ls="dashed",lw=2,label="All cause mortality, age 1-9 years")
	axes.grid(color="grey",ls="-",alpha=0.2)
	axes2.set_ylabel("Deaths (age 1-9) per 1000 per year")
	axes.set(ylabel="Measles cases per 1k per month",xlabel="Year",ylim=(0.,30.),xlim=("1903","1976"))
	axes.legend(loc=1)
	plt.tight_layout()

	## Plot the time series
	pop_1_9 = GetProcessedPopData(low=1,high=9)
	fig, axes = plt.subplots(figsize=(12,8))
	axes2 = axes.twinx()
	axes.plot(cases,color="k",label="Measles cases per 1000")
	axes.fill_between(cases.index,[0.]*len(cases),cases.values,color="grey",alpha=0.75)
	axes.plot([],c="C0",lw=2,label="Total population")
	axes.plot([],c="C1",lw=2,ls="dashed",label="1-9 year olds")
	axes2.plot(population,c="C0",lw=2)
	axes2.plot(pop_1_9,c="C1",lw=2,ls="dashed")
	axes.grid(color="grey",ls="-",alpha=0.2)
	axes.set(xlabel="Year",ylabel="Measles cases per 1k per month")
	axes2.set_ylabel("Population")
	axes.set_ylim((0.,30.))
	axes.legend(loc=2)
	plt.tight_layout()
	plt.show()



