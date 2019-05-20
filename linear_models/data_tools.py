""" data_tools.py

This file contains some helpful functions for moving from data frames to the 
feature matrices needed by the GLM class. The general goals of these functions are to
move from a pandas data frame with mixed columns (some numeric, some catagorical, etc.),
to the appropriate feature matrix. This is really similar to what's used in the neural network code, but
cleaned up for ease of use."""
from __future__ import print_function

import numpy as np 
import pandas as pd

def ToFeatureArray(raw_df, predictors, cov_types={}, intercept=False, transform=lambda x: x):

	""" Over arching process function, which takes a data frame and list of predictors which need to be
	converted. cov_type is a dictionary with predictor names as keys and either "factor" or "continuous" as
	values. If a predictor isn't in the dictionary, it's treatment is inferred based on dtype (floats and ints are
	treated as continuous, everything else is catagorical).

	If intercept is true, a column of ones is added to the begining of the output.
	transform is provided so that the factor type covariates have value 0 and 1 in link function space. That is,
	a log link sets 0's to 1 and 1's to exp(1). [This should only be used if you're doing OLS on transformed covariates, not
	the usual GLM thing]."""

	## Restrict attention to the relevant part of the dataframe.
	df = raw_df.loc[:,predictors]

	## The idea here is to restructure the DF to have extra
	## columns based on the raw columns. We'll call these dummy
	## predictors.
	dummy_predictors = []

	## Now we loop through the predictors, checking their type and
	## processing accordingly.
	for predictor in predictors:

		## Check if the method is specified
		covariate_type = cov_types.get(predictor, None)
	
		## Get the data type and check for int32, int64, float32, 
		## float64, etc.
		if covariate_type is None:
			type_name = df[predictor].dtype.name
			type_check = (type_name.startswith("int") or type_name.startswith("float"))
			if type_check:
				covariate_type = "continuous"
			else:
				covariate_type = "factor"

		## If it's an int or float, we assume it's not
		## catagorical. Then we can leave it alone.
		if covariate_type == "continuous":

			## Append new column, and leave things as is.
			dummy_predictors.append(predictor)
			continue

		## If the type is catagorical, then we need to create
		## a set of new columns with 1 or 0 entries that correspond
		## to each catagory.
		elif covariate_type == "factor":

			## Get the different catagories in alphabetical order
			catagories = df[predictor].value_counts().sort_index(axis=0).index
			
			## Then loop through the catagories and create the 
			## appropriate catagorical column
			for catagory in catagories:

				## Catagory name is predictor_catagory
				name = predictor + ": " + str(catagory)

				## We convert from bool to float for ease later, and apply the link function.
				df.loc[:,name] = (df.loc[:,predictor] == catagory).astype(float).apply(transform)
				dummy_predictors.append(name)

		else:
			raise NameError("Covariate types must be continuous, factor, or None.")

	## Construct the processed df
	dummy_df = df.loc[:,dummy_predictors]

	## Add an intercept if necessary
	if intercept:
		intercept_series = pd.Series(np.ones((len(df),)),index=dummy_df.index,name="intercept").apply(transform)
		dummy_df = pd.concat([intercept_series,dummy_df],axis=1)

	## return the factorized df (which can then be turned into a 
	## numpy array).
	return dummy_df