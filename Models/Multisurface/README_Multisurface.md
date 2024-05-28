The Multusurface folder contains example models implementing simple multi-dimensional models which can be given 
different dimensionality (ndim). Useful values of dimensionality are typically in the range 1-3. Models are implemented 
in the "series" form (_ser), "parallel" form (_par) and "nested" form (_nest).

Test files are included:

run_rate: Allows comparison of madels implemented using rate-independent and rate-dependent formulations.

run_spiral: Implements a spiral stress path for each model for comparison. Uses a data file "spiralpath.csv" to 
implement the spiral. Note that the default parameter settings for the models that are compared are chosen so that 
they give identical responses under unidirectional loading.

run_square: Implements a square strain path for each model for comparison. Note that the default parameter 
settings for the models that are compared are chosen so that they give identical responses under unidirectional loading.
