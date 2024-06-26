# Multisurface models
The Multisurface folder contains example models implementing simple multi-dimensional models which can be given 
different dimensionality (ndim). Useful values of dimensionality are typically in the range 1-3. Models are implemented 
in the "series" form (_ser), "parallel" form (_par) and "nested" form (_nest).

"Bounding surface" variants of the same models (see [doi: 10.1016/j.compgeo.2022.105143](https://doi.org/10.1016/j.compgeo.2022.105143)) 
are also available with tags _ser_b, _par_b, _nest_b.

### Test files:

__run_rate__: Allows comparison of models implemented using rate-independent and rate-dependent formulations.

__run_spiral__: Implements a spiral stress path for each model for comparison. Uses a data file "spiralpath.csv" to 
implement the spiral. Note that the default parameter settings for the models that are compared are chosen so that 
they give identical responses under unidirectional loading. This file also runs the "bounding surface" variants.

__run_square__: Implements a square strain path for each model for comparison. Note that the default parameter 
settings for the models that are compared are chosen so that they give identical responses under unidirectional loading.
