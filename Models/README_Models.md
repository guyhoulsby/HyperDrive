# Models

The Models folder contains a number of example models, arranged in sub-folders by categories:

__Mises__: "von Mises" type models for continua, implemented using Voigt notation (6-dimensional stress and strain vectors). Note 
that internally HyperDrive implements these models using Mandel notation, converting between the notations on input and output. 

__Multisurface__: A series of multisurface models implemented in the "series", "parallel" and "nested" forms. Each can be 
implemented at different levels of dimensionality (ndim). Example run files are provided.
