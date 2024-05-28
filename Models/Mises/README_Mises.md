# "von Mises" models
This folder contains examples of "von Mises" type continuum models expressed in Voigt notation (stress and strain as 6-dimensional 
vectors). Note that internally HyperDrive implements these models using Mandel notation, converting between the two notations 
on input and output.

The potential functions for these model are expressed using invariants of the stresses etc., as HyperDrive provides functions that
implement these and their derivatives.

The models are:

Mises: a basic elastic-perfectly plastic von Mises model. Requires just three constants: 
   K (bulk modulus), 
   G (shear modulus),  
   k (shear strength)

Mises_minimal: the same model as above, but implemented with minimal code, with none of the derivatives specified. As a result 
HyperDrive runs much slower, as it uses automatically derived differentials, but it gives the same result as the "Mises" model.

Mises_multi: A "multisurface" variant of the von Mises model, with a series implementation of surfaces. Each surface employs
kinematic hardening. For each surface two constants are required:
   k[i] (shear strength for that surface)
   H[i] (hardening modulus)

Test files are supplied:

run_Mises: test file for the "Mises" or "Mises_minimal" models

run_Mises_multi: test file for the "Mises_multi" model
