# "von Mises" models
This folder contains examples of "von Mises" type continuum models expressed in Voigt notation (stress and strain as 6-dimensional 
vectors). Note that internally HyperDrive implements these models using Mandel notation, converting between the two notations 
on input and output - see Section 11 of "Hyperdrive documentation.pdf" for a note on Voigt and Mandel notation, or see [Wikipedia](https://en.wikipedia.org/wiki/Voigt_notation).

The potential functions for these model are expressed using invariants of the stresses _etc_., as HyperDrive provides functions that
implement these and their derivatives.

### Models:

__Mises__: a basic elastic-perfectly plastic von Mises model. Requires just three constants: 
- _K_ (bulk modulus), 
- _G_ (shear modulus),  
- _k_ (shear strength)

__Mises_minimal__: the same model as above, but implemented with minimal code, with none of the derivatives specified. As a result 
HyperDrive runs much slower, as it uses automatically derived differentials, but it gives the same result as the "Mises" model.

__Mises_multi__: A "multisurface" variant of the von Mises model, with a series implementation of surfaces. Each surface employs
kinematic hardening. For each surface two constants are required:
 - _k_<sub>i</sub> (shear strength for that surface)
 - _H_<sub>i</sub> (hardening modulus)

### Test files:

__run_Mises__: test file for the "Mises" or "Mises_minimal" models

__run_Mises_multi__: test file for the "Mises_multi" model
