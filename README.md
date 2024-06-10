# GMRES implementation

## Notebooks:
### Arnoldi.
- Arnoldi: Arnoldi algorithm full basis.
- Arnoldi_one_iter: One new vector for the orthonormal basis using Arnoldi's algorithm.
-----
### GMRES. There are three versions. The first version uses the full arnoldi algorithm whereas the second one uses arnoldi_one_iter at each step of the GMRES algorithm. This way saving a lot of time as we only compute one more orthonormal vector at a time instead of the whole orthonormal basis. The second one also offers the restarting and preconditioning options. The third one is GMRES implemented in torch.
- GMRES_BASIC: Raw implementation of GMRES using Arnoldi.
- GMRES_OPTIMIZED: Optimized with Arnoldi_one_iter. Added restarting and preconditioning.
- GMRES_torch: GMRES implemented using torch.
----

------
## Python files:
- functions.py: arnoldi, arnoldi_one_iter and backsub functions used in GMRES.

## COMMENTS.md
- Some comments about the implementation and all the arbitrary choices made when implementing GMRES.

## numml installation
- clone repository and then pip3 install .
- But first I had to install cuda version 11.8 (same as pytorch) and then install microsoft c++ build tools including Microsoft Visual C++ 14.0 or greater is required (https://zs.fyi/archives/python-vc-14-0-error.html) 
