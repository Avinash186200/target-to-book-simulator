"""From model target weights to a book a desk can hold, and what is lost on the way.

BLAS and OpenMP thread counts are pinned to 1 before numpy is imported, so the
numbers in ``results/`` are independent of core count: bit-identical on the
platform that made them, and expected within 1e-6 elsewhere. Set the
variables yourself before importing if you want something else.
"""
import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

__version__ = "1.0.0"
