import numpy as np
from numba import njit

@njit
def _add_source_kernel(rho, indices, amount):
    for i in range(len(indices)):
        idx = indices[i]
        if 0 <= idx < len(rho):
            rho[idx] += amount

@njit
def _advect_kernel(rho, advected_rho):
    for i in range(len(rho)):
        rho[i] = advected_rho[i]

class DensityField:
    def __init__(self, num_cells):
        self.rho = np.zeros(num_cells)

    def add_source(self, indices, amount):
        indices = np.array(indices, dtype=np.int32)
        _add_source_kernel(self.rho, indices, amount)

    def advect(self, advected_rho):
        _advect_kernel(self.rho, advected_rho)

    def get_values(self):
        return self.rho.copy()
