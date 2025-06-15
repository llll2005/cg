import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np

class PoissonSolver:
    def __init__(self, divergence_matrix, gradient_matrix, mesh, dt=0.1, rho=1.0):
        self.D = divergence_matrix
        self.G = gradient_matrix
        self.mesh = mesh
        self.dt = dt
        self.rho = rho

        # 預計算泊松矩陣
        self.poisson_matrix = self._build_poisson_matrix()

    def _build_poisson_matrix(self):
        """構建對稱正定的泊松矩陣"""
        V_diag = sp.diags(1.0 / self.mesh.cell_volumes, format='csr')
        D_scaled = V_diag @ self.D
        return D_scaled @ self.G

    def solve_pressure(self, u_star):
        """求解壓力修正"""
        divergence = self.D @ u_star
        # 假設 poisson_matrix 已包含 1/V 縮放，不再除以 cell_volumes
        rhs = (self.rho / self.dt) * divergence

        pressure, info = spla.cg(self.poisson_matrix, rhs, rtol=1e-6, maxiter=1000)

        if info != 0:
            print(f"Warning: Pressure solver convergence issue, info={info}")

        return pressure




    def correct_velocity(self, u_star, pressure):
        grad_p = self.G @ pressure
        return u_star - (self.dt / self.rho) * grad_p

    def apply_boundary_conditions(self, pressure, u_corrected):
        """應用邊界條件：區分開放與封閉邊界"""
        for i, (a, b) in enumerate(self.mesh.face_to_cells):
            if a == b or b == -1:
                if hasattr(self.mesh, 'is_closed_boundary') and self.mesh.is_closed_boundary(i):
                    u_corrected[i, :] = 0.0  # 將整個速度向量設為0
        return u_corrected

    def extrapolate_ghost_cells(self):
        """此函式實作於 velocity field 類別中"""
        pass  # 留空由 VelocityField 類補上
