import numpy as np
from scipy.linalg import lstsq

class VelocityInterpolator:
    def __init__(self, mesh, velocity_field):
        self.mesh = mesh
        self.velocity_field = velocity_field
        self.epsilon_factor = 0.03  # Suggested in paper

    def interpolate_velocity(self, point):
        """Interpolate velocity at an arbitrary point using MLS."""
        cell = self.mesh.locate_cell(point)
        if cell is None:
            return np.zeros(3)

        face_ids = self.mesh.get_faces_around_point(point, rings=1)
        if len(face_ids) < 3:
            return np.zeros(3)

        positions = self.mesh.face_centers[face_ids] - point  # relative positions
        normals = self.velocity_field.face_normals[face_ids].astype(np.float64)
        values = self.velocity_field.u_normal[face_ids]  # scalar face-normal velocities

        avg_edge_length = self.mesh.get_average_edge_length_around_cell(cell)
        epsilon = self.epsilon_factor * avg_edge_length

        return self._linear_mls_interpolation(positions, normals, values, epsilon)

    def _linear_mls_interpolation(self, positions, normals, values, epsilon):
        """Linear MLS interpolation (paper eq.11)."""
        n_faces = len(positions)
        if n_faces < 4:
            return self._constant_mls_interpolation(positions, normals, values, epsilon)

        A = np.zeros((n_faces, 12), dtype=np.float64)
        b = values.copy().astype(np.float64)
        weights = np.zeros(n_faces)

        for i in range(n_faces):
            x, y, z = positions[i]
            n = normals[i]

            r_i = np.linalg.norm(positions[i])
            weights[i] = 1.0 / (r_i**2 + epsilon**2)

            # Row: [n, x*n, y*n, z*n]
            A[i, 0:3] = n
            A[i, 3:6] = x * n
            A[i, 6:9] = y * n
            A[i, 9:12] = z * n

        W = np.diag(np.sqrt(weights))
        A_weighted = W @ A
        b_weighted = W @ b

        try:
            coeffs, residuals, rank, s = lstsq(A_weighted, b_weighted, rcond=None)
            if rank < 12:
                return self._constant_mls_interpolation(positions, normals, values, epsilon)
            return coeffs[0:3]  # constant term = interpolated velocity vector
        except:
            return np.zeros(3)

    def _constant_mls_interpolation(self, positions, normals, values, epsilon):
        """Fallback: constant-weighted least squares."""
        n_faces = len(positions)
        A = np.zeros((n_faces, 3), dtype=np.float64)
        b = values.copy().astype(np.float64)
        weights = np.zeros(n_faces)

        for i in range(n_faces):
            r_i = np.linalg.norm(positions[i])
            weights[i] = 1.0 / (r_i**2 + epsilon**2)
            A[i, :] = normals[i]

        W = np.diag(np.sqrt(weights))
        A_weighted = W @ A
        b_weighted = W @ b

        try:
            u, residuals, rank, s = lstsq(A_weighted, b_weighted, rcond=None)
            return u if rank >= 3 else np.zeros(3)
        except:
            return np.zeros(3)
