import numpy as np
from numba import jit, prange

class ForceApplicator:
    def __init__(self, gravity=None, reference_density=1.0):
        if gravity is None:
            gravity = np.array([0.0, 0.0, -9.8], dtype=np.float64)
        self.gravity = gravity
        self.reference_density = reference_density  # 參考密度，用於浮力計算

    def apply_body_forces(self, velocity_field, density_field, mesh, dt):
        face_to_cells = velocity_field.face_to_cells
        face_normals = velocity_field.face_normals
        cell_volumes = mesh.cell_volumes
        densities = density_field.rho

        self._apply_body_forces_numba(
            velocity_field.u_normal,
            face_to_cells,
            face_normals,
            cell_volumes,
            densities,
            self.gravity,
            self.reference_density,
            dt
        )

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _apply_body_forces_numba(u_normal, face_to_cells, face_normals,
                                 cell_volumes, densities, gravity, ref_density, dt):
        n_faces = len(face_to_cells)

        for i in prange(n_faces):
            a, b = face_to_cells[i, 0], face_to_cells[i, 1]
            if a >= len(densities) or b >= len(densities):
                continue

            rho_a = densities[a]
            rho_b = densities[b] if b != a else rho_a
            V_a = cell_volumes[a]
            V_b = cell_volumes[b] if b != a else V_a

            buoyancy_a = (ref_density - rho_a) * gravity / rho_a
            buoyancy_b = (ref_density - rho_b) * gravity / rho_b

            total_force_a = V_a * buoyancy_a
            total_force_b = V_b * buoyancy_b

            avg_acceleration = (total_force_a + total_force_b) / (V_a + V_b)

            normal_accel = 0.0
            for j in range(len(gravity)):
                normal_accel += face_normals[i, j] * avg_acceleration[j]

            u_normal[i] += normal_accel * dt

    def apply_vorticity_confinement(self, velocity_field, mesh, epsilon=0.01, dt=0.01):
        vorticity = self._compute_vorticity(velocity_field, mesh)

        self._apply_vorticity_force_numba(
            velocity_field.u_normal,
            velocity_field.face_to_cells,
            velocity_field.face_normals,
            mesh.cell_volumes,
            vorticity,
            epsilon,
            dt
        )

    def _compute_vorticity(self, velocity_field, mesh):
        n_cells = len(mesh.cell_volumes)
        vorticity = np.zeros((n_cells, 3), dtype=np.float64)
        return vorticity

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _apply_vorticity_force_numba(u_normal, face_to_cells, face_normals,
                                     cell_volumes, vorticity, epsilon, dt):
        n_faces = len(face_to_cells)

        for i in prange(n_faces):
            a, b = face_to_cells[i, 0], face_to_cells[i, 1]
            if a >= len(vorticity) or b >= len(vorticity):
                continue

            V_a = cell_volumes[a]
            V_b = cell_volumes[b] if b != a else V_a

            avg_vorticity = [0.0, 0.0, 0.0]
            for j in range(3):
                avg_vorticity[j] = (V_a * vorticity[a, j] + V_b * vorticity[b, j]) / (V_a + V_b)

            vorticity_force = 0.0
            for j in range(3):
                vorticity_force += face_normals[i, j] * avg_vorticity[j]

            u_normal[i] += epsilon * vorticity_force * dt

    def apply_external_forces(self, velocity_field, density_field, mesh,
                               external_force_func=None, dt=0.01, time=0.0):
        if external_force_func is None:
            return

        cell_centers = mesh.get_cell_centers()

        for i, (a, b) in enumerate(velocity_field.face_to_cells):
            if a >= len(density_field.rho) or b >= len(density_field.rho):
                continue

            face_center = (cell_centers[a] + cell_centers[b]) / 2.0
            external_force = external_force_func(face_center, time)

            V_a = mesh.cell_volumes[a]
            V_b = mesh.cell_volumes[b] if b != a else V_a
            rho_a = density_field.rho[a]
            rho_b = density_field.rho[b] if b != a else rho_a

            f_a = external_force / rho_a
            f_b = external_force / rho_b

            avg_acceleration = (V_a * f_a + V_b * f_b) / (V_a + V_b)
            normal_accel = np.dot(velocity_field.face_normals[i], avg_acceleration)

            velocity_field.u_normal[i] += normal_accel * dt

    def step(self, velocity_field, density_field, mesh, dt, time=0.0):
        self.apply_body_forces(velocity_field, density_field, mesh, dt)
        # self.apply_vorticity_confinement(velocity_field, mesh, dt=dt)
        # self.apply_external_forces(velocity_field, density_field, mesh, dt=dt, time=time)
