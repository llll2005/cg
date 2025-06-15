# 改進的半拉格朗日平流實現
import numpy as np
from numba import jit, prange
from enum import Enum

class CellType(Enum):
    HEXAHEDRAL = 1
    TETRAHEDRAL = 2
    TRANSITION = 3

class SemiLagrangianAdvection:
    def __init__(self, mesh, velocity_field):
        self.mesh = mesh
        self.velocity_field = velocity_field

        # 預計算頂點速度用於快速插值
        self.vertex_velocities = None
        self.vertex_velocities_valid = False

        # 預計算的權重和鄰域信息
        self.face_neighborhoods = {}
        self.average_edge_lengths = {}

    def _update_vertex_velocities(self):
        """根據論文3.5節，使用常數基函數MLS計算頂點速度"""
        if self.vertex_velocities_valid:
            return

        self.vertex_velocities = np.zeros((len(self.mesh.vertices), 3), dtype=np.float64)

        for v_idx in range(len(self.mesh.vertices)):
            vertex = self.mesh.vertices[v_idx]
            # 找到與該頂點相鄰的面
            adjacent_faces = self.mesh.get_adjacent_faces(v_idx)

            if len(adjacent_faces) > 0:
                # 使用常數基函數MLS插值
                velocity = self._mls_constant_interpolation(vertex, adjacent_faces)
                self.vertex_velocities[v_idx] = velocity

        self.vertex_velocities_valid = True

    def _mls_constant_interpolation(self, position, face_indices):
        """移動最小二乘插值 - 常數基函數版本 (論文方程7-8)"""
        if len(face_indices) == 0:
            return np.zeros(3)

        # 建立法向量矩陣 N (方程7)
        normals = np.array([self.mesh.face_normals[i] for i in face_indices])

        # 建立右側向量 z (法向速度分量)
        z = np.array([self.velocity_field.u_normal[i] for i in face_indices])

        # 計算權重矩陣 W (方程9)
        weights = self._compute_weights(position, face_indices)

        # 解加權最小二乘問題: (N^T W N) u = N^T W z (方程10)
        return _solve_weighted_least_squares(normals, z, weights)

    def _mls_linear_interpolation(self, position, face_indices):
        """移動最小二乘插值 - 線性基函數版本 (論文方程11)"""
        if len(face_indices) == 0:
            return np.zeros(3)

        # 建立擴展矩陣 (方程11)
        face_centers = np.array([self.mesh.face_centers[i] for i in face_indices])
        relative_pos = face_centers - position

        # 構建增廣矩陣 [n_i, x_i*n_i, y_i*n_i, z_i*n_i] 對每個面
        A = np.zeros((len(face_indices), 12))  # 4個係數 × 3個速度分量

        for i, face_idx in enumerate(face_indices):
            normal = self.mesh.face_normals[face_idx]
            rel_pos = relative_pos[i]

            # 對於每個速度分量構建矩陣行
            for comp in range(3):
                base_idx = comp * 4
                A[i, base_idx] = normal[comp]           # 常數項
                A[i, base_idx + 1] = rel_pos[0] * normal[comp]  # x項
                A[i, base_idx + 2] = rel_pos[1] * normal[comp]  # y項
                A[i, base_idx + 3] = rel_pos[2] * normal[comp]  # z項

        # 右側向量
        z = np.array([self.velocity_field.u_normal[i] for i in face_indices])

        # 權重矩陣
        weights = self._compute_weights(position, face_indices)

        try:
            # 解加權最小二乘問題
            coeffs = _solve_weighted_least_squares_extended(A, z, weights)

            # 提取常數項 (u_0) - 論文中評估點作為原點，所以u(x) = u_0
            velocity = coeffs[[0, 4, 8]]  # 取出每個分量的常數項

        except np.linalg.LinAlgError:
            # 降級到常數插值
            velocity = self._mls_constant_interpolation(position, face_indices)

        return velocity

    def _compute_weights(self, position, face_indices):
        """計算移動最小二乘權重 (論文方程9)"""
        if len(face_indices) == 0:
            return np.array([])

        # 獲取或計算平均邊長
        avg_edge_length = self._get_average_edge_length(position)
        epsilon = 0.03 * avg_edge_length  # 論文建議值

        face_centers = np.array([self.mesh.face_centers[i] for i in face_indices])
        distances = np.linalg.norm(face_centers - position, axis=1)

        # w_ii = 1 / (r_i^2 + ε^2)
        weights = 1.0 / (distances**2 + epsilon**2)

        return weights

    def _get_average_edge_length(self, position):
        """獲取點周圍的平均邊長"""
        # 這裡可以預計算或快速估算
        # 簡化版本：使用全局平均值
        if not hasattr(self.mesh, 'global_avg_edge_length'):
            self.mesh.global_avg_edge_length = self._compute_global_avg_edge_length()
        return self.mesh.global_avg_edge_length

    def _compute_global_avg_edge_length(self):
        """計算全局平均邊長"""
        total_length = 0.0
        edge_count = 0

        for cell_idx in range(len(self.mesh.cells)):
            edges = self.mesh.get_cell_edges(cell_idx)
            for edge in edges:
                v1, v2 = edge
                length = np.linalg.norm(self.mesh.vertices[v1] - self.mesh.vertices[v2])
                total_length += length
                edge_count += 1

        return total_length / edge_count if edge_count > 0 else 1.0

    def _interpolate_velocity_at_point(self, position, use_fast_method=True):
        """在指定點插值速度"""
        containing_cell = self.mesh.locate_cell(position)

        if containing_cell is None:
            return np.zeros(3)

        cell_type = self.mesh.get_cell_type(containing_cell)

        if cell_type == CellType.HEXAHEDRAL:
            # 規則六面體使用三線性插值
            return self._trilinear_interpolation(position, containing_cell)
        else:
            # 四面體或過渡單元
            if use_fast_method:
                # 快速方法：使用預計算的頂點速度
                self._update_vertex_velocities()
                return self._linear_interpolation_from_vertices(position, containing_cell)
            else:
                # 精確方法：使用線性基函數MLS
                # 論文要求使用二環鄰域
                extended_faces = self.mesh.get_two_ring_faces(position)
                return self._mls_linear_interpolation(position, extended_faces)

    def _trilinear_interpolation(self, position, cell_id):
        """三線性插值用於規則六面體 (標準方法)"""
        # 獲取六面體的8個頂點
        vertices = self.mesh.get_cell_vertices(cell_id)
        vertex_positions = self.mesh.vertices[vertices]

        # 計算局部座標
        min_pos = np.min(vertex_positions, axis=0)
        max_pos = np.max(vertex_positions, axis=0)

        # 正規化座標 [0,1]^3
        local_pos = (position - min_pos) / (max_pos - min_pos)
        local_pos = np.clip(local_pos, 0.0, 1.0)

        # 三線性插值
        return _trilinear_interpolate(local_pos, self.velocity_field.get_face_velocities(cell_id))

    def _linear_interpolation_from_vertices(self, position, cell_id):
        """從頂點速度進行線性插值"""
        cell_vertices = self.mesh.get_cell_vertices(cell_id)
        vertex_positions = self.mesh.vertices[cell_vertices]
        vertex_vels = self.vertex_velocities[cell_vertices]

        # 計算重心座標
        barycentric = self.mesh.compute_barycentric_coordinates(position, cell_id)

        # 線性插值
        velocity = np.sum(vertex_vels * barycentric.reshape(-1, 1), axis=0)
        return velocity

    def advect_velocity(self, dt):
        """平流速度場 (論文算法核心)"""
        new_velocity = np.zeros_like(self.velocity_field.u_normal)

        # 先更新頂點速度用於後向追蹤
        self._update_vertex_velocities()

        # 並行化面循環
        face_centers = self.mesh.face_centers
        face_normals = self.mesh.face_normals

        # 使用numba加速
        new_velocity = _advect_velocity_numba(
            face_centers, face_normals, dt,
            self.mesh, self.velocity_field, self
        )

        self.velocity_field.set_normal_component(new_velocity)

        # 標記頂點速度需要重新計算
        self.vertex_velocities_valid = False

    def _backward_trace(self, start_pos, dt, use_fast_method=True):
        """後向追蹤 (RK2積分)"""
        # 第一步：中點法
        v1 = self._interpolate_velocity_at_point(start_pos, use_fast_method)
        mid_pos = start_pos - 0.5 * dt * v1

        # 第二步：使用中點速度
        v2 = self._interpolate_velocity_at_point(mid_pos, use_fast_method)
        back_pos = start_pos - dt * v2

        return back_pos

    def advect_particles(self, particles, dt):
        """平流粒子 (用於渲染)"""
        # 確保頂點速度已更新
        self._update_vertex_velocities()

        # 使用numba加速粒子平流
        positions = np.array([p.position for p in particles])
        new_positions = _advect_particles_numba(
            positions, dt, self.mesh, self.vertex_velocities
        )

        for i, particle in enumerate(particles):
            particle.position = new_positions[i]


# Numba加速函數
@jit(nopython=True)
def _solve_weighted_least_squares(normals, z, weights):
    """解加權最小二乘問題，移除不支援的 np.diag"""
    n_faces, n_dim = normals.shape

    NtWN = np.zeros((n_dim, n_dim))
    NtWz = np.zeros(n_dim)

    for i in range(n_faces):
        w = weights[i]
        n = normals[i]
        NtWN += w * np.outer(n, n)
        NtWz += w * n * z[i]

    try:
        velocity = np.linalg.solve(NtWN, NtWz)
    except:
        # 備案使用 pseudo-inverse（略低效，但穩定）
        velocity = np.linalg.pinv(NtWN) @ NtWz

    return velocity

@jit(nopython=True)
def _solve_weighted_least_squares_extended(A, z, weights):
    n_rows, n_cols = A.shape
    AtWA = np.zeros((n_cols, n_cols))
    AtWz = np.zeros(n_cols)

    for i in range(n_rows):
        w = weights[i]
        a = A[i]
        AtWA += w * np.outer(a, a)
        AtWz += w * a * z[i]

    try:
        coeffs = np.linalg.solve(AtWA, AtWz)
    except:
        coeffs = np.linalg.pinv(AtWA) @ AtWz

    return coeffs

@jit(nopython=True)
def _trilinear_interpolate(local_pos, face_velocities):
    """三線性插值實現"""
    x, y, z = local_pos

    # 8個頂點的權重
    weights = np.array([
        (1-x)*(1-y)*(1-z),  # 000
        x*(1-y)*(1-z),      # 100
        (1-x)*y*(1-z),      # 010
        x*y*(1-z),          # 110
        (1-x)*(1-y)*z,      # 001
        x*(1-y)*z,          # 101
        (1-x)*y*z,          # 011
        x*y*z               # 111
    ])

    # 加權平均
    velocity = np.zeros(3)
    for i in range(8):
        velocity += weights[i] * face_velocities[i]

    return velocity

@jit(nopython=True, parallel=True)
def _advect_velocity_numba(face_centers, face_normals, dt, mesh, velocity_field, advector):
    """使用numba加速的速度平流"""
    n_faces = len(face_centers)
    new_velocity = np.zeros(n_faces)

    for i in prange(n_faces):
        face_center = face_centers[i]
        face_normal = face_normals[i]

        # 後向追蹤
        back_pos = _backward_trace_numba(face_center, dt, advector)

        # 在追蹤點插值速度
        interpolated_velocity = _interpolate_at_point_numba(back_pos, mesh, advector)

        # 投影到面法向量
        new_velocity[i] = np.dot(interpolated_velocity, face_normal)

    return new_velocity

@jit(nopython=True)
def _backward_trace_numba(start_pos, dt, advector):
    """numba版本的後向追蹤"""
    # 簡化版本：使用歐拉法
    velocity = _interpolate_at_point_numba(start_pos, advector.mesh, advector)
    back_pos = start_pos - dt * velocity
    return back_pos

@jit(nopython=True)
def _interpolate_at_point_numba(position, mesh, advector):
    """numba版本的插值"""
    # 這裡實現簡化的插值邏輯
    # 實際實現需要根據具體的mesh結構調整
    return np.zeros(3)

@jit(nopython=True, parallel=True)
def _advect_particles_numba(positions, dt, mesh, vertex_velocities):
    """numba加速的粒子平流"""
    n_particles = len(positions)
    new_positions = np.zeros_like(positions)

    for i in prange(n_particles):
        # 簡化的前向歐拉積分
        velocity = _interpolate_velocity_at_particle(positions[i], mesh, vertex_velocities)
        new_positions[i] = positions[i] + dt * velocity

    return new_positions

@jit(nopython=True)
def _interpolate_velocity_at_particle(position, mesh, vertex_velocities):
    """在粒子位置插值速度"""
    # 實現基於頂點速度的插值
    # 這裡需要根據具體mesh結構實現
    return np.zeros(3)