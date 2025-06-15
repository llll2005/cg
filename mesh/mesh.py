import numpy as np
import numba

class HybridMesh:
    def __init__(self, vertices, faces, cells, face_to_cells):
        """
        預設輸入：
        vertices: (V,3) np.ndarray
        faces: list of list of vertex indices
        cells: list of list of vertex indices
        face_to_cells: (F,2) np.ndarray，儲存每個面所連接的兩個 cell 索引，若邊界第二個為 -1
        """
        self.vertices = vertices
        self.faces = faces
        self.cells = cells
        self.face_to_cells = face_to_cells.astype(np.int32)  # 強制轉為 np.int32

        self.num_cells = len(cells)
        self.num_faces = len(faces)

        # 建立 cell_to_faces CSR 結構
        self.cell_face_offsets, self.cell_face_indices = self.build_cell_to_faces()

        # 預計算中心點
        self.cell_centers = self.compute_cell_centers()
        self.face_centers = self.compute_face_centers()

        # face normals 預留，建議外部計算或另外寫函式
        self.face_normals = self.compute_face_normals()

    def build_cell_to_faces(self):
        """
        將 cell_to_faces 從 dict 改成 CSR 格式
        回傳兩個 numpy 陣列:
        cell_face_offsets: (num_cells+1,) int32
        cell_face_indices: (num_faces*avg_faces_per_cell,) int32

        每個 cell 的 face 範圍是 cell_face_indices[cell_face_offsets[i]:cell_face_offsets[i+1]]
        """
        # 先建立 list of list
        temp = [[] for _ in range(self.num_cells)]
        for f_idx, (a, b) in enumerate(self.face_to_cells):
            if a >= 0:
                temp[a].append(f_idx)
            if b >= 0 and b != a:
                temp[b].append(f_idx)

        total_faces = sum(len(flist) for flist in temp)
        cell_face_offsets = np.zeros(self.num_cells + 1, dtype=np.int32)
        cell_face_indices = np.zeros(total_faces, dtype=np.int32)

        pos = 0
        for i in range(self.num_cells):
            flist = temp[i]
            l = len(flist)
            cell_face_offsets[i] = pos
            cell_face_indices[pos:pos + l] = flist
            pos += l
        cell_face_offsets[self.num_cells] = pos
        return cell_face_offsets, cell_face_indices

    def compute_cell_centers(self):
        cells_np = [np.array(cell, dtype=np.int32) for cell in self.cells]
        vertices = self.vertices
        centers = np.zeros((self.num_cells, 3), dtype=vertices.dtype)

        for i, cell in enumerate(cells_np):
            pts = vertices[cell]
            centers[i] = pts.mean(axis=0)
        return centers

    def compute_face_centers(self):
        faces_np = [np.array(face, dtype=np.int32) for face in self.faces]
        vertices = self.vertices
        centers = np.zeros((self.num_faces, 3), dtype=vertices.dtype)

        for i, face in enumerate(faces_np):
            pts = vertices[face]
            centers[i] = pts.mean(axis=0)
        return centers

    def compute_face_normals(self):
        # 簡單計算多邊形面法向量：取前三頂點算叉積（不一定準）
        normals = np.zeros((self.num_faces, 3), dtype=self.vertices.dtype)
        for i, face in enumerate(self.faces):
            if len(face) < 3:
                normals[i] = np.array([0, 0, 0])
                continue
            p0, p1, p2 = [self.vertices[idx] for idx in face[:3]]
            v1 = p1 - p0
            v2 = p2 - p0
            n = np.cross(v1, v2)
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n /= norm
            normals[i] = n
        return normals

    def get_faces_around_cell(self, cell_idx, rings=1):
        """
        回傳 cell_idx 周圍 rings 層鄰接 faces 的索引列表
        使用 CSR 結構加速查詢
        """
        # 簡單 BFS 找鄰接 cell
        visited_cells = set([cell_idx])
        frontier = set([cell_idx])
        for _ in range(rings):
            new_frontier = set()
            for c in frontier:
                start = self.cell_face_offsets[c]
                end = self.cell_face_offsets[c+1]
                faces = self.cell_face_indices[start:end]
                for f in faces:
                    a, b = self.face_to_cells[f]
                    if a != c and a >= 0 and a not in visited_cells:
                        new_frontier.add(a)
                    if b != c and b >= 0 and b not in visited_cells:
                        new_frontier.add(b)
            visited_cells |= new_frontier
            frontier = new_frontier
            if not frontier:
                break

        # 收集所有面
        face_set = set()
        for c in visited_cells:
            start = self.cell_face_offsets[c]
            end = self.cell_face_offsets[c+1]
            for f in self.cell_face_indices[start:end]:
                face_set.add(f)
        return list(face_set)
