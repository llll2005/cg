import numpy as np
import numba

class VelocityField:
    def __init__(self, num_faces, face_normals, face_to_cells,
                 cell_face_offsets=None, cell_face_indices=None):
        self.u_normal = np.zeros(num_faces, dtype=np.float64)
        self.face_normals = face_normals
        self.face_to_cells = face_to_cells.astype(np.int32)

        # CSR 結構，若沒給就報錯或自行建
        if cell_face_offsets is None or cell_face_indices is None:
            raise ValueError("Must provide cell_face_offsets and cell_face_indices")
        self.cell_face_offsets = cell_face_offsets
        self.cell_face_indices = cell_face_indices

    def extrapolate_ghost_cells(self):
        self.u_normal = _extrapolate_ghost_cells_numba(
            self.u_normal,
            self.face_to_cells,
            self.cell_face_offsets,
            self.cell_face_indices
        )


@numba.njit
def _extrapolate_ghost_cells_numba(u_normal, face_to_cells, cell_face_offsets, cell_face_indices):
    for i in range(face_to_cells.shape[0]):
        a = face_to_cells[i, 0]
        b = face_to_cells[i, 1]
        if b == -1:
            start = cell_face_offsets[a]
            end = cell_face_offsets[a + 1]
            val_sum = 0.0
            count = 0
            for idx in range(start, end):
                f = cell_face_indices[idx]
                # 排除自己是 ghost face
                if face_to_cells[f, 1] != -1:
                    val_sum += u_normal[f]
                    count += 1
            if count > 0:
                u_normal[i] = val_sum / count
            else:
                u_normal[i] = 0.0
    return u_normal
