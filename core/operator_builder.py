import scipy.sparse as sp
import numpy as np

def build_operators(mesh):
    """根據論文 Eq.(3) 與 Eq.(4) 構建散度 D 和梯度 G 算子（稀疏矩陣）"""
    num_cells = len(mesh.cells)
    num_faces = len(mesh.faces)

    rows_D, cols_D, data_D = [], [], []
    rows_G, cols_G, data_G = [], [], []

    for i, (a, b) in enumerate(mesh.face_to_cells):
        # 邊界面判斷，若b是-1或非法索引表示邊界面，依照你的mesh定義調整
        if a < 0 or a >= num_cells:
            continue
        if b < 0 or b >= num_cells:
            b = a  # 對邊界面，視同a單元雙面處理（論文中會特別處理）

        n = mesh.face_normals[i]
        A = mesh.face_areas[i]

        # Eq.(3) 散度算子組成
        rows_D.append(a)
        cols_D.append(i)
        data_D.append(A)

        if b != a:
            rows_D.append(b)
            cols_D.append(i)
            data_D.append(-A)

        # Eq.(4) 梯度算子組成
        k_a = compute_k_factor(mesh, a, i)
        k_b = compute_k_factor(mesh, b, i)

        k_total = k_a + k_b
        if k_total == 0:
            # 避免除零，退化為單側係數
            if b == a:
                rows_G.append(i)
                cols_G.append(a)
                data_G.append(1.0 / (k_a if k_a != 0 else 1e-8))
            else:
                # 此狀況不理想，暫跳過
                continue
        else:
            if b != a:
                rows_G.extend([i, i])
                cols_G.extend([a, b])
                data_G.extend([-1.0 / k_total, 1.0 / k_total])
            else:
                rows_G.append(i)
                cols_G.append(a)
                data_G.append(1.0 / k_a)

    D = sp.coo_matrix((data_D, (rows_D, cols_D)), shape=(num_cells, num_faces)).tocsr()
    G = sp.coo_matrix((data_G, (rows_G, cols_G)), shape=(num_faces, num_cells)).tocsr()

    return D, G

def compute_k_factor(mesh, cell_idx, face_idx):
    """計算論文 Eq.(4) 中的 k 因子"""
    cell_type = mesh.cell_types[cell_idx]

    if cell_type in ["hex", "tran"]:
        h = mesh.get_cell_size(cell_idx)
        return 0.5 * h  # h/2
    elif cell_type == "tet":
        V_j = mesh.cell_volumes[cell_idx]
        A_i = mesh.face_areas[face_idx]
        return 3.0 * V_j / (4.0 * A_i)
    else:
        # 預設值，避免例外
        return 1.0
