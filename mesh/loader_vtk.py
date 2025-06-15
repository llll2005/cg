def load_vtk_to_hybrid_mesh(filename):
    import pyvista as pv
    import numpy as np
    from mesh.mesh import HybridMesh

    vtk = pv.read(filename)

    if not isinstance(vtk, pv.UnstructuredGrid):
        raise ValueError("輸入 VTK 檔案不是 tetra 格式的 UnstructuredGrid。請確認使用 delaunay_3d().")

    points = np.array(vtk.points)
    cells = []
    cell_types = []
    cell_map = {10: "tet", 12: "hex"}

    for i in range(vtk.n_cells):
        c = vtk.get_cell(i)
        t = vtk.celltypes[i]
        if t == 10 and len(c.point_ids) == 4:  # VTK_TETRA
            cells.append(list(c.point_ids))
            cell_types.append("tet")

    # 建立假面資料（可用 marching cube 或其他方法建立真正面）
    dummy_faces = [[0, 1, 2]] * (len(cells) * 4)
    dummy_normals = [[0, 0, 1]] * len(dummy_faces)
    dummy_face_to_cells = [(i, i) for i in range(len(dummy_faces))]

    return HybridMesh(cells, cell_types, points, dummy_faces, dummy_normals, dummy_face_to_cells)
