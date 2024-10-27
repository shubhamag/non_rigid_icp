import numpy as np
import os
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sksparse.cholmod import cholesky_AAt
import trimesh

def spsolve_chol(sparse_X, dense_b):
    factor = cholesky_AAt(sparse_X.T)
    return factor(sparse_X.T.dot(dense_b)).toarray()

def rigid_transform_3D(A, B):
    """
    Calculates the rigid body transformation between two sets of points A and B.
    """
    assert A.shape == B.shape

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    return R, t

def compute_vertex_normals(vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh.vertex_normals

def perform_mesh_registration(
    target_mesh_path,
    template_mesh_path,
    landmarks_path,
    output_dir,
    template_mesh_faces_path=None,
    mouth_mask_path=None,
    alpha_stiffness_list=None,
    beta_weights_list=None,
    bfm_landmark_indices=None,
    scale_landmark_indices_template=None,
    scale_landmark_indices_target=None,
    max_iterations=5,
):
    """
    Perform mesh registration by aligning a template mesh to a target point cloud,
    using landmark points and regularization.
    """
    # Read target point cloud
    target_mesh = trimesh.load(target_mesh_path)
    target_vertices = target_mesh.vertices
    print("Done reading target...")

    # Read landmarks
    gt_lms = np.loadtxt(landmarks_path)
    nL = gt_lms.shape[0]

    # Read template mesh
    template_mesh = trimesh.load(template_mesh_path)
    mesh_vertices = template_mesh.vertices
    mesh_faces = template_mesh.faces

    # Use provided faces if given
    if template_mesh_faces_path is not None:
        mesh_faces = np.loadtxt(template_mesh_faces_path, dtype=int) - 1  # assuming 1-based indexing

    # Landmark indices
    if bfm_landmark_indices is None:
        bfm_landmark_indices = np.array([2088, 5959, 10603, 14472, 8319, 5781, 11070, 19770, 35341])

    # Scaling indices
    if scale_landmark_indices_template is None:
        scale_landmark_indices_template = (2087, 14471)
    if scale_landmark_indices_target is None:
        scale_landmark_indices_target = (0, 3)  # Indices in gt_lms

    # use scale_landmark_indices to Compute scaling factor
    template_scale_length = np.linalg.norm(
        mesh_vertices[scale_landmark_indices_template[0], :] - mesh_vertices[scale_landmark_indices_template[1], :]
    )
    target_scale_length = np.linalg.norm(
        gt_lms[scale_landmark_indices_target[0]] - gt_lms[scale_landmark_indices_target[1]]
    )
    print("template mean mesh norm :", template_scale_length)
    print("ground truth mesh norm: ", target_scale_length)
    #apply the scaling
    scale = template_scale_length / target_scale_length
    mesh_vertices = mesh_vertices / scale

    # Apply rigid transformation
    R, t = rigid_transform_3D(mesh_vertices[bfm_landmark_indices, :], gt_lms)
    mesh_vertices = (R @ mesh_vertices.T).T + t.reshape(1, 3)
    nV = mesh_vertices.shape[0]

    # Center and scale
    centre = np.mean(target_vertices, axis=0)
    scale_factor = np.linalg.norm(
        mesh_vertices[scale_landmark_indices_template[0], :] - mesh_vertices[scale_landmark_indices_template[1], :]
    ) * 2

    #now, center and normalize both meshes coordinates
    target_vertices_original = target_vertices.copy()
    target_vertices = target_vertices - centre
    target_vertices = target_vertices / scale_factor

    mesh_vertices = mesh_vertices - centre
    mesh_vertices = mesh_vertices / scale_factor
    orig_mesh_vertices = mesh_vertices.copy()

    gt_lms = gt_lms - centre
    gt_lms = gt_lms / scale_factor

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target_vertices)
    nbrs50 = NearestNeighbors(n_neighbors=30, algorithm='kd_tree').fit(target_vertices)

    nV = mesh_vertices.shape[0]
    n = nV
    print("num verts: ", n)
    edges = []
    for face in mesh_faces:
        s = np.sort(face)
        edges.append(tuple(s[:2]))
        edges.append(tuple(s[1:]))

    edgeset = set(edges)
    e = len(edgeset)
    print("num edges:", e)

    M = sparse.lil_matrix((e, n), dtype=np.float32)

    for i, edge in enumerate(edgeset):
        M[i, edge[0]] = -1
        M[i, edge[1]] = 1

    gamma = 1
    G = np.diag([1, 1, 1, gamma]).astype(np.float32)
    Es = sparse.kron(M, G)
    print("Es shape:", Es.shape)

    UL = gt_lms

    if alpha_stiffness_list is None:
        alpha_stiffness_list = [50, 35, 20, 15, 7, 3, 2, 1]
    if beta_weights_list is None:
        beta_weights_list = [15, 14, 3, 2, 0.5, 0, 0, 0]

    if mouth_mask_path is not None:
        mouth_mask = np.loadtxt(mouth_mask_path).astype(np.int32)
    else:
        mouth_mask = None

    for cnt, (alpha_stiffness, beta) in enumerate(zip(alpha_stiffness_list, beta_weights_list)):
        print("using alpha stiffness and beta(lm wt): ", alpha_stiffness, beta)

        iter_max = max_iterations
        if alpha_stiffness < 25:
            iter_max = 3

        for iter_cnt in range(iter_max):
            #####DATA WEIGHTS########
            weights = np.ones([n, 1])
            if alpha_stiffness > 25:
                weights *= 0.5

            if alpha_stiffness > 15:  # no normals used
                distances, indices = nbrs.kneighbors(mesh_vertices)
                print("median mesh distance:", np.median(distances))
                indices = indices.squeeze()
                matches = target_vertices[indices]
            else:
                indices = None
                distances = np.zeros(mesh_vertices.shape[0])
                matches = np.zeros_like(mesh_vertices)
                mesh_norms = compute_vertex_normals(mesh_vertices, mesh_faces)
                dist50, inds50 = nbrs50.kneighbors(mesh_vertices)
                eps = 2.5e-3  # remember, rescaling system to 1

                for i in range(mesh_vertices.shape[0]):
                    pt_normal = mesh_norms[i]
                    corr50 = target_vertices[inds50[i]]
                    ab = mesh_vertices[i] - corr50
                    c = np.cross(ab, pt_normal)
                    line_dists = np.linalg.norm(c, axis=1)
                    fltr = line_dists < eps
                    if np.sum(fltr) > 0:
                        matches[i, :] = np.mean(corr50[fltr], axis=0)
                    distances[i] = np.linalg.norm(matches[i] - mesh_vertices[i])

            d_thresh = np.linalg.norm(
                mesh_vertices[scale_landmark_indices_template[0], :3] - mesh_vertices[scale_landmark_indices_template[1], :3]
            ) / 4
            mesh_vertices = np.hstack((mesh_vertices, np.ones([n, 1])))

            mismatches = np.where(distances > d_thresh)[0]
            weights[mismatches] = 0
            if mouth_mask is not None:
                weights[mouth_mask] = 0

            mesh_matches = np.where(weights > 0)[0]

            print("Setting up D and V...")
            B = sparse.lil_matrix((4 * e + n, 3), dtype=np.float32)
            DL = sparse.lil_matrix((nL, 4 * n), dtype=np.float32)
            V = sparse.lil_matrix((n, 4 * n), dtype=np.float32)

            B[4 * e: (4 * e + n), :] = weights * matches
            for i in range(n):
                V[i, 4 * i:4 * i + 4] = mesh_vertices[i]

            D = V.multiply(weights)

            lm_wtmat = beta * np.ones([DL.shape[0], 1])

            for i, lm in enumerate(bfm_landmark_indices):
                DL[i, 4 * lm:4 * lm + 4] = lm_wtmat[i] * mesh_vertices[lm]

            D = D.tocsr()
            DL = DL.tocsr()
            A = sparse.csr_matrix(sparse.vstack([alpha_stiffness * Es, D, DL]))

            B = sparse.vstack((B, lm_wtmat * gt_lms))

            print("B size after lms", B.shape)
            print("solving...")
            X = spsolve_chol(A, B)

            print("warping...")
            new_vertices = V.dot(X)

            if iter_cnt == iter_max - 1:
                print("saving...")
                new_mesh_path = os.path.join(output_dir, 'deformed_alpha_' + str(alpha_stiffness) + '_mesh.obj')
                if alpha_stiffness == 15:
                    new_mesh_path = os.path.join(output_dir, 'final_mesh.obj')
                vs = np.asarray(new_vertices) * scale_factor + centre
                if alpha_stiffness <= 15:
                    match_mesh = vs[mesh_matches]
                    targ_matches = (matches[np.where(weights > 0)[0]]) * scale_factor + centre
                    # Save matched meshes
                    matched_mesh = trimesh.Trimesh(vertices=match_mesh)
                    matched_mesh.export(os.path.join(output_dir, 'matched_alpha_' + str(alpha_stiffness) + '_mesh.ply'))
                    target_matches_mesh = trimesh.Trimesh(vertices=targ_matches)
                    target_matches_mesh.export(os.path.join(output_dir, 'matched_alpha_' + str(alpha_stiffness) + '_target.ply'))
                # Save deformed mesh
                deformed_mesh = trimesh.Trimesh(vertices=vs, faces=mesh_faces)
                deformed_mesh.export(new_mesh_path)

            mesh_vertices = new_vertices
