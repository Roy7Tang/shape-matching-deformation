# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Shape Matching

def opt_linear_trans(m, P, Q):
    m = np.asarray(m).flatten()
    P = np.asarray(P)
    Q = np.asarray(Q)
    M = np.diag(m)
    Apq = P.T @ M @ Q
    Aqq = Q.T @ M @ Q
    return Apq,Aqq

def jacobi_rota_decomp(A, max_iter=10):
    assert A.shape == (3, 3)
    A = A.copy()
    V = np.eye(3)

    for _ in range(max_iter):
        off_diag = [(0, 1), (0, 2), (1, 2)]
        i, j = max(off_diag, key=lambda x: abs(A[x]))

        if abs(A[i, j]) < 1e-10:
            break

        if A[j, j] - A[i, i] != 0:
            theta = 0.5 * np.arctan2(2 * A[i, j], A[j, j] - A[i, i])
        else:
            theta = np.pi / 4.0

        c, s = np.cos(theta), np.sin(theta)

        J = np.eye(3)
        J[i, i] = c
        J[j, j] = c
        J[i, j] = s
        J[j, i] = -s

        A = J.T @ A @ J
        V = V @ J

    d = np.diag(A)
    return V, d

def sqrtm_symmetric(A):
    V, d = jacobi_rota_decomp(A, max_iter=10)
    d = np.maximum(d, 0.0)
    return V @ np.diag(np.sqrt(d)) @ V.T

def shape_matching(x0, x, m, beta):
    m = np.asarray(m).flatten()
    x0 = np.asarray(x0)
    x = np.asarray(x)

    t0 = np.sum(m[:, None] * x0, axis=0) / np.sum(m)
    t  = np.sum(m[:, None] * x,  axis=0) / np.sum(m)

    Q = x0 - t0
    P = x  - t

    Apq, Aqq = opt_linear_trans(m, P, Q)
    S = sqrtm_symmetric(Apq.T @ Apq)

    R = Apq @ np.linalg.pinv(S)
    A = Apq @ np.linalg.pinv(Aqq)

    g = ((beta * A + (1-beta) * R) @ (x0 - t0).T).T + t
    return g

def x_transformer(x0, angles, translations):
    """
    Input:
        x0: (N,3) 3D points
       angles = (ax, ay, az): rotation angles as degree
       translations = (tx, ty, tz): how long do the points translate
    """
    ax, ay, az = np.deg2rad(angles)

    def rotat_mat(ax, ay, az):
        cx, sx = np.cos(ax), np.sin(ax)
        cy, sy = np.cos(ay), np.sin(ay)
        cz, sz = np.cos(az), np.sin(az)

        Rx = np.array([
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx,  cx]
        ])

        Ry = np.array([
            [ cy, 0, sy],
            [  0, 1,  0],
            [-sy, 0, cy]
        ])

        Rz = np.array([
            [cz, -sz, 0],
            [sz,  cz, 0],
            [ 0,   0, 1]
        ])

        return Rz @ Ry @ Rx  # Order of XYZ

    R = rotat_mat(ax, ay, az)
    tx, ty, tz = translations

    def rotation(P):
      return P @ R.T

    def translation(P):
      t = np.array([tx, ty, tz])
      return P + t

    # All points
    x = translation(rotation(x0))

    return x

#Cube
vals = [0, 1, 2, 3]

x0 = np.array([
    [x, y, z]
    for z in vals
    for y in vals
    for x in vals
    if (x in (0, 3) or y in (0, 3) or z in (0, 3))
])

idx_map = {tuple(p): i for i, p in enumerate(x0)}
edges = []

dirs = [(1,0,0), (0,1,0), (0,0,1)]

for i, p in enumerate(x0):
    for dx, dy, dz in dirs:
        q = (p[0] + dx, p[1] + dy, p[2] + dz)
        if q in idx_map :
            edges.append((i, idx_map[q]))

def simulate_traj():
    # parameters
    h = 1 / 300
    alpha = 0.5
    beta = 0.5
    g_vec = np.array([0, 0, -9.8])
    steps = 1200
    pbd_iters = 3
    damping = 1

    # reference shape
    vals = [0, 1, 2, 3] 

    x0 = np.array([
        [x, y, z]
        for z in vals
        for y in vals
        for x in vals
        if (x in (0, 3) or y in (0, 3) or z in (0, 3))
    ])

    n = x0.shape[0]
    m = np.ones(n)
    # initial pose: rotated cube

    angles = (25, 30, 15)
    translations = (0, 0, 2)
    
    x = x_transformer(x0, angles, translations)
    v = np.zeros_like(x)

    fext = m[:, None] * g_vec

    traj = np.zeros((steps + 1, n, 3))
    traj[0] = x
    
    for i in range(steps):
        v_tmp = v + h * (fext / m[:, None])
        v_tmp *= damping
        x_tmp = x + h * v_tmp

        x_proj = x_tmp.copy()
        for _ in range(pbd_iters):
        # Ground collision constraint
            contact = x_proj[:, 2] < 1e-9
            x_proj[contact, 2] = 0
        # Shape Matching constraint
            g = shape_matching(x0, x_proj, m, beta)
            x_proj += alpha * (g - x_proj)

        v = (x_proj - x) / h
        x = x_proj

        traj[i + 1] = x

    return traj

# Plot settings
def _set_axes(ax, traj):
    pts = traj.reshape(-1, 3)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    pad = 0.5

    ax.set_xlim(mins[0] - pad, maxs[0] + pad)
    ax.set_ylim(mins[1] - pad, maxs[1] + pad)
    ax.set_zlim(0.0, maxs[2] + pad)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # ground plane z=0
    gx = np.linspace(mins[0] - pad, maxs[0] + pad, 2)
    gy = np.linspace(mins[1] - pad, maxs[1] + pad, 2)
    GX, GY = np.meshgrid(gx, gy)
    GZ = np.zeros_like(GX)
    ax.plot_surface(GX, GY, GZ, alpha=0.2)

    # equal aspect in 3D
    xr = (maxs[0] - mins[0]) + 2*pad
    yr = (maxs[1] - mins[1]) + 2*pad
    zr = (maxs[2] - 0.0) + pad
    ax.set_box_aspect((xr, yr, zr))

def save_gif(traj, gif_path="Cube.gif", fps=60, every=5, figsize=(8, 8), dpi=160):
    # every=3 Take 1 in every 3 frames
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    _set_axes(ax, traj)

    scat = ax.scatter([], [], [], s=120)
    title = ax.set_title("")
    lines = []
    for _ in range(len(edges)):
        line, = ax.plot([], [], [], lw=2, color="k")
        lines.append(line)

    title = ax.set_title("")

    frames = list(range(0, len(traj), every))

    def update(frame_idx):
        x = traj[frame_idx]

        # update points
        scat._offsets3d = (x[:, 0], x[:, 1], x[:, 2])

        # update edges
        for line, (i, j) in zip(lines, edges):
            line.set_data([x[i, 0], x[j, 0]],
                          [x[i, 1], x[j, 1]])
            line.set_3d_properties([x[i, 2], x[j, 2]])

        title.set_text(f"step={frame_idx} min_z={x[:,2].min():.4f}")
        return [scat, *lines, title]

    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 / fps,
        blit=False
    )

    print("Saving GIF to:", os.path.abspath(gif_path))
    ani.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("Done. GIF saved.")

def compute_deformation_rms(traj, x0):
    """
    Compute D(t) after removing rigid translation and rotation.

    All particles are assumed to have equal mass. For each frame, the
    reference and current configurations are centered at their centroids,
    and the optimal proper rotation is computed with SVD.
    """
    traj = np.asarray(traj, dtype=float)
    x0 = np.asarray(x0, dtype=float)

    if traj.ndim != 3 or traj.shape[2] != 3:
        raise ValueError("traj must have shape (T, N, 3).")
    if x0.shape != traj.shape[1:]:
        raise ValueError("x0 must have shape (N, 3) matching traj.")

    # Equal masses: center of mass equals the arithmetic mean.
    x0_cm = np.mean(x0, axis=0)
    Q = x0 - x0_cm

    D_rms = np.zeros(traj.shape[0])

    for k, x in enumerate(traj):
        x_cm = np.mean(x, axis=0)
        P = x - x_cm

        # Optimal rotation minimizing sum_i ||P_i - R Q_i||^2.
        Apq = P.T @ Q
        U, _, Vt = np.linalg.svd(Apq)
        R = U @ Vt

        # Exclude reflections: enforce det(R) = +1.
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        Q_rotated = Q @ R.T
        residual = P - Q_rotated

        D_rms[k] = np.sqrt(np.mean(np.sum(residual**2, axis=1)))

    return D_rms

def plot_deformation_curve(D_rms, png_path="deformation_curve.png"):
    step_plot = 1
    idx = np.arange(0, len(D_rms), step_plot)
    
    plt.figure(figsize=(7, 4), dpi=160)
    plt.plot(idx, D_rms[idx], lw=0.8)
    plt.xlabel("Step")
    plt.ylabel(r"$D(t)$")
    plt.title("Deformation metric over time")
    plt.ylim(0, 0.009)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    print("Saved:", png_path)


# Main
if __name__ == "__main__":
    traj = simulate_traj()

    D_rms = compute_deformation_rms(traj, x0)

    plot_deformation_curve(
        D_rms,
        png_path="Dpbdfall.png")

    save_gif(
        traj,
        gif_path="Cubefallpbd.gif",
        fps=60,
        every=5,
        figsize=(8, 8),
        dpi=160
    )
