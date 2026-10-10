# shape-matching-deformation

This repository provides Python implementations of 3D shape matching and position-based dynamics (PBD) for particle-based deformation and collision simulations.

The project includes three simulation examples:

- **Shape Recovery** — deformation recovery of a cube using shape matching.
- **Falling Cube with Shape Matching** — gravity-driven motion and ground collision using velocity-based integration.
- **Falling Cube with PBD** — deformable cube simulation using iterative position projection and collision constraints.

The project also includes 3D animation and quantitative deformation analysis.

## Project Overview

The objective is to explore numerical methods for simulating the motion and deformation of 3D objects represented by particles.

The simulation workflow consists of:

1. constructing a reference particle configuration;
2. initializing particle positions, velocities, and masses;
3. computing shape-matching transformations;
4. updating particle positions under external forces and shape constraints;
5. handling ground collisions where applicable;
6. recording particle trajectories;
7. evaluating deformation using an RMS metric;
8. generating 3D animations and deformation plots.

## Repository Structure

```
.
├── smlingif.py     # Shape recovery simulation
├── smfallgif.py       # Falling cube using shape matching
├── pbdfallgif.py      # Falling cube using PBD
├── README.md
├── Cubefallrig.gif    # Generated GIF animations
└── Cubefallpbd.gif               

```

Each Python script can be executed independently.

## Shape Matching Method

Shape matching estimates a transformation between the reference configuration and the current particle positions.

Let $\mathbf{q}_i$ and $\mathbf{x}_i$ denote the reference and current positions of particle $i$, respectively.

### Center of Mass

The mass-weighted centers of mass are computed as

```math
\mathbf{t}_0=\frac{\sum_i m_i\mathbf{q}_i}{\sum_i m_i}
```

```math
\mathbf{t}=\frac{\sum_i m_i\mathbf{x}_i}{\sum_i m_i}
```

The centered particle positions are then used to construct the covariance matrices:

```math
A_{pq}=\sum_i m_i(\mathbf{x}_i-\mathbf{t})(\mathbf{q}_i-\mathbf{t}_0)^T
```

```math
A_{qq}=\sum_i m_i(\mathbf{q}_i-\mathbf{t}_0)(\mathbf{q}_i-\mathbf{t}_0)^T
```

### Rotation and Linear Transformation

The optimal rotational transformation is computed using polar decomposition:

```math
R=A_{pq}(A_{pq}^TA_{pq})^{-1/2}
```

The implementation uses a Jacobi eigenvalue decomposition to compute the symmetric matrix square root, together with the Moore–Penrose pseudoinverse for numerical handling of singular matrices.

The linear transformation is calculated as

```math
A=A_{pq}A_{qq}^{+}
```

where $A_{qq}^{+}$ denotes the Moore–Penrose pseudoinverse.

### Target Positions

The target positions are calculated as

```math
\mathbf{g}_i= \left[\beta A+(1-\beta)R\right] (\mathbf{q}_i-\mathbf{t}_0)+\mathbf{t}
```

where:

- $R$: rotational transformation;
- $A$: linear transformation;
- $\beta$: blending parameter between rotational and linear shape matching.

For $\beta=0$, the target transformation is purely rotational. Increasing $\beta$ introduces a greater contribution from the linear transformation.

## Simulation Methods

### 1. Shape Recovery

**File:** `smlingif.py`

The simulation starts with a unit cube consisting of eight particles. One corner particle is displaced from its reference position.

No external forces are applied. The particles evolve under shape-matching corrections and velocity damping.

The velocity and position updates are

```math
\mathbf{v}_i^{n+1} =d\left[\mathbf{v}_i^n+ \frac{\alpha}{h}(\mathbf{g}_i-\mathbf{x}_i^n)\right]
```

```math
\mathbf{x}_i^{n+1}= \mathbf{x}_i^n+h\mathbf{v}_i^{n+1}
```

where $h$ is the time step, $\alpha$ is the correction coefficient, and $d$ is the velocity damping factor.

Default parameters:

```
h = 1 / 500
alpha = 0.9
beta = 0.9
steps = 800
damping = 0.99
```

The initial displacement of the selected corner particle is

```
x[7] += np.array([0.1, 0.1, 0.1])
```

### 2. Falling Cube Using Shape Matching

**File:** `smfallgif.py`

A cube is constructed from particles distributed across its surface on a 4 × 4 × 4 grid. The cube is initially rotated and translated above a horizontal ground plane.

The simulation includes gravity, shape-matching corrections, velocity damping, and ground collision handling.

The particle velocities are updated using

```math
\mathbf{v}_i^{n+1} =d\left[ \mathbf{v}_i^n+ \frac{\alpha}{h}(\mathbf{g}_i-\mathbf{x}_i^n) +h\mathbf{a}_i \right]
```

followed by position integration.

Ground collision is handled by projecting particles below the ground plane back to $z=0$ and preventing downward velocity for particles in contact.

Default parameters:

```
h = 1 / 300
alpha = 1.0
beta = 0.0
steps = 1200
damping = 0.995
gravity = [0, 0, -9.8]
```

With $\beta=0$, this simulation uses rotational shape-matching targets.

### 3. Falling Cube Using Position-Based Dynamics

**File:** `pbdfallgif.py`

This simulation uses a position-based approach to update particle positions under gravity and geometric constraints.

At each time step:

1. Particle velocities and positions are predicted under gravity.
2. Particles penetrating the ground plane are projected back to $z=0$.
3. Shape-matching target positions are calculated.
4. Particle positions are corrected toward the target configuration.
5. Constraint projection is repeated for a fixed number of iterations.
6. Velocities are reconstructed from the corrected particle positions.

The shape-matching position correction is

```math
\mathbf{x}_i\leftarrow \mathbf{x}_i+\alpha(\mathbf{g}_i-\mathbf{x}_i)
```

Default parameters:

```
h = 1 / 300
alpha = 0.5
beta = 0.5
steps = 1200
pbd_iters = 3
damping = 1.0
gravity = [0, 0, -9.8]
```

Unlike the velocity-based shape-matching simulation, PBD directly modifies predicted particle positions through iterative constraint projection.

## Deformation Evaluation

All three scripts calculate a root-mean-square deformation metric after removing rigid-body translation and rotation.

For each frame, the current and reference configurations are centered at their respective centroids.

An optimal proper rotation is estimated using singular value decomposition (SVD).

The deformation metric is

```math
D(t)= \sqrt{ \frac{1}{N} \sum_{i=1}^{N} \left\| (\mathbf{x}_i-\mathbf{t})- R(\mathbf{q}_i-\mathbf{t}_0) \right\|^2 }
```

where $N$ is the number of particles.

This metric measures the deviation from the reference configuration independently of rigid-body translation and rotation.

A value of zero indicates that the current configuration matches the reference shape up to a proper rigid transformation.

## Simulation Results

### Shape Recovery

Unable to upload due to file size limit

### Falling Cube — Shape Matching



### Falling Cube — PBD



### Deformation Curves

The scripts generate deformation plots showing the evolution of $D(t)$ over simulation steps.

| **Simulation**                | **Output file** |
| ----------------------------- | --------------- |
| Shape Recovery                | `bdc09.png`     |
| Falling Cube — Shape Matching | `Dsmfall.png`   |
| Falling Cube — PBD            | `Dpbdfall.png`  |

These plots can be used to examine deformation recovery and residual deformation during the simulations.

## Requirements

- Python 3
- NumPy
- Matplotlib
- Pillow

Install the dependencies with:

```
pip install numpy matplotlib pillow
```

## Usage

### Run Shape Recovery

```
python "smlingif.py"
```

Generates:

- `Cube_beta_0.9.gif`
- `bdc09.png`

### Run Falling Cube with Shape Matching

```
python smfallgif.py
```

Generates:

- `Cubefallrig.gif`
- `Dsmfall.png`

### Run Falling Cube with PBD

```
python pbdfallgif.py
```

Generates:

- `Cubefallpbd.gif`
- `Dpbdfall.png`

The animation and plot files are saved in the current working directory.

## Model Parameters

| **Parameter** | **Description**                                                    |
| ------------- | ------------------------------------------------------------------ |
| `h`           | Simulation time step                                               |
| `alpha`       | Shape-matching correction strength                                 |
| `beta`        | Blending coefficient between rotational and linear transformations |
| `damping`     | Velocity damping factor                                            |
| `steps`       | Total number of simulation steps                                   |
| `pbd_iters`   | Number of constraint projection iterations in PBD                  |
| `g_vec`       | Gravitational acceleration vector                                  |

## Notes

- All particles are assigned equal masses.
- Ground collision is modeled using a horizontal plane at $z=0$.
- The cube is represented by discrete particles and edges rather than a volumetric finite-element mesh.
- Shape matching uses a custom Jacobi eigenvalue decomposition for the polar decomposition calculation.
- The simulations use fixed time steps and manually selected parameters.
- The collision and integration schemes are simplified implementations intended for numerical experimentation.

## Project Background

This project was developed as part of my master's thesis research on the numerical simulation of deformable objects using Shape Matching and Position-Based Dynamics.

## References

[1] M. Müller, B. Heidelberger, M. Teschner, and M. Gross,\
“Meshless deformations based on shape matching,”\
*ACM Transactions on Graphics*, vol. 24, no. 3, pp. 471–478, 2005.\
DOI: [10.1145/1073204.1073216](https://doi.org/10.1145/1073204.1073216)

[2] M. Müller, B. Heidelberger, M. Hennix, and J. Ratcliff,\
“Position based dynamics,”\
*Journal of Visual Communication and Image Representation*, vol. 18, no. 2, pp. 109–118, 2007.\
DOI: [10.1016/j.jvcir.2007.01.005](https://doi.org/10.1016/j.jvcir.2007.01.005)

## License

The source code in this repository is licensed under the MIT License.
