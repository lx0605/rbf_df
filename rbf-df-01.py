import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, lgmres


def generate_nodes(n):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    z = np.linspace(0, 1, n)
    X, Y, Z = np.meshgrid(x, y, z)
    nodes = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    return nodes

def identify_boundary(nodes, tol=1e-5):
    return np.any( (nodes <= tol) | (nodes >= 1 - tol), axis=1 )

# Parameters 
n = 30             # n**3 nodes
stencil_size = 7  # Larger stencil
poly_order = 3      # Cubic polynomials
phi_order = 3       # phi(r) = r^5

# Compute factor and power
factor = phi_order * (phi_order + 1)
power = phi_order - 2

# Generate nodes and boundary flags
nodes = generate_nodes(n)
bndry = identify_boundary(nodes)
interior = ~bndry
N = nodes.shape[0]

# Polynomial terms (1, x, y, z, x², y², z², xy, xz, yz)
m = 20 if poly_order >= 3 else 10  # Adjust based on poly_order
tree = KDTree(nodes)

# Initialize Laplacian matrix
L = lil_matrix((N, N))

for i in np.where(interior)[0]:
    # Get stencil
    _, indices = tree.query(nodes[i], k=stencil_size)
    stencil_nodes = nodes[indices]
    n_stencil = len(indices)
    
    # RBF matrix (phi(r) = r^n)
    A = np.zeros((n_stencil, n_stencil))
    for j in range(n_stencil):
        for k in range(j, n_stencil):
            r = np.linalg.norm(stencil_nodes[j] - stencil_nodes[k])
            A[j, k] = r**phi_order  # phi(r) = r^5
            A[k, j] = A[j, k]  # Symmetry
    
    # Regularization to stabilize the matrix
    A += 1e-9 * np.eye(n_stencil)
    
    # Polynomial matrix (quadratic terms)
    P = np.ones((n_stencil, m))
    P[:, 1:4] = stencil_nodes
    if poly_order >= 2:
        P[:, 4:7] = stencil_nodes**2
        # Cross terms: xy, xz, yz ensure directional coupling
        cross_terms = np.zeros((n_stencil, 3))
        cross_terms[:, 0] = stencil_nodes[:, 0] * stencil_nodes[:, 1]
        cross_terms[:, 1] = stencil_nodes[:, 0] * stencil_nodes[:, 2]
        cross_terms[:, 2] = stencil_nodes[:, 1] * stencil_nodes[:, 2]
        P[:, 7:10] = cross_terms
    
    
    #center = nodes[i]
    center = nodes[i] 
    (x0, y0, z0) = center
    # Laplacian of RBF (∇²φ(r) = factor * r^power)
    b_rbf = np.array([factor * (np.linalg.norm(node - center))**power for node in stencil_nodes])
    
    # Laplacian of polynomials
    b_poly = np.zeros(m)
    if poly_order >= 2:
        b_poly[4:7] = 2.0  # ∇²(x²) = 2, ∇²(y²) = 2, ∇²(z²) = 2 ,all other terms are 0
        if poly_order >= 3:
            # Laplacian of cubic terms (x³, y³, z³)
            b_poly[10] = 6 * x0  # ∇²(x³) = 6x
            b_poly[11] = 6 * y0  # ∇²(y³) = 6y
            b_poly[12] = 6 * z0  # ∇²(z³) = 6z
             # Laplacian of mixed cubic terms (x²y, x²z, xy², y²z, xz², yz²)
            b_poly[13] = 2 * y0  # ∇²(x²y) = 2y
            b_poly[14] = 2 * z0  # ∇²(x²z) = 2z
            b_poly[15] = 2 * x0  # ∇²(xy²) = 2x
            b_poly[16] = 2 * z0  # ∇²(y²z) = 2z
            b_poly[17] = 2 * x0  # ∇²(xz²) = 2x
            b_poly[18] = 2 * y0  # ∇²(yz²) = 2y

    # Solve augmented system
    aug_A = np.block([[A, P], [P.T, np.zeros((m, m))]])
    aug_b = np.hstack([b_rbf, b_poly])
    weights = np.linalg.lstsq(aug_A, aug_b, rcond=1e-6)[0][:n_stencil]
    
    # Assign weights
    L[i, indices] = weights

# Enforce Dirichlet BCs
L[bndry] = 0
L[bndry, bndry] = 1.0
L = L.tocsr()

# Assemble RHS 
u_exact = lambda x: np.sin(np.pi*x[:,0]) * np.sin(np.pi*x[:,1]) * np.sin(np.pi*x[:,2])
f = 3 * np.pi**2 * u_exact(nodes)
rhs = -f.copy()  #  ∇²u = -f → RHS = -f
rhs[bndry] = u_exact(nodes[bndry])

# Solve
# u, _ = lgmres(L, rhs, atol=1e-8, maxiter=1000)
u = spsolve(L, rhs)
error = np.linalg.norm(u - u_exact(nodes)) / np.linalg.norm(u_exact(nodes))
print(f"Relative error: {error:.4e}")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Reshape results to 3D grid
grid_size = (n, n, n)
U = u.reshape(grid_size)
U_exact = u_exact(nodes).reshape(grid_size)
error_3d = np.abs(U - U_exact)

# Take a slice at z = 0.5 (middle layer)
iz = n // 2
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y)

# Extract 2D slices
U_slice = U[:, :, iz]
U_exact_slice = U_exact[:, :, iz]
error_slice = error_3d[:, :, iz]

# Plot results
fig = plt.figure(figsize=(15, 5))

# Numerical solution
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(X, Y, U_slice, cmap='viridis')
ax1.set_title('Numerical Solution (z=0.5)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
fig.colorbar(surf1, ax=ax1, shrink=0.5)

# Exact solution
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(X, Y, U_exact_slice, cmap='viridis')
ax2.set_title('Exact Solution (z=0.5)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
fig.colorbar(surf2, ax=ax2, shrink=0.5)

# Error
ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(X, Y, error_slice, cmap='hot')
ax3.set_title('Absolute Error (z=0.5)')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
fig.colorbar(surf3, ax=ax3, shrink=0.5)

plt.tight_layout()
plt.show()