# svd_compressor/svd.py

"""
Contains the 'from scratch' SVD implementation using Power Iteration.

Note: This implementation is for educational purposes. It's less
stable and much slower than optimized library functions like numpy.linalg.svd.
It relies on matrix deflation, which can accumulate errors.
"""

import numpy as np
from numpy.linalg import norm
from math import sqrt
from random import normalvariate

def _generate_unit_vector(n: int) -> np.ndarray:
    """Generates a random n-dimensional unit vector."""
    not_normalized = [normalvariate(0, 1) for _ in range(n)]
    norm_val = sqrt(sum(x * x for x in not_normalized))
    # Handle potential zero vector if extremely unlucky
    if norm_val < 1e-15:
        return np.zeros(n)
    normalized = np.array([x / norm_val for x in not_normalized])
    return normalized

def _power_iteration(matrix: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Finds the dominant singular vector of A.T @ A or A @ A.T
    using the power iteration method.
    """
    n, m = matrix.shape

    # Start with a random unit vector
    # Choose dimension based on which matrix product is smaller
    vec_size = m if n > m else n
    current_vec = _generate_unit_vector(vec_size)

    # Determine which matrix product to use
    if n > m:
        # Find dominant eigenvector of A.T @ A (gives v)
        mat_product = matrix.T @ matrix
    else:
        # Find dominant eigenvector of A @ A.T (gives u)
        mat_product = matrix @ matrix.T

    iterations = 0
    max_iterations = 1000 # Safety break

    while iterations < max_iterations:
        iterations += 1
        last_vec = current_vec
        current_vec = mat_product @ last_vec
        current_vec = current_vec / norm(current_vec)

        # Check for convergence
        if abs(np.dot(current_vec, last_vec)) > 1 - epsilon:
            # print(f"Power iteration converged in {iterations} iterations.")
            return current_vec

    print("Warning: Power iteration did not converge within max_iterations.")
    return current_vec


def power_iteration_svd(matrix: np.ndarray, k: int = None, epsilon: float = 1e-10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs Singular Value Decomposition using the Power Iteration method
    combined with matrix deflation.

    Args:
        matrix: The input matrix (2D NumPy array).
        k: The number of singular values/vectors to compute. Defaults to min(n, m).
        epsilon: Convergence threshold for power iteration.

    Returns:
        A tuple (U, s, Vh):
        - U: Unitary matrix having left singular vectors as columns. (n x k)
        - s: The singular values, sorted in descending order. (k,)
        - Vh: Unitary matrix having right singular vectors as rows. (k x m)
    """
    matrix = np.array(matrix, dtype=float)
    n, m = matrix.shape
    
    if k is None:
        k = min(n, m)
    k = min(k, min(n, m)) # Cannot compute more than rank

    singular_values = []
    left_vectors = []
    right_vectors = []

    matrix_copy = matrix.copy()

    for i in range(k):
        if n > m:
            # Power iteration on A.T @ A gives v
            v_i = _power_iteration(matrix_copy, epsilon=epsilon)
            u_i_unnormalized = matrix_copy @ v_i
            sigma_i = norm(u_i_unnormalized)
            # Handle potential zero singular value
            u_i = u_i_unnormalized / sigma_i if sigma_i > epsilon else np.zeros(n)
        else:
            # Power iteration on A @ A.T gives u
            u_i = _power_iteration(matrix_copy, epsilon=epsilon)
            v_i_unnormalized = matrix_copy.T @ u_i
            sigma_i = norm(v_i_unnormalized)
            # Handle potential zero singular value
            v_i = v_i_unnormalized / sigma_i if sigma_i > epsilon else np.zeros(m)

        # Check for near-zero singular value and stop if necessary
        if sigma_i < epsilon:
            # print(f"Stopping early at k={i} due to near-zero singular value.")
            break
        
        singular_values.append(sigma_i)
        left_vectors.append(u_i)
        right_vectors.append(v_i)

        # Deflate the matrix: remove the component corresponding to this singular value
        matrix_copy -= sigma_i * np.outer(u_i, v_i)

    # Convert lists to NumPy arrays
    U = np.array(left_vectors).T  # (n x computed_k)
    s = np.array(singular_values) # (computed_k,)
    Vh = np.array(right_vectors) # (computed_k x m)

    return U, s, Vh
