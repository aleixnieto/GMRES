import numpy as np
from time import time

def arnoldi(A, b, r0, m, tol = 1e-12):
    """
    This function computes an orthonormal basis V_{m+1} = {v_1,...,v_{m+1}} of 
    K_{m+1}(A, r^{(0)}) = span{r^{(0)}, Ar^{(0)}, ..., A^{m}r^{(0)}}.

    Input parameters:
    -----------------
      A: array_like
          An (n x n) array.
      
      b: array_like
          Initial vector of length n.
      
      r0: array_like 
          Initial residual of length n.

      m: int
          One less than the dimension of the Krylov subspace. Must be > 0.
      
      tol: float, optional
          Tolerance for convergence.

    Output:
    -------
      Q: numpy.array 
          n x m array, the columns are an orthonormal basis of the Krylov subspace.
      
      H: numpy.array
          An (m + 1) x m array. It is the matrix A on basis Q. It is upper Hessenberg.
    """
    
    # Check inputs
    n = A.shape[0]
    assert A.shape == (n, n) and b.shape == (n,) and r0.shape == (n,), "Matrix and vector dimensions don not match"
    assert isinstance(m, int) and m >= 0, "m must be a positive integer"
    
    m = min(m, n)
    
    # Initialize matrices
    V = np.zeros((n, m + 1))
    H = np.zeros((m + 1, m))
    
    # Normalize input vector and use for Krylov vector
    beta = np.linalg.norm(r0)
    V[:, 0] = r0 / beta

    for k in range(1, m + 1):
        # Generate a new candidate vector
        w = A @ V[:, k - 1] # Note that here is different from arnoldi_one_iter as we iter over k from 1 to m. 
                            # In arnoldi_one_iter we have k as inputo to the function and here we have V[:, k - 1] as k starts at 1.
        
        # Orthogonalization
        for j in range(k):
            H[j, k - 1] = V[:, j] @ w
            w -= H[j, k - 1] * V[:, j]
        
        H[k, k - 1] = np.linalg.norm(w)

        # Check convergence
        if H[k, k - 1] <= tol:
            print(f"Converged in {k} iterations.")
            return V, H
        
        # Normalize and store the new basis vector
        V[:, k] = w / H[k, k - 1]
    
    return V, H

def arnoldi_one_iter(A, V, k, tol=1e-12):
    """
    Computes the new vectors of the Arnoldi iteration for both V_{k+1} and H_{k + 1, k}

    Input parameters:
    -----------------
    A : array_like
        An (n x n) array representing the matrix for which we are building the Krylov subspace.
          
    V : array_like
        An (n x (k + 1)) array. The current Krylov orthonormal basis.
      
    k : int
        One less than the step we are obtaining in the Arnoldi's algorithm to increase
        the dimension of the Krylov subspace. Must be >= 0.
      
    tol : float, optional
        Tolerance for convergence.
    
    Output:
    -------
    h_k : ndarray
        A (k + 2,) array representing the k-th column of the Hessenberg matrix H.
          
    v_new : ndarray or None
        A (n,) array representing the new orthonormal basis vector. 
        If the algorithm converges (h_k[k + 1] <= tol), returns None.
    """
    
    # Initialize k + 2 nonzero elements of H along column k.
    h_k = np.zeros((k + 2,))

    # Compute the matrix-vector product A * v_k
    v_new = A @ V[:, k]
    
    # Calculate first k+1 elements of the k-th Hessenberg column
    for j in range(k + 1):
        h_k[j] = v_new @ V[:, j]
        v_new -= h_k[j] * V[:, j]
    
    # Calculate the (k+1)-th element of the k-th Hessenberg column
    h_k[k + 1] = np.linalg.norm(v_new)

    if h_k[k + 1] <= tol:
        # Early termination if convergence is achieved (return None for v_new)
        return h_k, None
    else:
        # Normalize the new orthogonal vector to get the new basis vector
        v_new /= h_k[k + 1]

    return h_k, v_new

    
def back_substitution(A, b):
    """
    Solve a linear system using back substitution.
    
    Args:
    ----------
        A: list of lists
            Coefficient matrix (must be upper triangular).
        
        b: list
            Column vector of constants.
    
    Returns:
    --------
        list: Solution vector.
        
    Raises:
        ValueError: If the matrix A is not square or if its dimensions are incompatible with the vector b.
    """
    
    n = len(b)
    
    # Check if A is a square matrix
    if len(A) != n or any(len(row) != n for row in A):
        raise ValueError("Matrix A must be square.")
    
    # Check if dimensions of A and b are compatible
    if len(A) != len(b):
        raise ValueError("Dimensions of A and b are incompatible.")
    
    x = [0] * n
    
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - sum(A[i][j] * x[j] for j in range(i + 1, n))) / A[i][i]
    
    return x