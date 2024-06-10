Some comments about SciPy's gmres implementation scipy.sparse.linalg.gmres(A, b, x0 = None, restart = None, maxiter = None)

- if restart == None then gmres does not restart.
- if restart is omitted, restart == min(20, n).

- if maxiter != None and restart != None then the total number of iterations is restart*maxiter.
- if maxiter != None and restart == None then the total number of iterations is maxiter
- if maxiter == None --> maxiter == 10*n (At least always plots this) for restart == None and for restart == value

However, to use gmres without restart we can set maxiter = 1 and restart = val.

---------------------------------------------

Stopping criteria:
- Our implementation: relative error $r_r = \frac{||r^{(k)}||}{||r^{(0)}||}$.
- Scipy: rtol (tol before version 1.12.0): $||b - A @ x|| <= max(rtol*||b||, atol)$. Default rtol is 1e-5 and default atol is 0.

---------------------------------------------

All "arbitrary" choices in GMRES algorithm:
- Initial guess $x^{(0)}$.
- Stopping criterion and its tolerance. Common choices are $r_r = \frac{||r^{(k)}||}{||r^{(0)}||} < \epsilon$ or $r_r = \frac{||r^{(k)}||}{||b||} < \epsilon$ when $x^{(0)}=0.$
- Maximum number of iterations.
- Arnoldi algorithm. This algorithm has different versions: Classical Gram-Schmidt, Modified Gram-Schmidt, Householder Arnoldi. Note that varying from the GS algorithm to the Householder one changes the way that we compute the solution in the last line of the algorithm depending on if we save the $v_i$'s or not. See page 173 in "Iterative Methods for Sparse Linear Systems" by Saad.
- Orthogonalization method: we have mentioned different versions of the Arnoldi algorithm. However, there are other choices for the orthogonalization method that can affect numerical stability and convergence behavior. Other alternatives, such as iterative methods like the Lanczos process, are sometimes used.
- The way we minimiseolve $||\beta e_1 - H_{m+1,m}y||_2$: QR factorization and Given's rotations are the most common approaches. In both we transform the optimisation problem into solving a triangular system.
- Choice of the PRECONDITIONER.
- Restart Strategy for Restarted GMRES: In Restarted GMRES, the choice of when to restart the algorithm can impact its performance. This introduces another hyperparameter that needs to be chosen appropriately.
- GMRES Variant-Specific Parameters: Depending on the specific variant of GMRES being used (such as Quasi-GMRES, DQGMRES), there may be additional parameters or variations in the algorithm that need to be considered.

EXTRA IDEAS:
- Ordering: Reordering the system of equations can also improve the performance of GMRES. Techniques like Cuthill-McKee or nested dissection can reduce the fill-in and operation count of the preconditioner. The choice of reordering method is arbitrary.

- Memory management: In implementations of GMRES, decisions regarding memory management can influence performance. This includes considerations such as how much memory to allocate for storing vectors and matrices involved in the computation, especially for large-scale problems.

- Parallelization strategy: For large-scale problems, parallelization can significantly speed up the computation. Choosing an appropriate parallelization strategy, such as domain decomposition or task parallelism, is crucial for efficient execution on parallel architectures.

- Convergence monitoring and reporting: While you mentioned stopping criteria, how you monitor convergence and report results can vary. For instance, you may choose to log residual norms at each iteration, monitor the convergence history, or employ other diagnostic tools to assess convergence behavior.
