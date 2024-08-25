import numpy as np

def jacobi(A, b, tolerance=1e-6, max_iterations=1000):
    # n : number of equations and unknowns
    n = len(A)
    
    # x : starting guess vector
    x = np.zeros(n)
    D = np.diag(A)
    R = A - np.diag(D)
    i = 1
    final_tolerance = -np.inf
    for _ in range(max_iterations):
        x_new = (b - R @ x) / D
        # Usng l-infinity norm to check for convergence
        given_tolerance = np.linalg.norm(x_new - x, ord=np.inf)
        if given_tolerance < tolerance:
            final_tolerance = given_tolerance
            break
        i += 1
        x = x_new
    return (i, final_tolerance, x)

import numpy as np

def gauss_seidel(A, b, tolerance=1e-6, max_iterations=1000):
    # n : number of equations and unknowns
    n = len(A)
    
    # x : starting guess vector
    x = np.zeros(n)
    i = 1
    final_tolerance = -np.inf
    
    for _ in range(max_iterations):
        # Copy the current solution vector
        x_new = np.copy(x)
        
        for j in range(n):
            # Non-diagonal elements of A
            sum1 = np.dot(A[j, :j], x_new[:j])   # New elements
            sum2 = np.dot(A[j, j+1:], x[j+1:])   # Previous iteration elements
            
            # Update the solution vector
            x_new[j] = (b[j] - sum1 - sum2) / A[j, j]
        
        # using l-infinity norm to check for convergence
        given_tolerance = np.linalg.norm(x_new - x, ord=np.inf)
        if given_tolerance < tolerance:
            final_tolerance = given_tolerance
            break
        
        # update the solution vector
        x = x_new
        i += 1
    
    return (i, final_tolerance, x)
