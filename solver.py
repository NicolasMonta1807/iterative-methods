import sys
import numpy as np
from iterative import jacobi, gauss_seidel

def read_problem(file):
  try:
    with open(file, 'r') as f:
        lines = f.readlines()

    # Initialize variables
    A = []
    b = []
    tol = None
    max_iter = None
    
    # Read flag variables
    reading_A = False
    reading_b = False
    
    for line in lines:
        line = line.strip()
        
        if line == 'A':
            reading_A = True
            reading_b = False
            continue
        elif line == 'b':
            reading_A = False
            reading_b = True
            continue
        elif line.startswith('tol'):
            reading_A = False
            reading_b = False
            tol = float(line.split(",")[1])  # Read tolerance
        elif line.startswith('max'):
            reading_A = False
            reading_b = False
            max_iter = int(line.split(",")[1])  # Read max iterations
            
        if reading_A:
            A.append([float(num) for num in line.split(',')])  # Read matrix A
        elif reading_b:
            b = [float(num) for num in line.split(',')]  # Read vector b
    
    # Convert to numpy arrays
    A = np.array(A)
    b = np.array(b)
    
    # Return variables
    return A, b, tol, max_iter
  except FileNotFoundError:
    print(f"Error: File {file} not found")
    sys.exit(1)
    

def main():
    if len(sys.argv) < 2:
        print("Usage: python jacobi.py <input_file>")
        sys.exit(1)
        
    verbose = False
    if len(sys.argv) == 3 and sys.argv[2] == '--verbose':
        verbose = True
    
    input_file = sys.argv[1]
    A, b, tolerance, max_iterations = read_problem(input_file)
    
    print("------------------------")
    print(f"Solving: \n {A} \n")
    print(f"For: {b}")
    print("------------------------")
    
    
    iterations, found_tolerance, x = jacobi(A, b, tolerance, max_iterations, verbose)
    
    print("------------------------ JACOBI ------------------------")
    print(f"Number of iterations: {iterations}")
    print(f"Solution vector: \n {x.reshape(-1,1)}")
    print(f"Solution tolerance: {found_tolerance:e}")
    print("--------------------------------------------------------\n")
    
    iterations, found_tolerance, x = gauss_seidel(A, b, tolerance, max_iterations, verbose)
    print("--------------------- GAUSS-SEIDEL ---------------------")
    print(f"Number of iterations: {iterations}")
    print(f"Solution vector: \n {x.reshape(-1,1)}")
    print(f"Solution tolerance: {found_tolerance:e}")
    print("--------------------------------------------------------")
    
    
    
if __name__ == '__main__':
    main()
