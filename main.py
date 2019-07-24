import numpy as np
from scipy.sparse.linalg import spilu
from scipy.sparse.linalg import lgmres
from scipy.sparse.linalg import LinearOperator
import pickle
from scipy.sparse import csc_matrix
import time
import matplotlib.pyplot as plt
import random
import math

##GLOBALS
tol = 1e-6
iterations = 0
fill_step = 1
drop_step = 0.1

pre_its = []
pre_times = []

f = open("a_array_164", "rb")
matrix = pickle.load(f)
f.close()

f = open("b_array_164", "rb")
solution = np.load(f)
f.close()

##Does not come natitvly with scipy
def count_iterations(result):
    global iterations
    iterations += 1

def create_preconditioner(A, fill_factor=10, drop_tol=1e-4):
    iLU = spilu(A.tocsc(), fill_factor=fill_factor, drop_tol=drop_tol)
    iLUx = lambda x: iLU.solve(x)
    P = LinearOperator(matrix.shape, iLUx)
    return P

fill_range = np.arange(1, 10, fill_step)

if __name__ == "__main__":
    for i in fill_range:
        iterations = 0
        try:
            print("Starting with i value of", i) 
            P = create_preconditioner(matrix, fill_factor=i)
            start = time.time()
            (result, info) = lgmres(matrix, solution, tol=tol, atol=tol, M=P, callback=count_iterations)
            elapsed = time.time() - start
            pre_its.append(iterations)
            pre_times.append(elapsed)
            print("Finished drop_step step", i, "in", elapsed, "with", iterations, "iterations")
        except RuntimeError:
            print("Runtime error")
            pre_its.append(np.nan)
            pre_times.append(np.nan)
    plt.plot(fill_range, pre_its)
    plt.show()
    plt.plot(fill_range, pre_times)
    plt.show()