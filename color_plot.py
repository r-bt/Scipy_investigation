import numpy as np
from scipy.sparse.linalg import spilu
from scipy.sparse.linalg import lgmres
from scipy.sparse.linalg import gcrotmk
from scipy.sparse.linalg import LinearOperator
import pickle
import time
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator

tol = 1e-6
fill_factors = [8, 9, 10]
drop_tols = [-1, -2, -3]
av_iterations = 2

#Import our matrices
f = open("100x100", "rb")
A = pickle.load(f)
f.close()

f = open("100x100_solution", "rb")
b = np.load(f)
f.close()

# Get the time for a given fill_factor at drop_tol

def time_solve(drop_tol, fill_factor):
    iLU = spilu(A.tocsc(), fill_factor=fill_factor, drop_tol=10 ** drop_tol)
    iLUx = lambda x: iLU.solve(x)
    P = LinearOperator(A.shape, iLUx)
    ## Solve the equation
    start_time = time.time()
    x, info = gcrotmk(A, b, tol=tol, atol=tol, maxiter=1000, M=P)
    elapsed = time.time() - start_time
    print("Completed Fill factor", fill_factor, "at drop tol", 10 ** drop_tol, "in", elapsed)
    return elapsed

if __name__ == "__main__":
    # s = create_test_vals(1,10,-1,-7,fill_skip=[3,4])
    final = []
    pool = Pool(4)
    for j in range(0, av_iterations):
        res = []
        for i in fill_factors:
            time_solve_partial = partial(time_solve, fill_factor=i)
            row = pool.map(time_solve_partial, drop_tols)
            res.append(row)
        if(j == 0):
            final = res
        else:
            final = np.add(final, res)
    final = np.divide(final, av_iterations)
    print(final)
    #Save these values
    f = open("results", "ab")
    pickle.dump(res, f)
    f.close()
    #Plot these values
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(res, interpolation='nearest', cmap=cm.YlOrRd, extent=[8, 10, -3, -1])
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.colorbar(im)
    fig.savefig("times.png")

