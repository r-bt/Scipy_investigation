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
import cmocean as cmo

tol = 1e-6
fill_factors = [1, 2, 5, 6, 7, 8, 9, 10]
drop_tols = [-1,-2,-3,-4,-5,-6,-7]

size = "100"

av_iterations = 1

#Import our matrices
f = open("{}x{}".format(size, size), "rb")
A = pickle.load(f)
f.close()

f = open("{}x{}_solution".format(size, size), "rb")
b = np.load(f)
f.close()

# Get the time for a given fill_factor at drop_tol

def time_solve(drop_tol, fill_factor):
    try:
        iLU = spilu(A.tocsc(), fill_factor=fill_factor, drop_tol=10 ** drop_tol)
    except:
        print("Runtime error")
        return np.nan
    iLUx = lambda x: iLU.solve(x)
    P = LinearOperator(A.shape, iLUx)
    ## Solve the equation
    start_time = time.time()
    x, info = gcrotmk(A, b, tol=tol, atol=tol, maxiter=1000, M=P)
    elapsed = time.time() - start_time
    print("Completed Fill factor", fill_factor, "at drop tol", 10 ** drop_tol, "in", elapsed)
    return elapsed

def plot(matrix, name="times.png"):
    flipped = np.flip(matrix)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(np.log10(flipped), interpolation='nearest', cmap=cmo.cm.deep, extent=[-8, -1, 1, 9])
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Drop tolerance (log10)")
    ax.set_ylabel("Filter factor")
    plt.yticks([1, 2, 3, 4, 5, 6, 7, 8, 9,], ["1", "2", "5", "6", "7", "8", "9", "10", "11"])
    plt.title("151000 x 151000 matrix solve time")
    cbr = fig.colorbar(im)
    cbr.set_label("log10(time)")
    fig.savefig(name)

if __name__ == "__main__":
    multi_start = time.time()
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
    f = open("results_{}x{}_grid".format(size), "ab")
    pickle.dump(final, f)
    f.close()
    # Plot these values
    # f = open("results_25x25_grid", "rb")
    # final = pickle.load(f)
    # f.close()
    plot(final, name="{}x{}_grid.png".format(size))
    print("Took only", time.time() - multi_start)




