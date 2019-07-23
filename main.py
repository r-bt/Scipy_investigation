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
times = []
its = []
times_pre = []
its_pre = []
max_fill = 100000
drop_step = 0.00001

f = open("a_array", "rb")
matrix = pickle.load(f)
f.close()

f = open("b_array", "rb")
solution = np.load(f)
f.close()

##Does not come natitvly with scipy
def count_iterations(result):
    global iterations
    iterations += 1

def create_preconditioner(A, fill_factor=10, drop_tol=1e-4):
    iLU = spilu(A, fill_factor=fill_factor, drop_tol=drop_tol)
    # iLUx = lambda x: iLU.solve(x)
    # P = LinearOperator(matrix.shape, iLUx)
    return iLU

# if __name__ == "__main__":
#     for i in range(1, 1000):
#         iterations = 0
#         P = create_preconditioner(matrix, fill_factor=i)
#         start = time.time()
#         (results, info) = lgmres(matrix, solution, tol=tol, atol=tol, M=P, callback=count_iterations)
#         elapsed = time.time() - start
#         times_pre.append(elapsed)
#     plt.plot(range(1,1000), times_pre)
#     plt.show()


## Creates graph of magnitude of values in
mags = []

if __name__ == "__main__":
    for i in range(1, 1000):
        print(i)
        av = []
        P = create_preconditioner(matrix, fill_factor=i)
        inverse = P.L.A.dot(P.U.A)
        identity = matrix.dot(inverse)
        for i in range(1, 500):
            row = random.sample(range(0, identity.shape[0]), 1)
            column = random.sample(range(0, identity.shape[1]), 1)
            av.append(identity[row,column][0])
        mag = math.log10(np.mean(av))
        mags.append(mag)
    plt.plot(range(1, 1000), mags)
    plt.show()