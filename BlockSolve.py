import numpy as np
from scipy.sparse.linalg import spilu
from scipy.sparse.linalg import gcrotmk
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import LinAlgError
import pickle
from scipy.sparse import csc_matrix
from scipy.linalg import inv
from scipy.sparse.linalg import inv as sparse_inv
import matplotlib.pyplot as plt
import scipy.sparse

f = open("32x32", "rb")
A = pickle.load(f)
f.close()

f = open("32x32_solution", "rb")
b = np.load(f)
f.close()


class BlockSolve():
    def __init__(self, array, diagonal_deviation):
        self.array = array
        self.blocks = []
        self.info = []
        self.diagonal_deviation = diagonal_deviation
        self.P = np.identity(self.array.shape[0])

    def __rec_to_squares(self, rec):
        squares = []
        divisor = np.gcd(rec.shape[0], rec.shape[1])
        # Divide the first axis and then the second axis
        f_div = np.split(rec, rec.shape[0] / divisor)
        for arr in f_div:
            squares.extend(np.split(arr, rec.shape[1] / divisor, axis=1))
        return squares

    def __add_inverse(self, block, top_row, first_element):
        try:
            inverse = inv(block)
            for i, row in enumerate(range(top_row, top_row + block.shape[0])):
                for j, element in enumerate(range(first_element, first_element + block.shape[1])):
                    self.P[row, element] = inverse[i, j]
        except LinAlgError:
            raise scipy.linalg.LinAlgError("Matrix singular")

    def is_bottom_left(self, row, element):
        if (self.array[row, element] == 0):
            return False
        left_slot = self.array[row, element - 1] if element - 1 >= 0 else 0
        right_slot = self.array[row, element + 1] if element + 1 < self.array.shape[1] else 0
        bottom_slot = self.array[row + 1, element] if row + 1 < self.array.shape[0] else 0
        if (left_slot == 0 and bottom_slot == 0 and right_slot != 0):
            return True
        return False

    def is_top_right(self, row, element):
        if (self.array[row, element] == 0):
            return False
        right_slot = self.array[row, element + 1] if element + 1 < self.array.shape[1] else 0
        top_slot = self.array[row - 1, element] if row - 1 >= 0 else 0
        bottom_slot = self.array[row + 1, element] if row + 1 < self.array.shape[0] else 0
        if (right_slot == 0 and top_slot == 0 and bottom_slot != 0):
            return True
        return False

    def top_right_search(self, row, element, row_increment=1, element_increment=1):
        row, element = row + row_increment, element + element_increment
        while row < (self.array.shape[0]) and row >= 0 and element < (self.array.shape[1]) and element >= 0:
            if (self.is_top_right(row, element)):
                return [row, element]
            row += row_increment
            element += element_increment
            if (self.array[row, element] == 0):
                ##We're gone outside the boundarys of the block....most likly have rectangular block. Need to look at across and up
                if (self.array[row, element - element_increment] != 0):
                    # We have a vertical rectangular
                    element -= element_increment
                    element_increment = 0
                elif (self.array[row - row_increment, element] != 0):
                    # We have a horizontal rectangle
                    row -= row_increment
                    row_increment = 0
                if (row_increment == 0 and element_increment == 0):
                    return False
        return False

    def get_block(self, top_left_row, top_left_element, row_extent, element_extent):
        block = []
        for i, row in enumerate(range(top_left_row, row_extent + 1)):
            block.append([])
            for element in range(top_left_element, element_extent + 1):
                val = self.array[row, element]
                block[i].append(val)
        return np.array(block)

    def find(self):
        for row in range(0, self.array.shape[0]):
            for element in self.array[row].indices:
                if (element < row + self.diagonal_deviation and element > row - self.diagonal_deviation):
                    if (self.is_bottom_left(row, element)):
                        top_right = self.top_right_search(row, element, row_increment=-1)
                        if (top_right != False):
                            top_left = [top_right[0], element]
                            block = self.get_block(top_right[0], element, row, top_right[1])
                            size = block.shape[0] * block.shape[1]
                            self.blocks.append([top_left, block])
                            self.info.append([len(self.blocks) - 1, size])
        self.info = np.array(self.info)
        self.info = self.info[self.info[:, 1].argsort()]

    #         return [np.array(self.info), np.array(self.blocks)]

    def form_preconditioner(self):
        if (len(self.info) != 0):
            for index in self.info[:, 0]:
                block = self.blocks[index]
                block_top_row = block[0][0]
                block_first_element = block[0][1]
                if (block[1].shape[0] != block[1].shape[1]):
                    # Inverse function only works on square matrices so have to divide rectangles into squares
                    for square in self.__rec_to_squares(block[1]):
                        try:
                            self.__add_inverse(square, block_top_row, block_first_element)
                        except LinAlgError:
                            print("Matrix Signular")
                else:
                    try:
                        self.__add_inverse(block[1], block_top_row, block_first_element)
                    except LinAlgError:
                        print("Matrix Signular")
        else:
            raise ValueError("Please run BlockSolve.find() first!")

# Setup the block solver

solver = BlockSolve(A, 50)

solver.find()

solver.form_preconditioner()

P_sparse = scipy.sparse.csr_matrix(solver.P)

# Testing against the ILU

iLU = spilu(scipy.sparse.csc_matrix(A), fill_factor=1, drop_tol=0)
iLUx = lambda x: iLU.solve(x)
iLU_P = LinearOperator(A.shape, iLUx)

ILU_product = iLU_P.dot(A.todense())

# Form a product

product = A.dot(solver.P)

# Get the condition number

R_product = np.linalg.cond(product)

ILU_product = np.linalg.cond(ILU_product)

print("R_product was {} and ILU product was {}, therefore R_product is {} better than ILU_product".format(R_product, ILU_product, ILU_product / R_product))