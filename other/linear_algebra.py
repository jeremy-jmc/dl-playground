# https://towardsdatascience.com/introduction-to-linear-algebra-with-numpy-79adeb7bc060
# https://medium.com/@nishithakalathil/advanced-numpy-linear-algebra-and-more-734c6eac4b9c
# https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/pages/assignments/
# http://ocw.mit.edu/18-06SCF11

"""
A 2-dimensional array has two corresponding axes: 
    the first running vertically downwards across rows (axis 0), 
    and the second running horizontally across columns (axis 1).
"""

import numpy as np
np.random.seed(42)
# https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
# https://math.stackexchange.com/questions/20412/element-wise-or-pointwise-operations-notation

# * 2D Matrix Multiplication
# Create two 2-D matrices
matrix_a = np.random.randint(low=0, high=4, size=(3, 2))
matrix_b = np.random.randint(low=0, high=4, size=(2, 3))
matrix_b_t = np.transpose(matrix_b)

# Matrix multiplication for 2-D matrices
result_matrix = np.matmul(matrix_a, matrix_b)

print("Matrix A (2-D):")
print(matrix_a)
print("\nMatrix B (2-D):")
print(matrix_b)
print("\nResult of Matrix Multiplication (2-D):")
print(result_matrix)
print("\nMatrix B Transpose (2-D):")
print(matrix_b_t)
print("\nResult of Haddamard Product (2-D):")
print(matrix_a * matrix_b_t)


# * 3D Matrix Multiplication
# https://math.stackexchange.com/questions/63074/is-there-a-3-dimensional-matrix-by-matrix-product
# tensor contraction

# Create two 3-D matrices
matrix_3d_a = np.random.randint(low=0, high=4, size=(2, 3, 4))  # (3, 3, 2)
matrix_3d_b = np.random.randint(low=0, high=4, size=(2, 4, 3))  # (3, 2, 4)
matrix_3d_b_t = np.transpose(matrix_3d_b, axes=(0, 2, 1))

# Matrix multiplication for 3-D matrices
result_matrix_3d = np.matmul(matrix_3d_a, matrix_3d_b)

# TODO: resolve by hand
print("Matrix A (3-D):")
print(matrix_3d_a)
print("\nMatrix B (3-D):")
print(matrix_3d_b)
print("\nResult of Matrix Multiplication (3-D):")
print(result_matrix_3d)
print("\nMatrix B Transpose (3-D):")
print(matrix_3d_b_t)
print("\nResult of Haddamard Product (3-D):")
print(matrix_3d_a * matrix_3d_b_t)

# TODO: 4-D matrix multiplication
# https://stackoverflow.com/questions/47752324/matrix-multiplication-on-4d-numpy-arrays
# https://stackoverflow.com/questions/41870228/understanding-tensordot
a = np.array([[1, 7], [4, 3]]) 
b = np.array([[2, 9], [4, 5]]) 
c = np.array([[3, 6], [1, 0]]) 
d = np.array([[2, 8], [1, 2]]) 
e = np.array([[0, 0], [1, 2]])
f = np.array([[2, 8], [1, 0]])

m = np.array([[a, b], [c, d]])              # (2,2,2,2)
n = np.array([[e, f, a], [b, d, c]])        # (2,3,2,2)


print("Matrix M (4-D):")
print(m)
print("\nMatrix N (4-D):")
print(n)
print("\nResult of Matrix Multiplication (4-D):")
print(np.tensordot(m,n, axes=((1,3),(0,2))).swapaxes(1,2))
print(np.tensordot(n,m, axes=((0,2),(1,3))).transpose(2,0,3,1))
print(np.einsum('ijkl,jmln->imkn', m, n))


# TODO: einsum
# https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
# https://ajcr.net/Basic-guide-to-einsum/
# https://medium.com/artificialis/einsteins-summation-in-deep-learning-for-making-your-life-easier-7b3c44e51c42

"""
Repeating letters between input arrays means that values along those axes will be multiplied together. 
The products make up the values for the output array.

Omitting a letter from the output means that values along that axis will be summed.

We can return the unsummed axes in any order we like.
"""
A = np.array([0, 1, 2])

B = np.array([[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]])

print('\nEinsum:')
print((A[:, np.newaxis] * B).sum(axis=1))
print(np.einsum('i,ij->i', A, B))


A = np.array([[1, 1, 1],
              [2, 2, 2],
              [5, 5, 5]])

B = np.array([[0, 1, 0],
              [1, 1, 0],
              [1, 1, 1]])
#  if we gave no output labels but just write the arrow, we’d simply sum the whole array
print(np.einsum('ij,ij->', A, B))
print(np.einsum('ij,ij', A, B))

# if we leave out the arrow '->', NumPy will take the labels that appeared once and arrange them in alphabetical order
print(np.einsum('ij,jk', A, B))
print(np.einsum('ij,ij->i', A, B))
print(np.einsum('ij,jk->ik', A, B))
# if we want to control what our output looked like we can choose the order of the output labels ourself.
print(np.einsum('ij,jk->ki', A, B))
print(np.einsum('ij,jk->ijk', A, B))

# TODO: eigenvalues and eigenvectors

# TODO: SVD
# https://vikraantpai.medium.com/exploring-singular-value-decomposition-svd-from-scratch-in-python-a866774b7190
# https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/
# https://www.geeksforgeeks.org/singular-value-decomposition-svd/
# https://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm


# TODO: PCA
# https://towardsdatascience.com/principal-component-analysis-pca-from-scratch-in-python-7f3e2a540c51
# https://kozodoi.me/blog/20230326/pca-from-scratch
# https://www.python-engineer.com/courses/mlfromscratch/11_pca/
# https://www.askpython.com/python/examples/principal-component-analysis
