import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
from scipy.sparse import csr_matrix
from numpy.core.defchararray import index
from numpy.lib.function_base import meshgrid

start = time.time()

# Density of points in approximation of region
ngrid = 32
h = 1 / ngrid

def index_array(array):
    '''Takes in a binary array describing a region and indexes its points to represent vectorized functions'''
    array = np.copy(array)
    counter = 1
    (m, n) = array.shape
    for i in range(m):
        for j in range(n):
            if array[i][j] != 0:
                array[i][j] = counter
                counter += 1
    return array

def delsq(array):
    '''Takes an indexed array representing a region and returns the discretized laplacian operator for 
        vectorized functions in that region'''
    rows = []
    cols = []
    data = []
    (m, n) = array.shape
    
    for i in range(m):
        for j in range(n):
            val = int(array[i][j])
            if val != 0:
                rows.append(val-1)
                cols.append(val-1)
                data.append(4)
                for (a, b) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    if 0 <= i + a < m and 0 <= j + b < n:
                        val2 = int(array[i+a][j+b])
                        if val2 != 0:
                            rows.append(val-1)
                            cols.append(val2-1)
                            data.append(-1)

    size = int(np.max(array))
    return csr_matrix((data, (rows, cols)), shape=(size, size)).toarray() / (h**2)

# First isospectral region
x = np.array([0, 0, 2, 2, 3, 2, 1, 1, 0])
y = np.array([0, 1, 3, 2, 2, 1, 1, 0, 0])

# Second isospectral region
#x = np.array([0, 0, 2, 2, 3, 2, 1, 1, 0])
#y = np.array([1, 2, 2, 3, 2, 1, 1, 0, 1])

# Triangular region
#x = np.array([0, 1, 2, 0])
#y = np.array([0, 2, 0, 0])

# Square region
#x = np.array([0, 0, 1, 1, 0])
#y = np.array([0, 1, 1, 0, 0])

# Create a lattice of points spaced by h
dx, dy = np.meshgrid(np.arange(0, np.max(x) + h, h), np.arange(0, np.max(y) + h, h))
points = np.transpose(np.stack((dx, dy)))

# Check which lattice points are strictly within the boundary given by x, y
interior_points = []
poly = mplPath.Path(np.transpose(np.vstack((x, y))))
for i in range(np.max(x)*ngrid):
    for j in range(np.max(y)*ngrid):
        if poly.contains_point(points[i,j], radius = h/2):
            interior_points.append(points[i,j])
interior_points = np.transpose(np.array(interior_points))

# Create a binary grid denoting the points within the boundary
grid = np.zeros((np.max(x)*ngrid, np.max(y)*ngrid))
for k in range(len(interior_points[0])):
    i = int(interior_points[0][k] * ngrid)
    j = int(interior_points[1][k] * ngrid)
    if (i < np.max(x)*ngrid and j < np.max(y)*ngrid):
        grid[i][j] = 1

# Index grid for vectorized functions
indexed_grid = index_array(grid)

# Create discrete Laplacian from indexed grid
L = delsq(indexed_grid)

#plt.spy(L)
#plt.show()

print(np.linalg.eigh(L)[0][:10])
print(time.time() - start)