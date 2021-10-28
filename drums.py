import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from numpy.core.defchararray import index
from numpy.lib.function_base import meshgrid

# Beamer Template: https://www.overleaf.com/latex/templates/template-beamer-ufc/rvqwnmszpsvf

# Density of points in approximation of region
# ngrid = 32
# h = 1 / ngrid

def index_array(array):
    '''Takes in a binary array describing a region and indexes its points to represent
        vectorized functions'''
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
    '''Takes an indexed array representing a region and returns the discretized laplacian operator
        for vectorized functions in that region'''
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
    return csr_matrix((data, (rows, cols)), shape=(size, size)).toarray()

def undelsq(func, array):
    '''Takes a vectorized function and an indexed array and returns a 2D array
        representing the function'''
    (m, n) = array.shape
    temp = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if array[i][j] != 0:
                temp[i][j] = func[int(array[i][j]-1)]
    return temp

class Solver:
    def __init__(self, x, y, ngrid = 16):
        ''' x : an np.array of x coordinates describing a boundary
            y : an np.array of y coordinates describing a boundary
            ngrid : density of points in the discrete approximation of the region'''

        self.x = x
        self.y = y
        self.ngrid = ngrid
        self.h = 1/ngrid
        self.indexed_grid = []
        self.eigvals = []
        self.eigvecs = []

    def solve(self, k = 20):
        '''Finds the first k eigenvalues and eigenvectors of the negative
            laplacian on the boundary described by x and y'''

        # Create a lattice of points spaced by h
        dx, dy = np.meshgrid(np.arange(0, np.max(self.x) + self.h, self.h), np.arange(0, np.max(self.y) + self.h, self.h))
        points = np.transpose(np.stack((dx, dy)))

        # Check which lattice points are strictly within the boundary given by x, y
        interior_points = []
        poly = mplPath.Path(np.transpose(np.vstack((self.x, self.y))))
        for i in range(int(np.max(self.x)*self.ngrid)):
            for j in range(int(np.max(self.y)*self.ngrid)):
                if poly.contains_point(points[i,j], radius = self.h/2):
                    interior_points.append(points[i,j])
        interior_points = np.transpose(np.array(interior_points))

        plt.plot(self.x, self.y)
        plt.scatter(interior_points[0], interior_points[1], s = 1)
        plt.show()

        # Create a binary grid denoting the points within the boundary
        grid = np.zeros((int(np.max(self.x)*self.ngrid), int(np.max(self.y)*self.ngrid)))
        for p in range(len(interior_points[0])):
            i = int(interior_points[0][p] * self.ngrid)
            j = int(interior_points[1][p] * self.ngrid)
            if (i < np.max(self.x)*self.ngrid and j < np.max(self.y)*self.ngrid):
                grid[i][j] = 1

        # Index grid for vectorized functions
        self.indexed_grid = index_array(grid)

        # Create discrete Laplacian from indexed grid
        L = delsq(self.indexed_grid) / (self.h**2)

        self.eigvals, self.eigvecs = eigsh(L, k = k, which = 'SM')
        self.eigvecs = np.transpose(self.eigvecs)
    
    def show_eigvec(self, n = 0):
        '''Display the nth eigenvector'''
        plt.imshow(undelsq(self.eigvecs[n], self.indexed_grid), interpolation = 'none')
        plt.show()

if __name__ == '__main__':
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
    # size = 2
    #x = np.array([0, 0, size, size, 0])
    #y = np.array([0, size, size, 0, 0])

    start = time.time()
    s = Solver(x, y, ngrid = 8)
    s.solve()
    print(time.time() - start)

    #np.set_printoptions(threshold = np.inf)
    #print(s.eigvecs[0])
    #print(s.eigvecs[1])
    #print(s.eigvecs[2])

    s.show_eigvec(0)
    s.show_eigvec(1)
    s.show_eigvec(2)