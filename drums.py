import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
from numpy.core.defchararray import index
from numpy.lib.function_base import meshgrid

# Find discretized points which are in the boundary
ngrid = 5
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
    size = int(np.max(array))
    L = np.zeros((size, size))

    '''for k in range(size):
        i, j = np.where(array == k+1)
        i, j = i[0], j[0]
        L[k][k] = 4
        if 0 < i-1 < k '''

    (m, n) = array.shape
    for i in range(m):
        for j in range(n):
            val = int(array[i][j])
            if val != 0:
                L[val-1][val-1] = 4
                for (a, b) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    if 0 <= i + a < m and 0 <= j + b < n:
                        val2 = int(array[i+a][j+b])
                        if val2 != 0:
                            L[val-1][val2-1] = -1
    return L / (h**2)

#x = np.array([0, 0, 2, 2, 3, 2, 1, 1, 0])
#y = np.array([0, 1, 3, 2, 2, 1, 1, 0, 0])

x = np.array([0, 0, 2, 2, 3, 2, 1, 1, 0])
y = np.array([1, 2, 2, 3, 2, 1, 1, 0, 1])

#x = np.array([0, 1, 2, 0])
#y = np.array([0, 2, 0, 0])

xmax = np.max(x)
ymax = np.max(y)

poly = mplPath.Path(np.transpose(np.vstack((x, y))))

dx, dy = np.meshgrid(np.arange(0, xmax + h, h), np.arange(0, ymax + h, h))
points = np.transpose(np.stack((dx, dy)))
interior_points = []
for i in range(xmax*ngrid):
    for j in range(ymax*ngrid):
        if poly.contains_point(points[i,j], radius = h/2):
            interior_points.append(points[i,j])
interior_points = np.transpose(np.array(interior_points))

'''plt.scatter(x, y)
plt.plot(x, y)
plt.scatter(interior_points[0], interior_points[1], 2)
plt.show()'''

# Create a binary grid denoting the points within the boundary
grid = np.zeros((xmax*ngrid, ymax*ngrid))
for k in range(len(interior_points[0])):
    i = int(interior_points[0][k] * ngrid)
    j = int(interior_points[1][k] * ngrid)
    if (i < xmax*ngrid and j < ymax*ngrid):
        grid[i][j] = 1

indexed_grid = index_array(grid)

L = delsq(indexed_grid)
#plt.spy(L)
#plt.show()
print(np.linalg.eigh(L)[0])

'''x1, x2 = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1), indexing='ij')
A_tensor = A(x1, x2)
def L_op(u_vec: np.ndarray) -> np.ndarray:
    u_mat = u_vec.reshape(n - 1, -1)

    dpu = np.stack((
        (np.pad(u_mat, ((0, 1), (1, 0))) - np.pad(u_mat, ((1, 0), (1, 0)))) * n,
        (np.pad(u_mat, ((1, 0), (0, 1))) - np.pad(u_mat, ((1, 0), (1, 0)))) * n,
    ), axis=-1)

    Adpu = np.einsum('ijxy,ijy->ijx', A_tensor[:-1, :-1, :, :], dpu)

    Lu = (Adpu[1:, 1:, 0] - Adpu[:-1, 1:, 0]) * n + (Adpu[1:, 1:, 1] - Adpu[1:, :-1, 1]) * n

    return -Lu.reshape((n - 1) ** 2, 1)'''

