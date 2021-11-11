import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import matplotlib.animation as animation
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from numpy.core.defchararray import index
from numpy.lib.function_base import meshgrid

import sounddevice as sd

# Beamer Template: https://www.overleaf.com/latex/templates/template-beamer-ufc/rvqwnmszpsvf

fps = 30
fs = 48000

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

def vectorize(func, array):
    '''Takes a 2D array representing a function and returns the vectorized
        version of the function'''
    (m, n) = array.shape
    temp = np.arange(np.max(array))
    for i in range(m):
        for j in range(n):
            if array[i][j] != 0:
                temp[int(array[i][j]) - 1] = func[i][j]
    return temp

def unvectorize(func, array):
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
    def __init__(self, x, y, ngrid = 16, alpha = 1):
        ''' x : an np.array of x coordinates describing a boundary
            y : an np.array of y coordinates describing a boundary
            ngrid : density of points in the discrete approximation of the region'''

        self.x = x
        self.y = y
        self.alpha = alpha
        self.ngrid = ngrid
        self.h = 1 / ngrid
        self.grid = [] # Boolean grid
        self.indexed_grid = []
        self.eigvals = []
        self.eigvecs = []
        self.consts = []
        self.vmin, self.vmax = -0.05, 0.05

    def get_eigs(self, k = 20):
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

        # Save the binary array representing the boundary
        self.grid = grid

        # Index grid for vectorized functions
        self.indexed_grid = index_array(grid)

        # Create discrete Laplacian from indexed grid
        L = delsq(self.indexed_grid) / (self.h**2)

        self.eigvals, self.eigvecs = eigsh(L, k = k, which = 'SM')
        self.eigvecs = np.transpose(self.eigvecs)
    
    def show_eigvec(self, n = 0):
        '''Display the nth eigenvector'''
        plt.imshow(unvectorize(self.eigvecs[n], self.indexed_grid), interpolation = 'none')
        plt.show()

    def create_gaussian(self, a, b, sigma = 1, mu = 0):
        '''Creates a vectorized gaussian centered around (a, b)'''
        (m, n) = self.grid.shape
        x, y = np.meshgrid(np.linspace(0,np.max(self.x), m), np.linspace(0,np.max(self.y), n))
        d = np.sqrt((x - a)**2 + (y - b)**2)
        gauss = np.exp(-((d - mu)**2 / (2 * sigma**2)))
        return np.multiply(self.grid, np.transpose(gauss))

    def calc_consts(self, f):
        '''Calculate the constants ci for a solution given u(0,t) = 0 and u_t(0,t) = f'''
        func = vectorize(f, self.indexed_grid)
        for i in range(len(self.eigvecs)):
            phi = self.eigvecs[i]
            lam = self.eigvals[i]
            self.consts.append(np.dot(func, phi) / (self.alpha * np.sqrt(lam * np.dot(phi, phi))))
    
    def get_color_bounds(self):
        '''Calculate the maximum and minimum values the wave can reach'''
        for i in range(len(self.eigvecs)):
            phi = self.consts[i] * self.eigvecs[i]
            self.vmax = max(self.vmax, np.max(phi))
            self.vmin = min(self.vmin, np.min(phi))

    def animate(self):
        '''Animate the oscillation of a wave originating at some point in the region'''
        fig = plt.figure(figsize = (5, 5))
        start = unvectorize(0 * self.eigvecs[0], self.indexed_grid)

        u = 0
        for i in range(len(self.consts)):
            u += self.consts[i] * self.eigvecs[i] * np.sin(self.alpha * np.sqrt(self.eigvals[i]) * 1 / fps)
        
        #im = plt.imshow(start, interpolation = 'none', aspect = 'auto', vmin = self.vmin, vmax = self.vmax)
        im = plt.imshow(start, interpolation = 'none', aspect = 'auto', vmin = -np.max(u), vmax = np.max(u))
        #im = plt.imshow(start, interpolation = 'none', aspect = 'auto')

        def animate_func(t):
            u = 0
            for i in range(len(self.consts)):
                #print(i, end = ' ')
                u += self.consts[i] * self.eigvecs[i] * np.sin(self.alpha * np.sqrt(self.eigvals[i]) * t / fps)
            u = unvectorize(u, self.indexed_grid)
            g = self.grid - 1
            im.set_array(u + g)
            return [im]

        return animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = 10 * fps,
                               interval = 1000 / fps, # in ms
                               )

    def play_sound(self):
        alpha = 2*np.pi * (110/2) / np.sqrt(self.eigvals[0])
        beta = 1
        time = np.linspace(0, 3, fs*3)
        sound = np.array([sum([np.exp(-beta*t) * self.consts[i] * np.sin(np.sqrt(self.eigvals[i]) * alpha * t) * self.eigvecs[i][20] for i in range(10)]) for t in time])
        print(np.max(sound))
        sd.play(3 * sound / np.max(sound), fs)

if __name__ == '__main__':
    # First isospectral region
    #x = np.array([0, 0, 2, 2, 3, 2, 1, 1, 0])
    #y = np.array([0, 1, 3, 2, 2, 1, 1, 0, 0])

    # Second isospectral region
    x = np.array([0, 0, 2, 2, 3, 2, 1, 1, 0])
    y = np.array([1, 2, 2, 3, 2, 1, 1, 0, 1])

    # Triangular region
    #x = np.array([0, 1, 2, 0])
    #y = np.array([0, 2, 0, 0])

    # Square region
    size = 2
    x = np.array([0, 0, size, size, 0])
    y = np.array([0, size, size, 0, 0])

    start = time.time()
<<<<<<< HEAD
    s = Solver(x, y, ngrid = 8)
=======
    s = Solver(x, y, ngrid = 32)
>>>>>>> cdd3b0e1ad3ca5b6dd72d81f460e84c02a9620c3
    s.get_eigs()
    print(time.time() - start)

    gauss = s.create_gaussian(1.5, 1, sigma = 0.2)
    s.calc_consts(gauss)

    s.play_sound()

    anim = s.animate()
    plt.show()

    #np.set_printoptions(threshold = np.inf)
    #print(s.eigvecs[0])
    #print(s.eigvecs[1])
    #print(s.eigvecs[2])

    #s.show_eigvec(0)
    #s.show_eigvec(1)
    #s.show_eigvec(2)