# Importing Numpy package
import numpy as np
import matplotlib.pyplot as plt
  
# Initializing value of x-axis and y-axis
a, b = 3, 2
x, y = np.meshgrid(np.linspace(0,10,50), np.linspace(0,10,50))
dst = np.sqrt((x-a)**2 + (y-b)**2)
  
# Intializing sigma and muu
sigma = 1
muu = 0.000
  
# Calculating Gaussian array
gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
  
plt.imshow(gauss, interpolation = 'none')
plt.show()