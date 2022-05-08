# Drums
Simulate ripples through arbitrarily drawn drum faces!

**1)** Run interface.py to use the interactive pygame window. 
**2)** Draw your simple, closed, polygon drum.
  **a)** Hover your mouse over the window and press 'SPACE' to place nodes.
  **b)** Click and drag nodes to change their locations.
**3)** Press '/' to start the calculation (for more details on what happens, see below).
  **a)** You will first see the discretization of the drum. Close this window when you are done viewing it.
  **b)** Next, you will see the sparse matrix which represents an iteration of the wave equation on this discretized drum
         along with the first nine eigenfunctions of the drum. Close these windows when you are done viewing them.
**4)** Now that the solver is prepped, press '0' to run the simulation of your drawn surface!
