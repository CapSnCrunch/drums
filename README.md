# Drums
Simulate ripples through arbitrarily drawn drum faces!

<p align='center'>
  <img src='imgs/simulation.gif' width='250'>
  <h5 align = 'center'>Example of simulated waves in a star-shaped drum.</h5>
</p>

## Steps to Use
**1)** Run interface.py to use the interactive pygame window.<br/><br/>
**2)** Draw your simple, closed, polygon drum.<br/>
&ensp;  **a)** Hover your mouse over the window and press 'SPACE' to place nodes.<br/>
&ensp;  **b)** Click and drag nodes to change their locations.<br/><br/>
**3)** Press '/' to start the calculation (for more details on what happens, see below).<br/>
&ensp;  **a)** You will first see the discretization of the drum. Close this window when you are done viewing it.<br/>
&ensp;  **b)** Next, you will see the sparse matrix which represents an iteration of the wave equation on this discretized drum<br/>
&ensp;&ensp;         along with the first nine eigenfunctions of the drum. Close these windows when you are done viewing them.<br/><br/>
**4)** Now that the solver is prepped, press '0' to run the simulation of your drawn surface!<br/>
<br/>

<p align='center'>
  <img src='imgs/drawing-drum.gif' width='250'>
  <h5 align = 'center'>Drawing and editing a drum surface.</h5>
</p>
<br/>

<p align='center' style='flex'>
  <img src='imgs/discretization.PNG' height='250'>
  <img src='imgs/matrix.PNG' height='250'>
  <img src='imgs/eigenfunctions.PNG' height='250'>
  <h5 align = 'center'>Example of discretization, sparse finite-difference matrix, and eigenfunctions of the above drum.</h5>
</p>

## Mathematical Background
The simulation involves a finite-difference method of approximating the wave equation for an arbitrary boundary. By discretizing the simple closed polygon, we can construct a discrete laplacian from which we can approximate eigenfunctions and iterate them under the wave equation. Read my article <a href='https://capsncrunch.github.io/articles/can_you_hear_the_shape_of_a_drum.html'>here</a> for all the details.
