# Drums
Simulate ripples through arbitrarily drawn drum faces!

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

<p align='center'>
  <img src='imgs/drawing-drum.gif' width='500'>
  <h5 align = 'center'>Example of piecewise linear gradient selection</h5>
</p>

<p align='center'>
  <img src='imgs/para-color-example.gif' width='500'>
  <h5 align = 'center'>Example of piecewise linear gradient selection</h5>
</p>

<p align='center' style='flex'>
  <img src='imgs/discretization.PNG' width='400'>
  <img src='imgs/matrix.PNG' width='400'>
  <img src='imgs/eigenfunctions.PNG' width='400'>
  <h5 align = 'center'>Example of piecewise linear gradient selection</h5>
</p>

<p align='center'>
  <img src='imgs/simulation.gif' width='500'>
  <h5 align = 'center'>Example of piecewise linear gradient selection</h5>
</p>
