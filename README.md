
<script type="text/javascript" async
src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js? 
config=TeX-MML-AM_CHTML"
</script>

Overview of the program

This program solves the 1D-diffusion problem using the continuous Galerkin (CG) method. The method is implemented with three different boundary conditions, Dirichlet, Neumann, and Robin. The CG method derivation and all the necessary mathematics are in the file
[CG_method.pdf](./CG_method.pdf)


The main program uses the Module_ice_ocean and Inputs modules. The Module_ice_ocean contains all the subroutines needed in the CG method, while the Inputs module contains the inputs data, exact solution, problem domain. The Main_program contains two different test cases: unit' or 'ice-ocean.

 The unit is a solution of diffusion equation  for which you have an exact solution

 $$x_{1,2} = \frac{-b \pm \sqrt{b^2-4ac}}{2b}.$$

To run the program, one needs to download the files in the code folder, then open the Main_program in jupyter notebook. The Main_program can switch from unit test to ice-ocean by changing: test_case = 'unit' or 'ice-ocean. The unit case will show the graph of the exact and numerical solutions and the convergence rates. The ice-ocean will show the temperate and salinity profiles near the ice-ocean interface at different ambient sea-water temperatures and the graph of the melt rate.


    In the case of unit test, one need to choose the problem boundary conditions (BCs):

    Dirichlet BC is: icase = 1
    Neumann BC is: icase = 2
    Robin BC is: icase = 3