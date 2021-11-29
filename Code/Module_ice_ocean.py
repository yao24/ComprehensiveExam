'''
                                          ----------------------------------------

                                               COMPREHENSIVE EXAM ARTIFACT


                                               Author: Yao Gahounzo
                                                       Computing PhD
                                                       Computational Math, Science and Engineering
                                                       Boise State University

                                               Date: September 29, 2021

                                          ----------------------------------------

    This file contains all the subroutines of solving the 1D Diffusion equation based on continuous galerkin (CG) method.

    There are subroutines for different boundary conditions (Dirichlet, Neumann, and Robin).

    Lagrange polynomial have been used as the basis function for the spatial integration.
    Legendre_Gauss_Lobatto (LGL) formulas is used to compute the integration weight.

    Lobatto points have been used in the grid construction.


'''



# Import some modules
from numpy import *
from time import perf_counter
from scipy.sparse import csr_matrix
import copy
from scipy.optimize import fsolve
from scipy import special



def Legendre_deriv(Q, x):
    
    '''
    This function computes the Legendre polynomial and its derivative
    
    Inputs:
    -------
            Q  : Integration order(N+1: for exact, N: for inexact integration)
            x  : value of x at which the polynomial is evaluated
            
    Outputs:
    -------
           L1  : Value of the polynomial at x
           dL1 : First derivative
           ddLi: Second derivative
    '''
    # Initialization
    L0 = 1; dL0 = 0    # the first legendre polynomial
    L1 = x; dL1 = 1    # derivative of the first legendre polynomial
    ddL0 = 0; ddL1 = 0 # second derivative of first legendre polynomial
    
    for i in range(2, Q+1):
        
        Li = ((2*i-1)/i)*x*L1 - ((i-1)/i)*L0  # iteration of the polynomials
        dLi = i*L1 + x*dL1                    # first derivative of legendre polynomial at i-th iteration
        ddLi = (i+1.0)*dL1 + x*ddL1           # second derivative of legendre polynomial at i-th iteration
        
        L0,L1 = L1,Li
        
        dL0,dL1 = dL1,dLi
       
        ddL0,ddL1 = ddL1,ddLi
    # return legendre polynomial and its first and second derivatives   
    return L1, dL1, ddL1                   

def Lobatto_deriv(Q, x):
    
    '''
    This function computes the Lobatto polynomial and its derivative
    
    Inputs:
    -------
            Q  : Integration order(N+1: for exact, N: for inexact integration)
            x  : value of x at which the polynomial is evaluated
            
    Outputs:
    -------
           B  : Value of the polynomial at x
           dB : First derivative
    '''
    # call of the function "Legendre_deriv" to compute legendre polynomial and its first and second derivatives 
    L,dL, ddL = Legendre_deriv(Q-1, x)     
    B = (1.0-x**2)*dL                      # lobatto polynomial
    dB = -2.0*x*dL + (1.0-x**2)*ddL        # derivative of lobatto polynomial   
    
    # return lobatto polynomial and its first derivative
    return B, dB 

def Lobatto_p(Q):
    
    '''
    This function computes the Lobatto points
    
    Input:
    -------
            Q  : Integration order(N+1: for exact, N: for inexact integration)
            
    Output:
    -------
           X: array containing the Lobatto points
    '''
    
    X = []                                      # Array that contains lobatto points
    K = 100                                     # Order of approximation of Newton method
    e = 1e-20                                   # tolerance
    for i in range(Q+1):
        xik = cos(((2*i+1)/(2*(Q+1)-1))*pi)         # Chebchev points

        for k in range(K):
            out1, out2 = Lobatto_deriv(Q+1, xik)
            xikk = xik - out1/out2              # approximation of the solution using Newton

            if abs(xikk-xik) < e:

                break

            xik = xikk

        X.append(xikk)
    
    # return array that contains the roots of lobatto polynomial
    return array(X[::-1])

# Lagrange basis for single value x
def LagrangeBasis(N, i, xl, Xr):
    
    '''
    This function computes the Lagrange polynomial(basis function) and its derivative
    
    Inputs:
    -------
            N  : polynomial order
            i  : ith polynomial 
            xl : values at which the polynomial is evaluated
            Xr : Lobatto points or the roots of the generating polynomial used to construct the basis function
            
    Outputs:
    -------
           L   : Value of the polynomial
           dL  : Derivative
    '''
    L = 1; dL = 0
        
    for j in range(N+1):
            
        prod = 1
        
        if (j != i):
            L = L*(xl-Xr[j])/(Xr[i]-Xr[j])
                
            for k in range(N+1):
                if (k!=i  and k!=j):
                    
                    prod = prod*(xl-Xr[k])/(Xr[i]-Xr[k])
        
            dL = dL+prod/(Xr[i]-Xr[j])
    
    # return the value of lagrange polynomial and its first derivative at xl
    return L, dL

# Lagrange basis for an array that contains value of x
def LagrangeBasis_deriv(N,Q,Xn, Xq):

    # initialize
    l_basis = zeros((N+1,Q+1))
    dl_basis = zeros((N+1,Q+1))
    
    for k in range(Q+1):
        xl = Xq[k]
        
        for i in range(N+1):
            # Call of LagrangeBasis function
            l_basis[i,k], dl_basis[i,k] = LagrangeBasis(N, i, xl, Xn)
            
    return l_basis, dl_basis

# intma function
def intma_cdg(N, Ne, method_type):
    
    '''
    This function computes the intma array for the CG or DG. The function computes the position of each grid point 
    in each element.
    
    Inputs:
    -------
            N          : polynomial order
            Ne         : number of element
            method_type: CG or DG
            
    Output:
    -------
           intma: (matrix) that contains intma values (position of each grid point)
    '''
    
    intma = zeros((N+1,Ne))
    
    # intma for CG
    if (method_type == 'cg'):
        for e in range(1,Ne+1):
        
            t = (e-1)*N
            r = N*e
            intmm = []
            for s in range(t, r+1):
                intmm.append(s)
            intma[:,e-1] = array(intmm)
    
    # intma for DG
    if (method_type == 'dg'):
        for e in range(1,Ne+1):
        
            t = int((e-1)*N)
            r = int(e*N)

            intmm = []
            for s in range(t, r+1):
                it = e-1+s
                intmm.append(it)
            intma[:,e-1] = array(intmm)
        
    return intma


#funtion that compute weight values based on Legendre-Gauss-Lobatto (LGL) quadrature formulas
def weight(Q):
    
    '''
    This function computes the quadrature weight for the integration
    
    Inputs:
    -------
            Q : Integration order(N+1: for exact, N: for inexact integration)
            
    Output:
    -------
           w : (array) that contains the weight values
    '''
    
    xi = Lobatto_p(Q)
    w = zeros(Q+1)
    for i in range(Q+1):
        # call of the function "Legendre_deriv" to compute legendre polynomial and its first and second derivatives
        out1, out2, out3 = Legendre_deriv(Q, xi[i])
        w[i] = 2/(Q*(Q+1)*(out1)**2)        # compute the weight
        
    return w 

# Construction grid points
def grid_dg(N,Ne, xe, ax, bx):
    
    '''
    This function computes all the grid points in our problem domain
    
    Inputs:
    -------
            Q     : Integration order(N+1: for exact, N: for inexact integration)
            Ne    : Number of elements in the domain
            xe    : grid points within the element
            ax, bx: boundaries
            
    Output:
    -------
           grid: (matrix) that contains all the grid points
    '''
    
    grid = zeros((N+1,Ne))

    xel = linspace(ax,bx,Ne+1)

    for e in range(1,Ne+1):
        
        ae = xel[e-1] ; be = xel[e]

        xsi = ((be-ae)/2)*(xe-1) + be
        
        for i in range(N+1):

            grid[i,e-1] = xsi[i]
            
    return grid

# Element mass matrix

def Element_matrix(N,Q, wght,l_basis):
    
    '''
    This function computes the element mass matrix
    
    Inputs:
    -------
            Q      : Integration order(N+1: for exact, N: for inexact integration)
            N      : Polynomial order
            wght   : weights
            l_basis: basis function values(Lagrangian)
            
    Output:
    -------
           Me: Element mass matrix
    '''
    
    # initialisation of the matrix
    Me = zeros((N+1, N+1))       
    
    for k in range(Q+1):           # loop through integration points
        wk = wght[k]               # quadrature weight
        
        for i in range(N+1):       # loop through rows
            xi = l_basis[i,k]      # value of the basis function
            
            for j in range(N+1):   # loop through columns
                xj = l_basis[j,k]  # value of the basis function
                
                Me[i,j] = Me[i,j] + wk*xi*xj

    Me = (1/2)*Me

    return Me

# Direct  Stiffness Summation (DSS) operator

def DSS_operator(A, Ne, N, Np, intma, coord, matrix_type = None):
    
    '''
    This function is the Direct Stiffness Summation operator
    
    Inputs:
    -------
            A       : Matrix (element mass matrix(Me), ...)
            N       : Polynomial order
            Ne      : Number of elements
            Np      : Number of global grid points
            intma   : Array that contains position of each grid point
            coord   : all the grid points
      matrix_type   : matrix type (mass or differentiation or laplacian matrix)
            
    Output:
    -------
           M: Global matrix
    '''
    
    # Initialize the global matrix
    M = zeros((Np, Np))
    
    for e in range(1,Ne+1):
        x = coord[:,e-1]
        # grid size in each element
        dx=x[-1]-x[0]
            
        for j in range(N+1):                          # loop over the columns

            J = int(intma[j,e-1])                     # position of the i-th row in the global position

            for i in range(N+1):                      # loop over the rows

                I = int(intma[i, e-1])                # position of the i-th row in the global position

                
                if (matrix_type == 'diff'):            # diff = differentiation matrix
                    M[I,J] = M[I,J] + A[i,j]
                
                elif(matrix_type == "Lmatrix"):        # Lmatrix = Laplacian matrix
                    M[I,J] = M[I,J] + (1/dx)*A[i,j]

                else:                                  # For mass matrix
                    
                    M[I,J] = M[I,J] + dx*A[i,j]
                    
    return M


def LmatrixE(Ne,N,Q,wq,dl_basis):
    
    '''
    This function computes the element laplacian matrix
    
    Inputs:
    -------
            Q      : Integration order(N+1: for exact, N: for inexact integration)
            N      : Polynomial order
            wght   : weights
            l_basis: basis function values
            
    Output:
    -------
           Le: Element laplacian matrix
    '''
    
    # initialisation of the matrix
    Le = zeros((N+1, N+1))       
    
    for k in range(Q+1):                  # loop through integration points
        wk = wq[k]                        # quadrature weight
        
        for j in range(N+1):              # loop through columns
            xj = dl_basis[j,k]            # value of the derivative of the basis function
            
            for i in range(N+1):          # loop through rows
                xi = dl_basis[i,k]        # value of the derivative of the basis function
                
                Le[i,j] = Le[i,j] - 2*wk*xi*xj

    return Le

# Exact solution
def exact_solution(coord,Ne,N,intma, Np, time, ax, bx, icase, exactSol, g):
    
    '''
    This function compute the exact solution
    
    Inputs:
    -------
            Q        : Integration order(N+1: for exact, N: for inexact integration)
            N        : Polynomial order
            Ne       : Number of elements
            coord    : All the grid points
            ax, bx   : Left and right boundaries of the physical domain
            icase    : For the initial condition type(icase = 1, for gaussian or 2, for sine)
            time     : The time the exact solution is computed
            Np       : Number of global grid points
            g        : Derivative for neumann condition
            intma    : Array that contains position of each grid point
            
            
    Output:
    -------
           qe   : exact values
           qe_x : derivative values
    '''
    
    #Initialize
    qe = zeros(Np)
    qe_x = zeros(Np)
    
    #Generate Grid Points
    for e in range(1, Ne + 1):
        for i in range(N+1):
            x = coord[i,e-1]
            ip = int(intma[i,e-1])                 # index of the element point in the global grid points
            
            qe[ip] = exactSol(x, time, icase)      # value of the exact solution
            qe_x[ip] = g(x, time, icase)           # value of the derivative function
                
    return qe, qe_x

# Exact solution
def exact_solution1(coord,Ne,N,intma, Np, time, ax, bx, c, initial_condition):
    
    '''
    This function compute the exact solution
    
    Inputs:
    -------
            Q       : Integration order(N+1: for exact, N: for inexact integration)
            N       : Polynomial order
            Ne      : Number of elements
            coord   : all the grid points
            c       : velocity
            ax, bx  : Left and right boundaries of the physical domain
            case    : For the initial condition type(icase = 1, for gaussian or 2, for sine)
            time    
            
    Output:
    -------
           qe : exact values
    '''
    
    #Initialize
    qe = zeros(Np)

    timec = time - floor(time)
    #Generate Grid Points
    for e in range(1, Ne + 1):
        for i in range(N+1):
            x = coord[i,e-1]
            ip = int(intma[i,e-1])
            
            xbar = c*timec
            if (xbar > bx): 
                xbar = ax + (xbar - bx)
            
            qe[ip] = initial_condition(x - xbar, c)
                
    return qe

def bc_Neumann(Mmatrix_inv, Np, c, ax, bx, time, icase, g,exactSol):
    
    '''
    This  function apply Neumann boundary condition at the boundaries.
    '''
    
    B = zeros(Np)
    B[0] =  -c*g(ax, time, icase)
    B[-1] =  c*g(bx, time, icase)
    
    RHS = Mmatrix_inv@B
    
    return RHS

def bc_Dirichlet(exactSol, Np, RHS, ax, bx, time, icase):
    
    '''
    This  function apply dirichlet boundary condition at the boundaries. We applied the strong form.
    '''
    
    RHS[0] =  exactSol(ax, time, icase)
    RHS[-1] =  exactSol(bx, time, icase)
    
    return RHS

def bc_robin(M, Np,RHS, c, ax, bx, time, icase, exactSol,g,dt):
    
    '''
    This function apply the Robin boundary condition at the boundaries
    B1: contains the boundary values of the soltion at the right side (end) of the domain.
    RHS[0]: takes the value of the function at the begining of the domain, a strong dirichlet condition is applied.
    g: is the boundary function or the somehow the function that represent the derivative of the solution at the boundary.
    '''
    
    b = c*g(bx, time, icase)
    B1 = b*M[:,-1]
    RHS = RHS + dt*B1
    RHS[0] =  exactSol(ax, time, icase)
    
    return RHS

# Salinity boundary conditions
def bc_salt(M,hB, c2, Np, RHS,SB, V, bcst2,dt):
    
    b = -c2*hB(bcst2,SB, V)

    B1 = b*M[:,0]
    
    RHS[-1] = RHS[-2]
    RHS = RHS + dt*B1
    
    return RHS


# Temperature boundary conditions
def bc_temp(M,hT, c1, Np, RHS, V, bcst1,dt):
    
    a = -c1*hT(bcst1, V)
    
    B1 = a*M[:,0]
   
    RHS[-1] = RHS[-2]
    RHS = RHS + dt*B1
    
    return RHS

# Time integration methods
def IRK_coefficients(ti_method):
    
    '''
    This function compute the coefficients of the different implicit Runge Kutta methods,
    ti_method denotes the order of the methods.
    '''
  
    if(ti_method == 1):
        stages = 2
        alpha = zeros((stages,stages))
        beta = zeros(stages)
        alpha[1,0] = 0
        alpha[1,1] = 1
        beta[:] = alpha[stages-1,:]
    elif(ti_method == 2):
        stages = 3
        alpha = zeros((stages,stages))
        beta = zeros(stages)
        alpha[1,0] = 1 - 1/sqrt(2)
        alpha[1,1] = alpha[1,0]
        alpha[2,0] = 1/(2*sqrt(2))
        alpha[2,1] = alpha[2,0]
        alpha[2,2] = alpha[1,1]
        beta[:]=alpha[stages-1,:]
    elif(ti_method == 3):
        stages = 4
        alpha = zeros((stages,stages))
        beta = zeros(stages)
        alpha[1,0] = 1767732205903.0/4055673282236.0
        alpha[1,1] = 1767732205903.0/4055673282236.0
        alpha[2,0] = 2746238789719.0/10658868560708.0
        alpha[2,1] = -640167445237.0/6845629431997.0
        alpha[2,2] = alpha[1,1]
        alpha[3,0] = 1471266399579.0/7840856788654.0
        alpha[3,1] = -4482444167858.0/7529755066697.0
        alpha[3,2] = 11266239266428.0/11593286722821.0
        alpha[3,3] = alpha[1,1]
        beta[:] = alpha[stages-1,:]
    elif(ti_method == 4):
        stages = 6
        alpha = zeros((stages,stages))
        beta = zeros(stages)
        alpha[1,0] = 1.0/4.0
        alpha[1,1] = 1.0/4.0
        alpha[2,0] = 8611.0/62500.0
        alpha[2,1] = -1743.0/31250.0
        alpha[2,2] = alpha[1,1]
        alpha[3,0] = 5012029.0/34652500.0
        alpha[3,1] = -654441.0/2922500.0
        alpha[3,2] = 174375.0/388108.0
        alpha[3,3] = alpha[1,1]
        alpha[4,0] = 15267082809.0/155376265600.0
        alpha[4,1] = -71443401.0/120774400.0
        alpha[4,2] = 730878875.0/902184768.0
        alpha[4,3] = 2285395.0/8070912.0
        alpha[4,4] = alpha[1,1]
        alpha[5,0] = 82889.0/524892.0
        alpha[5,1] = 0.0
        alpha[5,2] = 15625.0/83664.0
        alpha[5,3] = 69875.0/102672.0
        alpha[5,4] = -2260.0/8211.0
        alpha[5,5] = alpha[1,1]
        beta[:] = alpha[stages-1,:]
    elif(ti_method == 5):
        stages = 8
        alpha = zeros((stages,stages))
        beta = zeros(stages)
        alpha[1,0] = 41.0/200.0
        alpha[1,1] = 41.0/200.0
        alpha[2,0] = 41.0/400.0
        alpha[2,1] = -567603406766.0/11931857230679.0
        alpha[2,2] = alpha[1,1]
        alpha[3,0] = 683785636431.0/9252920307686.0
        alpha[3,1] = 0.0
        alpha[3,2] = -110385047103.0/1367015193373.0
        alpha[3,3] = alpha[1,1]
        alpha[4,0] =  3016520224154.0/10081342136671.0
        alpha[4,1] = 0.0
        alpha[4,2] = 30586259806659.0/12414158314087.0
        alpha[4,3] = -22760509404356.0/11113319521817.0
        alpha[4,4] = alpha[1,1]
        alpha[5,0] = 218866479029.0/1489978393911.0
        alpha[5,1] = 0.0
        alpha[5,2] = 638256894668.0/5436446318841.0
        alpha[5,3] = -1179710474555.0/5321154724896.0
        alpha[5,4] = -60928119172.0/8023461067671.0
        alpha[5,5] = alpha[1,1]
        alpha[6,0] = 1020004230633.0/5715676835656.0
        alpha[6,1] = 0.0
        alpha[6,2] = 25762820946817.0/25263940353407.0
        alpha[6,3] = -2161375909145.0/9755907335909.0
        alpha[6,4] = -211217309593.0/5846859502534.0
        alpha[6,5] = -4269925059573.0/7827059040749.0
        alpha[6,6] = alpha[2,2]
        alpha[7,0] = -872700587467.0/9133579230613.0
        alpha[7,1] = 0.0
        alpha[7,2] = 0.0
        alpha[7,3] = 22348218063261.0/9555858737531.0
        alpha[7,4] = -1143369518992.0/8141816002931.0
        alpha[7,5] = - 39379526789629.0/19018526304540.0
        alpha[7,6] = 32727382324388.0/42900044865799.0
        alpha[7,7] = alpha[1,1]
        beta[:] = alpha[stages-1,:]
    elif(ti_method == 6):
        stages = 10
        alpha = zeros((stages,stages))
        beta = zeros(stages)
        alpha[1,1] = 0.2928932188134525
        alpha[2,1] = 0.3812815664617709
        alpha[3,1] = 0.4696699141100893
        alpha[4,1] = 0.5580582617584078
        alpha[5,1] = 0.6464466094067263
        alpha[6,1] = 0.7348349570550446
        alpha[7,1] = 0.8232233047033631
        alpha[8,1] = 0.9116116523516815
        alpha[9,1] = 0.7071067811865476
        alpha[9,9] = 0.2928932188134525
        beta[:] = alpha[stages-1,:]

    return alpha, beta, stages

                
def diff_Solver(N,Q,nel, Np, ax, bx, integration_type, g, exactSol,c, u, CFL, Tfinal,\
                method_type, icase, alpha1, beta1, ti_method, time_method):

    '''
    This function is CG/DG solver for 1D advection-diffusion equation
    
    Inputs:
    -------
            N              : Polynomial order
            Q              : Integration order(N+1: for exact, N: for inexact integration)
            nel            : Number of element
            nel0           : The first number of element 
            Np             : Global grid points(nel*N+1 for CG, nel*(N+1) for DG)
            ax, bx         : Left and right boundaries of the physical domain
            intergatio_type: Exact or Inexact integration
            method_type    : CG or DG
            icase          : For the initial condition type(icase = 1, for gaussian or 2, for sine)
            u              : velocity
            Courant_max    : CFL
            Tfinal         : Ending time for the computation
            time_step      : function that compute the time step and number of time( double time per element)
    Outputs:
    --------
    
            qe         : Exact solution
            q          : Numerical solution
            coord      : All grid points
            intma      : Intma(CG/DG)
    '''

    
    # Compute Interpolation and Integration Points
    
    t_start = perf_counter()  # start timing
    
    xgl = Lobatto_p(N)   # Compute Lobatto points
    
    xnq = Lobatto_p(Q)   # Compute Lobatto points
    
    wnq = weight(Q)      # Compute the weight values
    
    
    # Create intma
    intma = intma_cdg(N, nel, method_type)
    
    # Create Grid and space stuff
    
    coord = grid_dg(N, nel,xgl, ax, bx)
    
    #dx = coord[1,0] - coord[0,0]
    
    # time stuff  
    dx = (bx-ax)/Np
    dt_est = CFL*dx**2/u
    ntime = int(floor(Tfinal/dt_est)+1)

    dt = Tfinal/ntime
    
    print("N = {:d}, nel = {:d}, Np = {}".format(N,nel,Np))
    print("\tdt = {:.4e}".format(dt))
    print("\tNumber of time steps = {}".format(ntime))
    
    # Lagrange basis and its derivatives
    l_basis, dl_basis = LagrangeBasis_deriv(N,Q,xgl, xnq)

    # Form Element Mass Matrix
    Me = Element_matrix(N,Q,wnq,l_basis)
    
    # Form Global Mass and Differentiation Matrices
    GMmatrix = DSS_operator(Me, nel, N, Np, intma, coord, None)

    # Form element and global Laplacian matrix
    Le = LmatrixE(nel,N, Q,wnq,dl_basis)

    Lmatrix = DSS_operator(Le, nel, N, Np, intma, coord,"Lmatrix") 

    # Solve system at the interior points

    if(alpha1 == 0 and beta1 == 1):     # Dirichlet
        
        bcType = "Dirichlet"        
        
        Lmatrix[0,:] = 0
        Lmatrix[0,0] = 1
        Lmatrix[-1,:] = 0
        Lmatrix[-1,-1] = 1
        
        GMmatrix[0,0] = 1
        GMmatrix[-1,-1] = 1
        Dhmatrix = c*Lmatrix

    elif(alpha1 == 1 and beta1 != 0):   # Robin
        
        bcType = 'robin'
     
        Lmatrix[0,:] = 0
        Lmatrix[0,0] = 1
        
        GMmatrix[0,0] = 1

        Dhmatrix = c*Lmatrix
        Dhmatrix[-1,-1] = Dhmatrix[-1,-1] - c*beta1
        
    elif(alpha1 == 1 and beta1 == 0):  # Neumann
        
        bcType = "Neumann"
        Dhmatrix = c*Lmatrix
    
    # Inverse of global mass matrix
    GMmatrix_inv = linalg.inv(GMmatrix)
    
    # Compute Initial Solutions
    t_time = 0
    qe, qe_x = exact_solution(coord,nel,N,intma, Np, t_time, ax, bx, icase, exactSol, g)
    
    # Integretation method coefficients
    alpha,beta, stages = IRK_coefficients(ti_method)
    
    # Implicit Range Kutta method stuff
    Amatrix = GMmatrix - dt*alpha[stages-1,stages-1]*Dhmatrix
    Amatrix_inv = linalg.inv(Amatrix)
    
    # Time integration
    
    if(time_method == "IRK"):

        # Implicit time Integration
        q = qe

        Qt = zeros((Np,stages))
        R = zeros((Np,stages))

        #ntime = 1
        tn = 0
        for itime in range(1,ntime+1):

            t_time = t_time + dt

            Qt[:,0] = q[:]
            R[:,0] = Dhmatrix@Qt[:,0]
            for i in range(1,stages):         # get stage solution 
                aa = 0
                R_sum = zeros(Np)

                for j in range(i):
                    R_sum = R_sum + alpha[i,j]*R[:,j]
                    aa += alpha[i,j]

                RR = GMmatrix@Q[:,0] + dt*R_sum

                Qt[:,i] = Amatrix_inv@RR
                
                # Apply boundary conditions at the intermediate time step
                if(bcType == "Dirichlet"):
                    Qt[:,i] = bc_Dirichlet(exactSol, Np, Qt[:,i], ax, bx, tn + aa*dt, icase)
                elif(bcType == 'robin'):
                    Qt[:,i] = bc_robin(GMmatrix_inv, Np,Qt[:,i], c,ax, bx, tn + aa*dt, icase, exactSol,g,dt) 

                elif(bcType == "Neumann"):

                    qbc = bc_Neumann(GMmatrix_inv, Np,c,ax, bx, tn + aa*dt, icase, g,exactSol)

                    Qt[:,i] = Qt[:,i] + dt*qbc

                R[:,i] = Dhmatrix@Qt[:,i]

            R_sum = zeros(Np)
            
            for i in range(stages):            # solution update
                R_sum = R_sum + beta[i]*R[:,i]

            qp = q + dt*GMmatrix_inv@R_sum

            tn = tn + dt
            # Apply boundary conditions
            if(bcType == "Dirichlet"):

                qp = bc_Dirichlet(exactSol, Np, qp, ax, bx, t_time, icase)                                        

            elif(bcType == 'robin'):

                qp = bc_robin(GMmatrix_inv, Np,qp, c, ax, bx, t_time, icase, exactSol,g,dt)

            elif(bcType == "Neumann"):
                qbc = bc_Neumann(GMmatrix_inv, Np,c,ax, bx, t_time, icase, g,exactSol)

                qp = qp + dt*qbc

            # update the solution q
            q = qp   
            
    elif(time_method == "BDF2"):
        
        # Initialize for temperature
        q0 = qe
        q = qe
        
        Qt = zeros((Np,stages))
        R = zeros((Np,stages))
        
        # Computation of the second step solution using IRK method
        # Begining of IRK time integration
        t_time = t_time + dt
        tn = 0
        Qt[:,0] = q[:]
        R[:,0] = Dhmatrix@Qt[:,0]

        for i in range(1,stages):
            aa = 0
            R_sum = zeros(Np)

            for j in range(i):
                R_sum = R_sum + alpha[i,j]*R[:,j]
                aa += alpha[i,j]

            RR = GMmatrix@Q[:,0] + dt*R_sum

            Qt[:,i] = Amatrix_inv@RR
            
            # Apply boundary conditions at the intermediate time step
            if(bcType == "Dirichlet"):
                Qt[:,i] = bc_Dirichlet(exactSol, Np, Qt[:,i], ax, bx, tn + aa*dt, icase)
            elif(bcType == 'robin'):
                Qt[:,i] = bc_robin(GMmatrix_inv, Np,Qt[:,i], c,ax, bx, tn + aa*dt, icase, exactSol,g,dt) 

            elif(bcType == "Neumann"):

                qbc = bc_Neumann(GMmatrix_inv, Np,c, ax, bx, tn + aa*dt, icase, g,exactSol)
                Qt[:,i] = Qt[:,i] + dt*qbc

            R[:,i] = Dhmatrix@Qt[:,i]

        R_sum = zeros(Np)
        
        for i in range(stages):
            R_sum = R_sum + beta[i]*R[:,i]

        q = q + dt*GMmatrix_inv@R_sum
        
        # Apply boundary conditions 
        if(bcType == "Dirichlet"):

            q = bc_Dirichlet(exactSol, Np, q, ax, bx, t_time, icase)                                        

        elif(bcType == 'robin'):

            q = bc_robin(GMmatrix_inv, Np,q, c,ax, bx, t_time, icase, exactSol,g,dt)

        elif(bcType == "Neumann"):
            qbc = bc_Neumann(GMmatrix_inv, Np,c,ax, bx, t_time, icase, g,exactSol)

            q = q + dt*qbc
           
        # End of IRK time integration
        
        A = 3*GMmatrix - 2*dt*c*Lmatrix
        
        if(alpha1 == 1 and beta1 != 0):
            
            A[-1,-1] = A[-1,-1] + 2*c*dt*beta1
        
        A_inv = linalg.inv(A)
        
        Rmatrix = A_inv@GMmatrix
        
        #ntime = 0
        for itime in range(2,ntime+1):
            
            t_time = t_time + dt
            
            qp = Rmatrix@(4*q - q0)
            
            # Apply boundary conditions
            if(bcType == "Dirichlet"):
                
                qp = bc_Dirichlet(exactSol, Np, qp, ax, bx, t_time, icase)                                        

            elif(bcType == 'robin'):
                
                qp = bc_robin(A_inv, Np,qp, c,ax, bx, t_time, icase, exactSol,g,2*dt)

            elif(bcType == "Neumann"):
                
                qbc = bc_Neumann(A_inv, Np,c,ax, bx, t_time, icase, g,exactSol)

                qp = qp + 2*dt*qbc
                
            q0 = q
            q = qp
            
    elif(time_method == "BDF3"):
        
        # Initialize for temperature
        q0 = qe
        q = qe

        Qt = zeros((Np,stages))
        R = zeros((Np,stages))
        
        # Computation of the second and third step solutions of BDF3 using IRK method
        # Begining of IRK time integration
        q21 = zeros((Np,2))
        tn = 0
        for itime in range(1,3):

            t_time = t_time + dt

            Qt[:,0] = q[:]
            R[:,0] = Dhmatrix@Qt[:,0]

            for i in range(1,stages):          # get stage solution 
                aa = 0
                R_sum = zeros(Np)

                for j in range(i):
                    R_sum = R_sum + alpha[i,j]*R[:,j]
                    aa += alpha[i,j]

                RR = GMmatrix@Qt[:,0] + dt*R_sum

                Qt[:,i] = Amatrix_inv@RR
                
                # Apply boundary conditions at the intermediate time step
                if(bcType == "Dirichlet"):
                    Qt[:,i] = bc_Dirichlet(exactSol, Np, Qt[:,i], ax, bx, tn + aa*dt, icase)
                elif(bcType == 'robin'):
                    Qt[:,i] = bc_robin(GMmatrix_inv, Np,Qt[:,i], c,ax, bx, tn + aa*dt, icase, exactSol,g,dt)

                elif(bcType == "Neumann"):

                    qbc = bc_Neumann(GMmatrix_inv, Np,c,ax, bx, tn + aa*dt, icase, g,exactSol)
                    Qt[:,i] = Qt[:,i] + beta[i]*dt*qbc

                R[:,i] = Dhmatrix@Qt[:,i]

            R_sum = zeros(Np)
            
            for i in range(stages):           # solution update
                R_sum = R_sum + beta[i]*R[:,i]

            qp = q + dt*GMmatrix_inv@R_sum

            tn = tn + dt
            
            # Apply boundary conditions
            if(bcType == "Dirichlet"):

                qp = bc_Dirichlet(exactSol, Np, qp, ax, bx, t_time, icase)                                        

            elif(bcType == 'robin'):

                qp = bc_robin(GMmatrix_inv, Np,qp, c,ax, bx, t_time, icase, exactSol,g,dt)

            elif(bcType == "Neumann"):
                qbc = bc_Neumann(GMmatrix_inv, Np,c, ax, bx, t_time, icase, g,exactSol)

                qp = qp + dt*qbc

            # update the solution q
            q = qp 

            q21[:,itime-1] = qp
            
        # End of IRK time integration
        
        A = 11*GMmatrix - 6*dt*c*Lmatrix
       
        if(alpha1 == 1 and beta1 != 0):
            
            A[-1,-1] = A[-1,-1] + 6*c*dt*beta1
        
        A_inv = linalg.inv(A)
        
        Rmatrix = A_inv@GMmatrix
        
        q1 = q21[:,0]    # second step solution
        q = q21[:,1]     # third step solution
        
        # Compute the rest of time integration
        for itime in range(3,ntime+1):       
            
            t_time = t_time + dt
            
            qp = Rmatrix@(18*q - 9*q1 + 2*q0)
            
            # Apply boundary conditions
            if(bcType == "Dirichlet"):
                
                qp = bc_Dirichlet(exactSol, Np, qp, ax, bx, t_time, icase)                                        

            elif(bcType == 'robin'):
                
                qp = bc_robin(A_inv, Np,qp, c, ax, bx, t_time, icase, exactSol,g,6*dt)

            elif(bcType == "Neumann"):
                
                qbc = bc_Neumann(A_inv, Np,c, ax, bx, t_time, icase, g,exactSol)

                qp = qp + 6*dt*qbc
            
            # Updates 
            q0 = q1
            q1 = q
            q = qp
            
    # End of time integration   
                             
    time_f = perf_counter() - t_start
    
    #t_time = Tfinal
    qexact, qx = exact_solution(coord,nel,N,intma, Np, t_time, ax, bx, icase, exactSol, g)
        
    return qexact, q, coord, intma, time_f



# Ice-ocean interaction solver



def ice_ocean_Solver(N,Q,nel, Np, ax, bx, integration_type, hT, hB, \
                        initial_Temp, initial_Salt, cst1, c1, bcst1,cst2, c2,bcst2, 
                        Tw, gammaS,gammaT, cw, Li, ci, Ti, b, c, pb, a, Sw, K, M, CFL, Tfinal,\
                        u,coefF, Meltrate, SaltB,method_type,ti_method, time_method):
    '''
    This function is CG/DG solver for 1D advection-diffusion equation
    
    Inputs:
    -------
            N              : Polynomial order
            Q              : Integration order(N+1: for exact, N: for inexact integration)
            nel            : Number of element
            nel0           : The first number of element 
            Np             : Global grid points(nel*N+1 for CG, nel*(N+1) for DG)
            ax, bx         : Left and right boundaries of the physical domain
            intergatio_type: Exact or Inexact integration
            method_type    : CG or DG
            icase          : For the initial condition type(icase = 1, for gaussian or 2, for sine)
            u              : velocity
            Courant_max    : CFL
            Tfinal         : Ending time for the computation
            time_step      : function that compute the time step and number of time( double time per element)
    Outputs:
    --------
    
            qe         : Exact solution
            q          : Numerical solution
            coord      : All grid points
            intma      : Intma(CG/DG)
    '''

    
    # Compute Interpolation and Integration Points
    
    t_start = perf_counter()  # start timing
    
    xgl = Lobatto_p(N)   # Compute Lobatto points
    
    xnq = Lobatto_p(Q)   # Compute Lobatto points
    wnq = weight(Q)      # Compute the weight values
    
    # Create intma
    intma = intma_cdg(N, nel, method_type)
    
    # Create Grid and space stuff
    
    coord = grid_dg(N, nel,xgl, ax, bx)
    
    dx = coord[1,0] - coord[0,0]
    
    
    # time stuff                 
    dt_est = CFL*dx**2/u
    dt_est = 1e-2
    ntime = int(floor(Tfinal/dt_est))
    dt = Tfinal/ntime
    
    print("N = {:d}, nel = {:d}, Np = {}".format(N,nel,Np))
    print("\tdt = {:.4e}".format(dt))
    print("\tNumber of time steps = {}".format(ntime))
    
    # Lagrange basis and its derivatives
    l_basis, dl_basis = LagrangeBasis_deriv(N,Q,xgl, xnq)
    
    # Form Element Mass matrix
    Me = Element_matrix(N,Q,wnq,l_basis)
    
    # Form Global Mass matrix
    GMmatrix = DSS_operator(Me, nel, N, Np, intma, coord, None)

    # Form element and global Laplacian matrix
    Le = LmatrixE(nel,N, Q,wnq,dl_basis)

    Lmatrix = DSS_operator(Le, nel, N, Np, intma, coord,"Lmatrix") 
    
    # Inverse of global mass matrix
    GMmatrix_inv = linalg.inv(GMmatrix)
    
    # Compute Initial Solutions
    t_time = 0
    Te = exact_solution1(coord,nel,N,intma, Np, t_time, ax, bx, cst1, initial_Temp)
    Se = exact_solution1(coord,nel,N,intma, Np, t_time, ax, bx, cst2, initial_Salt)
    
    # Integretation method coefficients
    alpha,beta, stages = IRK_coefficients(ti_method)
    
    # Preparation for the time integration
    
    # Implicit Range Kutta method stuff
    DhmatrixT = c1*Lmatrix
    DhmatrixS = c2*Lmatrix
    
    AmatrixT = GMmatrix - dt*alpha[stages-1,stages-1]*DhmatrixT
    AmatrixT_inv = linalg.inv(AmatrixT)
    
    AmatrixS = GMmatrix - dt*alpha[stages-1,stages-1]*DhmatrixS
    AmatrixS_inv = linalg.inv(AmatrixS)
    
    # Time integration
    
    if(time_method == "IRK"):

        
        # Initialize for temperature
        T1 = Te
        T = Te
        Tp = Te
        # Initialize for salinity
        S1 = Se
        S = Se
        Sp = Se

        QT = zeros((Np,stages))
        RT = zeros((Np,stages))
        
        QS = zeros((Np,stages))
        RS = zeros((Np,stages))

        #ntime = 1
        tn = 0
        for itime in range(1,ntime+1):

            t_time = t_time + dt
            
            # Temperature and salinity of the far filed
            Tw = T[int(Np/2)]
            Sw = S[int(Np/2)]

            L = coefF(Tw, gammaS,gammaT, cw, Li, ci, Ti, b, c, pb, a, Sw)
         
            # compute salinity at the boundary
            SB = SaltB(K,L,M,Sw)

            # Compute melt rate
            V = Meltrate(Sw, SB, gammaS)

            QT[:,0] = T
            RT[:,0] = DhmatrixT@QT[:,0]
            QS[:,0] = S
            RS[:,0] = DhmatrixS@QS[:,0]
            
            for i in range(1,stages):          # get stage solutions
                aa = 0
                RT_sum = zeros(Np)
                RS_sum = zeros(Np)

                for j in range(i):
                    RT_sum += alpha[i,j]*RT[:,j]
                    RS_sum += alpha[i,j]*RS[:,j]
                    aa += alpha[i,j]

                RRT = GMmatrixT@QT[:,0] + dt*RT_sum
                RRS = GMmatrixS@QS[:,0] + dt*RS_sum

                QT[:,i] = Amatrix_inv@RRT
                QS[:,i] = Amatrix_inv@RRS
                
                # Apply boundary conditions at the intermediate time step
                
                QT[:,i] = bc_temp(GMmatrix_inv,hT, c1, Np, QT[:,i], V, bcst1,dt)
                QS[:,i] = bc_salt(GMmatrix_inv,hB, c2, Np, QS[:,i],SB, V, bcst2,dt)
            
                RT[:,i] = DhmatrixT@QT[:,i]
                RS[:,i] = DhmatrixS@QS[:,i]

            RT_sum = zeros(Np)
            RS_sum = zeros(Np)
            
            for i in range(stages):         # solution update
                RT_sum += beta[i]*RT[:,i]
                RS_sum += beta[i]*RS[:,i]

            Tp = T + dt*GMmatrix_inv@RT_sum
            Sp = S + dt*GMmatrix_inv@RS_sum

            tn = tn + dt
            # Apply boundary conditions
            
            Tp = bc_temp(GMmatrix_inv,hT, c1, Np, Tp, V, bcst1,dt)
            Sp = bc_salt(GMmatrix_inv,hB, c2, Np, Sp,SB, V, bcst2,dt)

            # update the solution q
            T = Tp 
            S = Sp 
            
            
    elif(time_method == "BDF3"):
        
        # Initialize for temperature
        T0 = Te
        T = Te
        # Initialize for salinity
        S0 = Se
        S = Se

        QT = zeros((Np,stages))
        RT = zeros((Np,stages))
        
        QS = zeros((Np,stages))
        RS = zeros((Np,stages))
        
        T21 = zeros((Np,2))
        S21 = zeros((Np,2))
        #ntime = 1
        tn = 0
        
        VT = zeros(ntime+1)
        SBB = zeros(ntime+1)
        
        for itime in range(1,3):

            t_time = t_time + dt
            
            # Temperature and salinity of the far filed
            Tw = T[5]
            Sw = S[5]

            L = coefF(Tw, gammaS,gammaT, cw, Li, ci, Ti, b, c, pb, a, Sw)

            # compute salinity at the boundary
            SB = SaltB(K,L,M,Sw)
            
            # Compute melt rate
            V = Meltrate(Sw, SB, gammaS)

            QT[:,0] = T
            RT[:,0] = DhmatrixT@QT[:,0]
            QS[:,0] = S
            RS[:,0] = DhmatrixS@QS[:,0]
            
            for i in range(1,stages):

                RT_sum = zeros(Np)
                RS_sum = zeros(Np)

                for j in range(i):
                    RT_sum += alpha[i,j]*RT[:,j]
                    RS_sum += alpha[i,j]*RS[:,j]
                    

                RRT = GMmatrix@QT[:,0] + dt*RT_sum
                RRS = GMmatrix@QS[:,0] + dt*RS_sum

                QT[:,i] = AmatrixT_inv@RRT
                QS[:,i] = AmatrixS_inv@RRS
                
                # Apply boundary conditions at the intermediate time step
                
                QT[:,i] = bc_temp(GMmatrix_inv,hT, c1, Np, QT[:,i], V, bcst1,dt)
                QS[:,i] = bc_salt(GMmatrix_inv,hB, c2, Np, QS[:,i],SB, V, bcst2,dt)
                
                RT[:,i] = DhmatrixT@QT[:,i]
                RS[:,i] = DhmatrixS@QS[:,i]

            RT_sum = zeros(Np)
            RS_sum = zeros(Np)
            
            for i in range(stages):
                RT_sum += beta[i]*RT[:,i]
                RS_sum += beta[i]*RS[:,i]

            Tp = T + dt*GMmatrix_inv@RT_sum
            Sp = S + dt*GMmatrix_inv@RS_sum

            # Apply boundary conditions
            
            Tp = bc_temp(GMmatrix_inv,hT, c1, Np, Tp, V, bcst1,dt)
            Sp = bc_salt(GMmatrix_inv,hB, c2, Np, Sp,SB, V, bcst2,dt)
            
            # update the solution q
            T = Tp 
            S = Sp 
            
            T21[:,itime-1] = Tp
            S21[:,itime-1] = Sp
            
        # End of IRK time integration
        
        AT = 11*GMmatrix - 6*dt*c1*Lmatrix
        AS = 11*GMmatrix - 6*dt*c2*Lmatrix
        
        AT_inv = linalg.inv(AT)
        AS_inv = linalg.inv(AS)
        
        RmatrixT = AT_inv@GMmatrix
        RmatrixS = AS_inv@GMmatrix
        
        #ntime = 0
        T1 = T21[:,0]
        T = T21[:,1]
        
        S1 = S21[:,0]
        S = S21[:,1]
        
        for itime in range(3,ntime+1):
            
            t_time = t_time + dt
            
            # Temperature and salinity of the far filed
            Tw = T[5]
            Sw = S[5]

            L = coefF(Tw, gammaS,gammaT, cw, Li, ci, Ti, b, c, pb, a, Sw)

            # compute salinity at the boundary
            SB = SaltB(K,L,M,Sw)

            # Compute melt rate
            V = Meltrate(Sw, SB, gammaS)
            
            Tp = RmatrixT@(18*T - 9*T1 + 2*T0)
            Sp = RmatrixS@(18*S - 9*S1 + 2*S0)
            
            # Apply boundary conditions
                
            Tp = bc_temp(AT_inv,hT, c1, Np, Tp, V, bcst1,6*dt)
            Sp = bc_salt(AS_inv,hB, c2, Np, Sp,SB, V, bcst2,6*dt)

            # Updates 
            T0 = T1
            T1 = T
            T = Tp
            
            S0 = S1
            S1 = S
            S = Sp
            
    # End of time integration   
                             
    time_f = perf_counter() - t_start        # End of the timing
        
    return S, T, coord, intma, time_f
