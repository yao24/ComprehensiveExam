
'''
                                          ----------------------------------------

                                               COMPREHENSIVE EXAM ARTIFACT


                                               Author: Yao Gahounzo
                                                       Computing PhD
                                                       Computational Math, Science and Engineering
                                                       Boise State University

                                               Date: September 29, 2021

                                          ----------------------------------------


'''


# Import some modules
from numpy import *
from time import perf_counter
from scipy.sparse import csr_matrix
import copy
from scipy.optimize import fsolve
from scipy import special

# import module that contains the functions and solver
from Module_ice_ocean import*


def AllZeros(f,xmin,xmax,N):
    
    '''
    
    This subroutine help to compute the zero of the function (e.g f(x) = 0) and it is used 
    in the computation of the exact solution in the case of robin BC.
    
    '''
    
    # Inputs :
    # f : function of one variable
    # [xmin - xmax] : range where f is continuous containing zeros
    # N : control of the minimum distance (xmax-xmin)/N between two zeros

    dx=(xmax-xmin)/N
    x2=xmin
    y2=f(x2)
    z=[]
    for i in range (1,N):
        x1=x2
        y1=y2
        x2=xmin+i*dx
        y2=f(x2)
        if (y1*y2<=0):                          
            
            z.append(fsolve(f,(x2*y1-x1*y2)/(y1-y2))[0])
    return array(z)

# This function is used in the computation of the exact solution in the case of robin BC
func = lambda x: tan(3*x) +2*x


# Exact solution
def exactSol(x, t, icase):
    
    if(icase == 1): # Dirichlet
        # ax = 0, bx = 2*pi
        return exp(-t)*sin(x)
    
    elif(icase == 2): # Neumann
        
        return 2*pi*cos(x)*exp(-t)
    elif(icase == 3):  # robin condition
        M = 5
        X = zeros(M)
        for n in range(1,M+1):

            z = AllZeros(func,(2*n-1)*pi/6,n*pi/3,100)
            X[n-1] = z[0]

        cn = 200*(3*X-sin(3*X))/(3*X*(3*X-sin(3*X)*cos(3*X)))

        un = 0

        for n in range(1, M+1):

            un += cn[n-1]*exp(-X[n-1]**2*t/25)*sin(X[n-1]*x)
        return un
    
# Derivative of the exact solution
def g(x, t, icase):
    
    if(icase == 1):  # dirichlet
        return exp(-t)*cos(x)
    
    elif(icase == 2):  # Neumann
        return -2*pi*sin(x)*exp(-t)
    
    elif(icase == 3):  # robin condition
        return 0.0*exactSol(x, t, icase)


def domain(icase):
    
    '''
    The problem domain
    ax: starting point
    bx: ending point
    '''
    
    if(icase == 1):
        ax = 0; bx = 2*pi
    elif(icase == 2):
        ax = 0; bx = 2*pi
    elif(icase == 3):
        ax = 0; bx = 3
        
    print('Domain: [{}, {}]\n'.format(ax,bx))
    
    return ax, bx
    
    
    
def coeff_diffu(icase):
    
    '''
    This function output the coefficient of the diffusivity depending of the problem we are solving.
    '''
    
    if(icase == 1):
        c = 1
    elif(icase == 2):
        c = 1
    elif(icase == 3):
        c = 1/25
    
    # problem type
    print('Problem type: Diffusion\n')
    
    return c   

def boundary_conditions_coeff(icase):
    
    '''
    Depending on the icase, this function print the type of the boundary condition used.
    
    Boundary conditition type:                  
    
        Dirichlet: alpha = 0, beta = 1
        Neumann  : alpha = 1, beta = 0
        Robin    : alpha = 1, beta != 0
    
    '''
    
    if(icase == 1):
        print('Boundary conditition type: Dirichlet\n')
        alpha = 0 ; beta = 1
    elif(icase == 2):
        print('Boundary conditition type: Neumann\n')
        alpha = 1 ; beta = 0
    elif(icase == 3):
        print('Boundary conditition type: Robin\n')
        alpha = 1 ; beta = 1/2
        
    return alpha, beta

def TempB(Sb, a,b,c,pb): 
    '''
    Sb: interface salinity
    pb: interface pressure
    a,b,c: constants
    TB = aSb + b + cpb 
    '''
    return a*Sb + b + c*pb

def Meltrate(Sw, Sb, gammaS): 
    '''
    Sw: ambient salinity
    Sb: interface salinity
    gammaS: salinity exchange velocity
    Melt rate: V = gammaS*(Sw - Sb)/Sb 
    '''
    return gammaS*(Sw - Sb)/Sb

def SaltB(K,L,M,Sw): 
    '''
    Interface salinity 
    '''
    D = L**2 - 4*K*M*Sw
    Sb1 = (-L + sqrt(D))/(2*K) 
    Sb2 = (-L - sqrt(D))/(2*K)
    return Sb1, Sb2


# Coefficient in the quadratic equation for salinity at the boundary
def coefK(a, gammaS, gammaT,cw,cI):
    return a*(1 - gammaS*cI/(gammaT*cw))

def coefF(Tw, gammaS,gammaT, cw, Li, cI, TS, b, c, pb, a, Sw):
    A = -Tw - (gammaS/(cw*gammaT))*(Li - cI*TS) 
    B = (b + c*pb)*(1 - gammaS*cI/(cw*gammaT))
    C = a*(gammaS*cI/(cw*gammaT))*Sw
    return A + B + C

def coefM(gammaS, gammaT, cw, Li, cI, TS, a, b, c, pb):
    A = (gammaS/(cw*gammaT))*(Li - cI*TS) 
    B = cI*(gammaS/(cw*gammaT))*(b + c*pb)
    return A + B


# define exact, source functions
def initial_Temp(x, cst1):
    return cst1 + 9.5e-4*x

def initial_Salt(x, cst2):
    return cst2 + 4.0e-4*x
    
# Neumann boundary condition
def hT(bcst1, V):
    return rho*bcst1*V

def hB(bcst2,SB, V):
    return rho*bcst2*SB*V



# Values of the parameters in the ice-ocean simulation

a = -5.73e-2     # Salinity coefficient of freezing equation(˚C*psu^{-1})
b =  9.39e-2     # Constant coefficient of freezing equation(˚C)
c = -7.53e-8     # Pressure coefficient of freezing equation(˚C*Pa^{-1})
cI = 2009.0      # Specific heat capacity ice(J*kg^-1*K^-1)
cw = 3974.0      # Specific heat capacity water(J*kg^-1*K^-1)
Li = 3.35e+5     # Latent heat fusion(J*kg^-1)
Tw = 2.3         # Temperature of water(˚C)
TS = -25         # Temperature of ice(˚C)
Sw = 35          # Salinity of water(psu)
Sc = 2500        # Schmidt number
Pr = 14          # Prandtl number
mu = 1.95e-6     # Kinematic viscosity of sea water(m^2*s^-1)
pb = 1.0e+7      # Pressure at ice interface(Pa)
kT = mu/Pr       # Thermal diffusivity(m^2*s^-1)
kS = mu/Sc       # Salinity diffusivity(m^2*s^-1)

kT = 1.3e-7
kS = 7.4e-10
rhoI = 920       # density of ice(kg m^-3)
rhoW = 1025      # density of sea water(kg*m^-3)

gammaT = 5.0e-5        # Thermal exchange velocity(m*s^-1)

gammaS = 0.04*gammaT   # Salinity exchange velocity(m*s^-1)

rho = rhoI/rhoW        # report between ice density and seawater density

# Coeffecients in quadratic equation for salinity at the boundary

K = coefK(a, gammaS, gammaT,cw,cI)
M = coefM(gammaS, gammaT, cw, Li, cI, TS, a, b, c, pb)


def ice_simulation(N,Q,nel,Np, ax, bx, integration_type,method_type,ti_method, time_method,CFL,Tw,Tfinal,u):
    # For temperature
    cst1 = Tw               # Constant that initialize the initial temperature
    c1 = kT                 # Temperature diffusivity
    bcst1 = Li/(cw*kT)      # Constant term for the temperature gradient at the boundary
    # For salinity
    cst2 = Sw               # Constant that initialize the initial salinity
    c2 = kS                 # Salinity diffusivity
    bcst2 = 1/kS            # Constant term for the salinity gradient at the boundary

    # Call the ice-ocean solver for the diffusion problem

    '''
    outputs:
    --------
    S          : Salinity
    T          : Temperature
    coord      : All grid points
    intma      : Intma(CG/DG)
    '''

    S, T, coord, intma, tf = ice_ocean_Solver(N,Q,nel, Np, ax, bx, integration_type, hT, hB,\
                                    initial_Temp, initial_Salt, cst1, c1, bcst1,cst2, c2,bcst2, 
                                    Tw, gammaS,gammaT, cw, Li, cI, TS, b, c, pb, a, Sw, K,M, CFL,\
                                    Tfinal,u,coefF,Meltrate, SaltB,method_type,\
                                    ti_method, time_method)
    
    return S,T,coord,intma,tf