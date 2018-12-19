'''This subroutine tests three solvers in a finer grid'''

import numpy as np
from RKF5 import adp_RK
from matplotlib import pyplot as plt
from numpy import linalg as LA
from Limiter import limit_minmod
from Approx import HLLC
from ROE import Riemann_Roe_Pike
from Exact import Riemann_exact
import time


'''The time of evaluation in the paper is unknown,
   so a similar result in toro's book has been extracted'''
sod_density  = np.genfromtxt('data/toro_sod_density.dat',delimiter=',')
sod_velocity = np.genfromtxt('data/toro_sod_velocity.dat',delimiter=',')
sod_pressure = np.genfromtxt('data/toro_sod_pressure.dat',delimiter=',')
sod_IE       = np.genfromtxt('data/toro_sod_IE.dat',delimiter=',')


# some constants
a =   -1                          # the right boundary
b =   2                           # the left  boundary
n =   600                         # number of grid points
tfinal = 0.26                       # final time
err    = 1e-5                     # tolerance for adap 
p      = 5                        # time iteration order
gamma  = 1.4


def ini_U(x): 
    def ini_d_v_p(x):        #[density,velocity,pressure,energy]
        return np.array([np.where(x >= 0.5,0.125,1), 0*x , np.where(x >= 0.5,0.1,1)])
    U = ini_d_v_p(x)         #[density,momentum,energy]
    return np.array([U[0], U[0]*U[1], (U[2]/(gamma-1) + .5*U[0]*U[1]**2)])
  

def euler_solve(solver, U0, a, b, n, tfinal, limit):
    dx     = (b - a)/n                    # the grid size
    x      = np.linspace(a+dx/2, b-dx/2, n)  # Element midpoints (centroids)
    U0x    = U0(x)                 # the initial status
    Ushape = U0x.shape
    idxL   = np.arange(-1, n-1)    # the left index
    idxR   = np.arange(1, n+1) % n # the right index 

    def rhs(t, U):
        U = U.reshape(Ushape)
        jump = U[:,idxR] - U[:,idxL]
        r = np.zeros_like(jump)
        np.divide(U - U[:,idxL], jump, out=r, where=(jump!=0))
        g = limit(r) * jump / (2*dx)
#adp_RK(RHS, u0, tfinal, h, p, e):
        fluxL = solver(U[:,idxL] + g[:,idxL] * dx/2, U - g * dx/2,x,t,U0x)
        return (fluxL - fluxL[:,idxR]).flatten() / dx
    
    hist = adp_RK(rhs, U0x.flatten(), tfinal, dx/10, p, err)
    return x, [(t, U.reshape(Ushape)) for t, U in hist]



# 1 : the  solvers with time estimate
start1   = time.time()
x, hist1   = euler_solve(HLLC,   ini_U, a, b, n, tfinal, limit_minmod)
end1     = time.time()
print(end1-start1)
start2   = time.time()
x, hist2   = euler_solve(Riemann_Roe_Pike, ini_U, a, b, n, tfinal, limit_minmod)
end2     = time.time()
print(end2-start2)
start3  = time.time()
x, hist3   = euler_solve(Riemann_exact,  ini_U, a, b, n, tfinal, limit_minmod)
end3     = time.time()
print(end3-start3)

'''each plotting procedure starts with a time finding, 
   which will find the closest time near 0.25s
'''
t_sod = 0.25
for i in range(len(hist1)-1):
    if (t_sod - hist1[i][0])>0 and (t_sod - hist1[i+1][0])<0:
        if abs(t_sod - hist1[i][0])>=abs(t_sod - hist1[i+1][0]):
            idx1 = i+1
        else:
            idx1 = i

for i in range(len(hist2)-1):
    if (t_sod - hist2[i][0])>0 and (t_sod - hist2[i+1][0])<0:
        if abs(t_sod - hist2[i][0])>=abs(t_sod - hist2[i+1][0]):
            idx2 = i+1
        else:
            idx2 = i

for i in range(len(hist3)-1):
    if (t_sod - hist3[i][0])>0 and (t_sod - hist3[i+1][0])<0:
        if abs(t_sod - hist3[i][0])>=abs(t_sod - hist3[i+1][0]):
            idx3 = i+1
        else:
            idx3 = i
U1 = hist1[idx1][1]               # the state vector of time \approx 0.25 of solver 1
U2 = hist2[idx2][1]               # the state vector of time \approx 0.25 of solver 2
U3 = hist3[idx3][1]               # the state vector of time \approx 0.25 of solver 3
# start to plot
#######################__density__######################            
plt.figure()
plt.plot(sod_density[:,0], sod_density[:,1],label='Sod') 
plt.plot(x[1::3], U1[0][1::3], '.',label='HLLC solver')
plt.plot(x[1::3], U2[0][1::3], '.',label='Roe-Pike solver')
plt.plot(x[1::3], U3[0][1::3], '.',label='Exact solver')
plt.title('Density comparison of different Riemann solvers')
plt.legend(loc='upper right')
plt.xlim(0,1)
plt.ylim(0,1.2)
plt.xlabel('x')
plt.ylabel('Density')

#######################__velocity__######################
plt.figure()
plt.plot(sod_velocity[:,0], sod_velocity[:,1],label='Sod') 
plt.plot(x[1::3], U1[1][1::3]/U1[0][1::3], '.', label='HLLC solver')
plt.plot(x[1::3], U2[1][1::3]/U2[0][1::3], '.', label='Roe-Pike solver')
plt.plot(x[1::3], U3[1][1::3]/U3[0][1::3], '.', label='Exact solver')
plt.title('Velocity comparison of different Riemann solvers')
plt.legend(loc='upper left')
plt.xlim(0,1)
plt.ylim(0,1.2)
plt.xlabel('x')
plt.ylabel('Velocity')

#######################__pressure__######################
plt.figure()
plt.plot(sod_pressure[:,0], sod_pressure[:,1],label='Sod')
plt.plot(x[1::3], (gamma-1)*(U1[2][1::3] - .5*U1[1][1::3]**2/U1[0][1::3]), '.', label='HLLC solver')
plt.plot(x[1::3], (gamma-1)*(U2[2][1::3] - .5*U2[1][1::3]**2/U2[0][1::3]), '.', label='Roe-Pike solver')
plt.plot(x[1::3], (gamma-1)*(U3[2][1::3] - .5*U3[1][1::3]**2/U3[0][1::3]), '.', label='Exact solver')
plt.title('Pressure comparison of different Riemann solvers')
plt.legend(loc='upper right')
plt.xlim(0,1)
plt.ylim(0,1.2)
plt.xlabel('x')
plt.ylabel('Pressure')

#######################__IE__######################
plt.figure()
plt.plot(sod_IE[:,0], sod_IE[:,1],label='Sod')
plt.plot(x[1::3], (U1[2][1::3] - .5*U1[1][1::3]**2/U1[0][1::3])/U1[0][1::3], '.', label='HLLC solver')
plt.plot(x[1::3], (U2[2][1::3] - .5*U2[1][1::3]**2/U2[0][1::3])/U2[0][1::3], '.', label='Roe-Pike solver')
plt.plot(x[1::3], (U3[2][1::3] - .5*U3[1][1::3]**2/U3[0][1::3])/U3[0][1::3], '.', label='Exact solver')
plt.title('Internal Energy comparison of different Riemann solvers')
plt.legend(loc='lower right')
plt.xlim(0,1)
plt.ylim(1,3)
plt.xlabel('x')
plt.ylabel('Internal Energy')
plt.show()
        

