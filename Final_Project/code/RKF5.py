''' This is an explicit 5th order Runge-Kutta solver for the time 
    iteration based on the code in the Jupyter notebook:FDTransient 
    of the class, and an adaptive error control is embedded
'''
import numpy as np
import pandas
from fractions import Fraction
from matplotlib import pyplot as plt
from numpy import linalg as LA


def adp_RK(RHS, u0, tfinal, h, p, e):  #u0 is the initial value , p is the order of the method, e is the tolerence
    def RKF():
        dframe = pandas.read_html('https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method')[0]
        # Clean up unicode minus sign, NaN, and convert to float
        dfloat = dframe.applymap(lambda s: s.replace('−', '-') if isinstance(s, str) else s) \
            .fillna(0).applymap(Fraction).astype(float)

        # Extract the Butcher table
        darray = np.array(dfloat)
        A_RKF = darray[:6,2:]
        b_RKF = darray[6:,2:]
        return A_RKF , b_RKF[0,:] , b_RKF[1,:]

    A, b1, b2 = RKF()
    c = np.sum(A, axis=1)      # vector of abscissa
    s = len(c)                 # number of stages
    u   = u0.copy()
    u_e = u0.copy()            # initialize the error control vector
    t = 0
    hist = [(t,u0)]            # the initial status pair
    
    ########__decide the final step size__########
    while t < tfinal:
        if tfinal - t < 1.01*h:
            h = tfinal - t
            tnext = tfinal
        else:
            tnext = t + h
        h = min(h, tfinal - t)
    ##############################################

        fY = np.zeros((len(u0), s))                  #the approximation matrix at t = t + h, col for each stage
        for i in range(s):                           # i = 0,1,2...s-1
            Yi = u.copy() 
            for j in range(i):
                Yi += h * A[i,j] * fY[:,j]
            fY[:,i] = RHS(t + h*c[i], Yi).ravel()
                                              
        e_loc =  LA.norm (h * fY @ (b1-b2), np.inf)
        c_loc  =  e_loc/(h**p)
        h_star =  np.power(e/c_loc,1/p)
        #print(e_loc-e,h,t)
        if (e_loc>=e) and (h_star>=0.0005):
            h = h_star* 0.5                                    # Safe factor  =  0.1
                   
        else:
            u     += h * fY @ b1
            t = tnext
            hist.append((t, u.copy())) 
                   
    return hist