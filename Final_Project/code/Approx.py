''' This subroutine contains the HLL solver, Rusanov solver 
    and the HLLC solver, the structure of the solvers is based 
    on class, Sod's paper and Toro's book
''' 
import numpy as np
gamma = 1.4              # the gas constant

def flux_euler(U):
    rho = U[0] 
    u   = U[1] / rho
    p   = (gamma-1)*(U[2] - .5*U[1]*u)
    # return the flux vector 
    return np.array([U[1], U[1]*u + p, (U[2]+p)*u])


def HLL(UL,UR,x,t,U0x):
    # current state variables
    rhoL = UL[0]
    rhoR = UR[0]
    uL = UL[1] / rhoL
    uR = UR[1] / rhoR
    pL = (gamma-1)*(UL[2] - .5*UL[1]*uL)
    pR = (gamma-1)*(UR[2] - .5*UR[1]*uR)


    aL = np.sqrt(gamma*pL/rhoL)      # Left local sound speed 
    aR = np.sqrt(gamma*pR/rhoR)      # right local sound speed
    sL = np.minimum(uL - aL, uR - aR)# left shock speed
    sR = np.maximum(uL + aR, uR + aR)# right shock speed
    fL = flux_euler(UL)
    fR = flux_euler(UR)
    
    return np.where(sL > 0, fL,
            np.where(sR < 0, fR,
                (sR*fL - sL*fR + sL*sR*(UR - UL)) / (sR-sL))) 

def Rsnov(UL,UR,x,t,U0x):
    # current state variables
    rhoL = UL[0]
    rhoR = UR[0]
    uL = UL[1] / rhoL
    uR = UR[1] / rhoR
    pL = (gamma-1)*(UL[2] - .5*UL[1]*uL)
    pR = (gamma-1)*(UR[2] - .5*UR[1]*uR)


    aL = np.sqrt(gamma*pL/rhoL)      # Left local sound speed 
    aR = np.sqrt(gamma*pR/rhoR)      # right local sound speed
    sL = np.minimum(uL - aL, uR - aR)# left shock speed
    sR = np.maximum(uL + aR, uR + aR)# right shock speed
    fL = flux_euler(UL)
    fR = flux_euler(UR)
    
    return np.where(sL > 0, fL,
                           np.where(sR < 0, fR,
                                  (fL+fR)/2 - 0.5*sR*(UR-UL)))


def HLLC(UL,UR,x,t,U0x):
    # current state variables
    rhoL = UL[0]
    rhoR = UR[0]
    uL  = UL[1] / rhoL
    uR  = UR[1] / rhoR
    pL  = (gamma-1)*(UL[2] - .5*UL[1]*uL)
    pR  = (gamma-1)*(UR[2] - .5*UR[1]*uR)

    aL = np.sqrt(gamma*pL/rhoL)      # Left local sound speed 
    aR = np.sqrt(gamma*pR/rhoR)      # right local sound speed
    sL = np.minimum(uL - aL, uR - aR)# left shock speed
    sR = np.maximum(uL + aR, uR + aR)# right shock speed
    fL     = flux_euler(UL)
    fR     = flux_euler(UR)
    # Star region stands for the contact discontinuity
    s_star = (pR-pL+rhoL*uL*(sL-uL)-rhoR*uR*(sR-uR))/(rhoL*(sL-uL)-rhoR*(sR-uR))

    pLR = 0.5*(pL+pR+rhoL*(sL-uL)*(s_star-uL)+
            rhoR*(sR-uR)*(s_star-uR))

    D      = np.zeros((3,len(sR)))
    fstarL = np.zeros((3,len(sR)))
    fstarR = np.zeros((3,len(sR)))
    for i in range(len(s_star)):
        D[1,i] = 1
        D[2,i] = s_star[i] 

    for i in range(len(s_star)):
        fstarL[:,i] = (s_star[i]*(sL[i]*UL[:,i]-fL[:,i]) + sL[i]*pLR[i]*D[:,i])/(sL[i]-s_star[i])
        fstarR[:,i] = (s_star[i]*(sR[i]*UR[:,i]-fR[:,i]) + sR[i]*pLR[i]*D[:,i])/(sR[i]-s_star[i])
    
    # output flux
    F_HLLC = np.zeros((3,len(sR)))
    for j in range(len(sL)):
        if  sL[j]>=0:
            F_HLLC[:,j]  = fL[:,j]
        elif sL[j]<=0 and 0<=s_star[j]:
            F_HLLC[:,j]  = fstarL[:,j]
        elif s_star[j]<=0 and 0<=sR[j]:
            F_HLLC[:,j]  = fstarR[:,j]
        else:
            F_HLLC[:,j]  = fR[:,j]

    return F_HLLC




                                  

