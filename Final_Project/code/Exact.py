'''This subroutine contains the solver evaluating the 
   exact Riemann problem and the solver's sturcture is 
   based on the lecture, Sod's paper and Toro's book.
'''
import numpy as np
from numpy import linalg as LA
gamma = 1.4              # the gas constant
 

def flux_euler(U):
    rho = U[0] 
    u   = U[1] / rho
    p   = (gamma-1)*(U[2] - .5*U[1]*u)
    return np.array([U[1], U[1]*u + p, (U[2]+p)*u])



def Riemann_exact(UL,UR,x,t,ini):
    if t!=0:
        #print(t)
        rhoL = UL[0]
        rhoR = UR[0]
        uL = UL[1] / rhoL
        uR = UR[1] / rhoR
        pL = (gamma-1)*(UL[2] - .5*UL[1]*uL)
        pR = (gamma-1)*(UR[2] - .5*UR[1]*uR)

        # define data dependent constant
        AL = 2/(gamma+1)/rhoL
        BL = (gamma-1)/(gamma+1)*pL
        AR = 2/(gamma+1)/rhoR
        BR = (gamma-1)/(gamma+1)*pR


        
        aL = np.sqrt(gamma*pL/rhoL)      # Left local sound speed 
        aR = np.sqrt(gamma*pR/rhoR)      # right local sound speed

        # iterative nonlinear solver
        p = 0.5*(pL + pR)

        #p     = pL + pR
        funL  = np.zeros(len(p))
        funR  = np.zeros(len(p))
        dfunL = np.zeros(len(p))
        dfunR = np.zeros(len(p))

        for i in range(20):  #max iter =20

            for j in range(len(p)):
                if p[j] > pL[j]:
                    funL[j] = (p[j]-pL[j])*(AL[j]/(p[j]+BL[j]))**0.5  #left shock
                else:
                    funL[j] = (2*aL[j])/(gamma-1)*(-1 + (p[j]/pL[j])**\
                        ((gamma-1)/2/gamma)) #left rarefaction
            for j in range(len(p)):
                if p[j] > pR[j]:
                    funR[j] = (p[j]-pR[j])*(AR[j]/(p[j]+BR[j]))**0.5  #right shock
                else:
                    funR[j] = (2*aR[j])/(gamma-1)*(-1 + (p[j]/pR[j])**
                        ((gamma-1)/2/gamma)) #right rarefaction

            residual = funL + funR + (uR - uL)
            #print(LA.norm(residual))
            if LA.norm(residual) < 1e-10:
                u   = 0.5*(uL+uR) + 0.5*(funR-funL)                         
                break
            elif i+1 == 20:
                raise RuntimeError('Newton solver failed to converge')


            #Compute the derivative for iterative solver
            for j in range(len(p)):
                if p[j] > pL[j]:
                    dfunL[j] = (AL[j]/(BL[j]+p[j]))**0.5*(1-(p[j]-pL[j])/2/(BL[j]+p[j]))
                else:
                    dfunL[j] = 1/rhoL[j]/aL[j]*(p[j]/pL[j])**(-(gamma+1)/2/gamma)

            for j in range(len(p)):
                if p[j] > pR[j]:
                    dfunR[j] = (AR[j]/(BR[j]+p[j]))**0.5*(1-(p[j]-pR[j])/2/(BR[j]+p[j]))
                else:
                    dfunR[j] = 1/rhoR[j]/aR[j]*(p[j]/pR[j])**(-(gamma+1)/2/gamma)

            delta_p = -residual/(dfunL + dfunR)


            '''line serach to prevent negative pressure, pressure cannot be 
               negative cuz densitty cannot be negetive.'''
            while min(p + delta_p) <= 0 :
                delta_p *= 0.5
            p += delta_p


       #solving for density, shock speed, the sound speed, 
       #and the wave speed in the star regions

        rho_sL = np.zeros(len(p))      # rho_star on the left
        rho_sR = np.zeros(len(p))      # rho_star on the right
        SL = np.zeros(len(p))          # left shock speed
        SR = np.zeros(len(p))          # right shock speed
        a_sL = np.zeros(len(p))        # left sound speed_star
        a_sR = np.zeros(len(p))        # right sound speed_star
        RHL = np.zeros(len(p))         # left rare head speed
        RHR = np.zeros(len(p))         # right rare head speed
        RTL = np.zeros(len(p))         # left rare tail speed
        RTR = np.zeros(len(p))         # right rare tail speed


        for j in range(len(p)):
            if p[j] > pL[j]:
                rho_sL[j] = rhoL[j]*((p[j]/pL[j]+(gamma-1)/(gamma+1))/((gamma-1)/(gamma+1)*p[j]/pL[j] +1))
                SL[j]     = uL[j]- aL[j]*(((gamma+1)*p[j]/2/gamma/pL[j])+(gamma-1)/2/gamma)**(0.5)
                #a_sL[j] = aL[j]
            else:
                rho_sL[j] = rhoL[j]*(p[j]/pL[j])**(1/gamma)
                a_sL[j]   = aL[j]*(p[j]/pL[j])**((gamma-1)/2/gamma)
                RHL[j]    = uL[j] - aL[j]
                RTL[j]    = u[j]- a_sL[j]
        for j in range(len(p)):
            if p[j] > pR[j]:
                rho_sR[j] = rhoR[j]*((p[j]/pR[j]+(gamma-1)/(gamma+1))/((gamma-1)/(gamma+1)*p[j]/pR[j] +1))
                SR[j]     = uR[j]+ aR[j]*(((gamma+1)*p[j]/2/gamma/pR[j])+(gamma-1)/2/gamma)**(0.5)
                #a_sR[j]   = aR[j]
            else:
                rho_sR[j] = rhoR[j]*(p[j]/pR[j])**(1/gamma)
                a_sR[j]   = aR[j]*(p[j]/pR[j])**((gamma-1)/2/gamma)
                RHR[j]    = uR[j] + aR[j]
                RTR[j]    = u[j]  + a_sR[j]


        #Sampling the solutions
        U0 = np.zeros_like(UL)        # Initialize the output state vector
 
        for i in range(len(p)):
            if (uL[i] -aL[i]) < 0 < (u[i] - a_sL[i]):   # left sonic rarefaction, uo = a_sL

                U0[0,i] = rhoL[i]*(2/(gamma+1) + (gamma-1)*\
                    (uL[i])/(gamma+1)/aL[i])**(2/(gamma-1))                                      # toro(4.56)
                U0[1,i] = U0[0,i] * 2/(gamma+1)*(aL[i] + \
                    (gamma-1)/2*uL[i])
                U0[2,i] = (U0[0,i]/rhoL[i])**gamma*pL[i]/(gamma-1) + 0.5*U0[1,i]**2/U0[0,i]
           

            elif (u[i] + a_sR[i]) < 0 < (uR[i] + aR[i]):#  right sonic rarefaction, u0 = -a_sR
                U0[0,i] = rhoR[i]*(2/(gamma+1) - (gamma-1)*\
                    (uR[i])/(gamma+1)/aR[i])**(2/(gamma-1)) # toro(4.56)
                U0[1,i] = U0[0,i] * 2/(gamma+1)*(-aR[i] + \
                    (gamma-1)/2*uR[i])
                U0[2,i] = (U0[0,i]/rhoR[i])**gamma*pR[i]/(gamma-1) + 0.5*U0[1,i]**2/U0[0,i]


            if   ((pL[i] >= p[i] and 0 <= (uL[i] - aL[i])) or \
                 (pL[i] < p[i] and 0 < (rho_sL[i]*u[i] - UL[1,i]))):
            # Left rarefaction or shock is supersonic
                U0[:,i] = UL[:,i]

            elif ((pR[i] >= p[i] and uR[i] + aR[i] <= 0) or \
                 (pR[i] < p[i] and UR[1,i] - rho_sR[i]*u[i]) > 0):
            # Right rarefaction or shock is supersonic
                U0[:,i] = UR[:,i]

            # two sides of the contact discontinuity   
            # elif (uL[i] -aL[i]) >= SL[i] or (uL[i] -aL[i])>= RTL[i]:
             
            #     U0[0,i] = rho_sL[i]
            #     U0[1,i] = U0[0,i]*u[i]
            #     U0[2,i] = p[i]/(gamma-1) + 0.5*U0[1,i]**2/U0[0,i]
            
            # elif (uR[i] + aR[i])<= RTR[i] or (uR[i] + aR[i]) <= SR[i]:
            #     U0[0,i] = rho_sR[i]
            #     U0[1,i] = U0[0,i]*u[i]
            #     U0[2,i] = p[i]/(gamma-1) + 0.5*U0[1,i]**2/U0[0,i]
            else:
                U0[0,i] = 0.5*(rho_sL[i]+rho_sR[i])
                U0[1,i] = U0[0,i]*u[i]
                U0[2,i] = p[i]/(gamma-1) + 0.5*U0[1,i]**2/U0[0,i]


  
    else:
        U0 = ini
    
    return flux_euler(U0)



