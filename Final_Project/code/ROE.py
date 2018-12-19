''' This subroutine contains the Roe's and Roe-Pike's 
    method based on Toro's book 
'''
import numpy as np
gamma = 1.4              # the gas constant

def flux_euler(U):
    rho = U[0] 
    u   = U[1] / rho
    p   = (gamma-1)*(U[2] - .5*U[1]*u)
    # return the flux vector 
    return np.array([U[1], U[1]*u + p, (U[2]+p)*u])

def Riemann_Roe(UL,UR,x,t,U0x):
    #print(t)
    rhoL = UL[0]
    rhoR = UR[0]
    uL = UL[1] / rhoL
    uR = UR[1] / rhoR
    pL = (gamma-1)*(UL[2] - .5*UL[1]*uL)
    pR = (gamma-1)*(UR[2] - .5*UR[1]*uR)


    HL = (UL[2]+pL)/rhoL                # total enthalpy left
    HR = (UR[2]+pR)/rhoR                # total enthalpy right


    ########__compute avarge variables__#######
    avgu = (np.sqrt(rhoL)*uL + np.sqrt(rhoR)*uR)/(np.sqrt(rhoL) + np.sqrt(rhoR))  # avg velocity
    avgH = (np.sqrt(rhoL)*HL + np.sqrt(rhoR)*HR)/(np.sqrt(rhoL) + np.sqrt(rhoR))  # avg enthalpy
    avgV = avgu**2
    avga = ((gamma-1)*(avgH-0.5*avgV**2))**0.5   #avg sound speed
    
    ########__compute the avg eigens___##########
    Lmd1   = avgu - avga
    Lmd2   = avgu
    Lmd3   = avgu + avga

    ######___compute the avg eigenvectors___########
    K1 = np.zeros((3,len(pL)))
    K2 = np.zeros((3,len(pL)))
    K3 = np.zeros((3,len(pL)))
    for i in range(len(pL)):
        K1[:,i] = [1, avgu[i]-avga[i], avgH[i] - avgu[i]*avga[i]]
        K2[:,i] = [1, avgu[i], 0.5*avgV[i]**2]
        K3[:,i] = [1, avgu[i]+avga[i], avgH[i] + avgu[i]*avga[i]]


    #######___compute the wave strength__########
    Du1 = rhoR  - rhoL
    Du2 = UR[1] - UL[1]
    Du3 = UR[2] - UL[2]
    alp2 = (gamma-1)/(avga)**2*(Du1*(avgH-avgu**2)+avgu*Du2-Du3)
    alp1 = 1/2/avga*(Du1*(avgu + avga)- Du2 - avga* alp2)
    alp3 = Du1 - (alp1 + alp2)

    #######___compute the fluxes___#########
    fL = flux_euler(UL)
    fR = flux_euler(UR)

    F = np.zeros_like(fL)
    for i in range(len(pL)):
        F[:,i] = 0.5*(fL[:,i] + fR[:,i]) - \
            0.5*(alp1[i]*abs(Lmd1[i])*K1[:,i] \
             + alp2[i]*abs(Lmd2[i])*K2[:,i] + alp3[i]*abs(Lmd3[i])*K3[:,i])

    return F



def Riemann_Roe_Pike(UL,UR,x,t,U0x):
    #print(t)
    rhoL = UL[0]
    rhoR = UR[0]
    uL = UL[1] / rhoL
    uR = UR[1] / rhoR
    pL = (gamma-1)*(UL[2] - .5*UL[1]*uL)
    pR = (gamma-1)*(UR[2] - .5*UR[1]*uR)


    HL = (UL[2]+pL)/rhoL                # total enthalpy left
    HR = (UR[2]+pR)/rhoR                # total enthalpy right


    ########__compute avarge variables__#######
    avgrho  = np.sqrt(rhoL*rhoR)
    avgu = (np.sqrt(rhoL)*uL + np.sqrt(rhoR)*uR)/(np.sqrt(rhoL) + np.sqrt(rhoR))  # avg velocity
    avgH = (np.sqrt(rhoL)*HL + np.sqrt(rhoR)*HR)/(np.sqrt(rhoL) + np.sqrt(rhoR))  # avg enthalpy
    avgV = avgu**2
    avga = ((gamma-1)*(avgH-0.5*avgV**2))**0.5   #avg sound speed
    
    ########__compute the avg eigens___##########
    Lmd1   = avgu - avga
    Lmd2   = avgu
    Lmd3   = avgu + avga

    ######___compute the avg eigenvectors___########
    K1 = np.zeros((3,len(pL)))
    K2 = np.zeros((3,len(pL)))
    K3 = np.zeros((3,len(pL)))
    for i in range(len(pL)):
        K1[:,i] = [1, avgu[i]-avga[i], avgH[i] - avgu[i]*avga[i]]
        K2[:,i] = [1, avgu[i], 0.5*avgV[i]**2]
        K3[:,i] = [1, avgu[i]+avga[i], avgH[i] + avgu[i]*avga[i]]


    #######___compute the wave strength__########
    Dp   = pR - pL
    Drho = rhoR - rhoL 
    Du   = uR  - uL
    
    alp1 = 1/2/avga**2*(Dp - avgrho*avga*Du)
    alp2 = Drho - Dp/avga**2
    alp3 = 1/2/avga**2*(Dp + avgrho*avga*Du)

    #######___compute the fluxes___#########
    fL = flux_euler(UL)
    fR = flux_euler(UR)

    F = np.zeros_like(fL)
    for i in range(len(pL)):
        F[:,i] = 0.5*(fL[:,i] + fR[:,i]) - \
            0.5*(alp1[i]*abs(Lmd1[i])*K1[:,i] \
             + alp2[i]*abs(Lmd2[i])*K2[:,i] + alp3[i]*abs(Lmd3[i])*K3[:,i])

    return F
