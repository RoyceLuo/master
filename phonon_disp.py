#This function calculates the phonon anisotropy of material. 
#The material density and stiffness tensor need to be added manually. (thinking of building the material database)
#Input is the crystal cut.
#Output is the fitted parameter of the slowness surface. 

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def slow_surf(lx,ly,lz):
    rho, c11, c33, c44, c12, c13, c14= (2650, 86.6e9, 106.4e9, 58.0e9, 6.7e9, 12.4e9, 17.8e9) #For quartz
    c66 = (c11-c12)/2
    #==========Quartz (triagonal) (SiC Hexangonal, c14=0)===============
    gamma = np.matrix([[c11*lx**2+c66*ly**2+c44*lz**2+2*c14*ly*lz,(c12+c66)*lx*ly+2*c14*lx*lz,(c13+c44)*lx*lz+2*c14*lx*ly],
                        [(c12+c66)*lx*ly+2*c14*lx*lz,c66*lx**2+c11*ly**2+c44*lz**2-2*c14*ly*lz,(c13+c44)*ly*lz+c14*(lx**2-ly**2)],
                        [(c13+c44)*lx*lz+2*c14*lx*ly,(c13+c44)*ly*lz+c14*(lx**2-ly**2),c44*(lx**2+ly**2)+c33*lz**2]])
    w,v = np.linalg.eig(gamma)
    vinv = np.sqrt(rho/w) #1/velocity
    temp_l = abs(np.matrix([lx,ly,lz]) @ np.asmatrix(v)) #distinguish longitudinal from others
    vl_inv = np.sum(vinv*np.asarray(temp_l >= temp_l.max())) #longitudinal mode's speed
    return vl_inv



def slow_fit(xdata,S,C,E):
    return S + C*xdata[:,0]**2 + E*xdata[:,1]**2

#def slow_fit(xdata,S,A,B,C,D,E):
    #    return S + A*xdata[:,0] + B*xdata[:,1] + C*xdata[:,0]**2 + D*xdata[:,0]*xdata[:,1] + E*xdata[:,1]**2  

def disp_cal(which_cut): #phonon wavelength, refractive index, density, stiffness tensor, prop direction
    N_theta = 10 #theta range 
    N_phi = 100 #phi range
    theta = np.linspace(0,1e-2,N_theta) #the diverging angle is lambda/pi/w0 ~ 4e-3 when w0 ~ 40um. 
    phi = np.linspace(0,2*np.pi,N_phi) 
    v_inv = np.zeros((N_phi,N_theta)) #slowness surface array
    THE, PHI = np.meshgrid(theta,phi)

    ### Different for different cut
    if which_cut == 'z':
        LX = np.sin(THE)*np.cos(PHI)
        LY = np.sin(THE)*np.sin(PHI)
        LZ = np.cos(THE)
        for m in np.arange(N_phi):
            for n in np.arange(N_theta):
                v_inv[m,n] = slow_surf(LX[m,n],LY[m,n],LZ[m,n]) 
        
        vx_inv, vy_inv, vz_inv = v_inv*(LX, LY, LZ)

    elif which_cut == 'x':
        LX = np.cos(THE) 
        LY = np.sin(THE)*np.cos(PHI)
        LZ = np.sin(THE)*np.sin(PHI)
        for m in np.arange(N_phi):
            for n in np.arange(N_theta):
                v_inv[m,n] = slow_surf(LX[m,n],LY[m,n],LZ[m,n])
        vz_inv, vx_inv, vy_inv = v_inv*(LX, LY, LZ)

    elif which_cut == 'y':
        LX = np.sin(THE)*np.sin(PHI)
        LY = np.cos(THE) 
        LZ = np.sin(THE)*np.cos(PHI)
        for m in np.arange(N_phi):
            for n in np.arange(N_theta):
                v_inv[m,n] = slow_surf(LX[m,n],LY[m,n],LZ[m,n])
        vy_inv, vz_inv, vx_inv = v_inv*(LX, LY, LZ)

    else: 
        np.error("Input Error! Propagating axis: x or y or z?")

    #return vx_inv, vy_inv, vz_inv
    xdata = np.vstack((vx_inv.ravel(),vy_inv.ravel())).T
    ydata = vz_inv.ravel()

    #p_init = (1/vl,0,0,-vl/2,0,-vl/2) ##### (S0, A, B, C, D, E)
    p_init = (vz_inv[0,0],-1/2/vz_inv[0,0],-1/2/vz_inv[0,0]) #(1/vl,-vl/2,-vl/2) ##### (S0, C, E)
    popt, pcov = curve_fit(slow_fit, xdata, ydata, p0=p_init)

    #P_aniso = -1/(2*popt[0]*popt[3]) #### If fit kz vs kx and ky, then P_aniso=-1/2CS0
    P_aniso = -1/(2*popt[0]*popt[1])
    print("The anisotropy-parameter is %.3f.\n"%P_aniso)
    print("Initial parameters are", p_init,"\nOptimized parameters are", popt)

    vz_inv_fit = slow_fit(xdata, *popt).reshape(v_inv.shape)

    fig, axs = plt.subplots(1,2,figsize=(12,5))
    axs[0].contourf(vx_inv*1e4, vy_inv*1e4, vz_inv*1e4)
    axs[0].set_xlabel(r'$k_x/\omega (10^{-4}s/m)$')
    axs[0].set_ylabel(r'$k_y/\omega (10^{-4}s/m)$')
    axs[0].set_title('Christoffel solution')

    axs[1].contourf(vx_inv*1e4, vy_inv*1e4, vz_inv_fit*1e4)
    axs[1].set_xlabel(r'$k_x/\omega (10^{-4}s/m)$')
    axs[1].set_title('Fitted surface')
    plt.show()

    return popt


