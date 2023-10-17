import numpy as np
import math
import matplotlib.pyplot as plt

import phonon_disp as pd
import z_profile as zp
import eig_solver as es




def main():
    ph_lamda = 1.55e-6/2/1.54 #phonon wavelength
    k0 = 2*math.pi/ph_lamda
    L = 0.5e-3 #Crystal thickness
    #===========load mirror profiles=================
    fn1 = r"Z:\Data\Royce\Yale Facilities\Zygo\20230609_BC005_BC004_BC001C_xcut\BC004\Lens00_FOV0p4mm_Stitch3x3.datx"
    fn2 = r"Z:\Data\Royce\Yale Facilities\Zygo\20230609_BC005_BC004_BC001C_xcut\BC005\Flat01_SideB_FOV0p4mm_Stitch3x3.datx"
    ZA,ZB,xy_reso,r = zp.profile_load(fn1,fn2)
    N_span = ZB.shape[0]
    print(N_span)
    #=========phonon dispersion=======================
    popt = pd.disp_cal('z') #get fitted parameters of slowness surface
    P_aniso = -1/(2*popt[0]*popt[1])
    fB = ph_lamda/popt[0] #Brillouin frequency
    FSR = 1/popt[0]/2/L
    #=======construct propagation coordinates===========
    #real space
    x_zp = np.fft.fftshift(np.fft.fftfreq(N_span,d = 1/(xy_reso*N_span)))
    X, Y = np.meshgrid(x_zp, x_zp)
    win_fun = (abs(X)**2 + abs(Y)**2) < r**2 #windown function

    #k-space
    k_x = 2*math.pi*np.fft.fftfreq(N_span, d=xy_reso)
    K_X, K_Y = np.meshgrid(k_x, k_x)

    #K_Z = -1j*np.sqrt(K_X**2+K_Y**2-k**2+0j) ###########isotropic medium propagation
    K_Z = pd.slow_fit(np.vstack((K_X.ravel()/(2*math.pi*fB),K_Y.ravel()/(2*math.pi*fB))).T, *popt).reshape(K_X.shape) * (2*math.pi*fB) #calculate Kz from fitted results

    prop_phi = np.exp(-1j*K_Z*L) #propagation phase, the anisotropy is embedded
    mirra_phi = np.exp(+2j*k0*ZA) #Flat side phase
    mirrb_phi = np.exp(+2j*k0*ZB) #Dome side phase

    print("Mirror phase generated. Start optimizing beam waist...")
    w0_init = 30e-6
    w0 = es.opt_M00(w0_init,k0, X,Y, prop_phi,mirra_phi,mirrb_phi)
    print("Optimized beam waist is %.2um"%(w0*1e6))
    zR = k0*w0**2/2
    wL = w0*np.sqrt(1+(L/P_aniso/zR)**2)
    RL = L/P_aniso + zR**2/L*P_aniso
    print('zR is %.2fmm, wL is %.2fum.'%(zR*1e3,wL*1e6))
    print('Estimated ROC is %.2fmm. ' % (RL*1e3))

    #==========plot the optimized beam waist and window function================
    u0 = es.HG_Fun(X,Y,k0,w0,L,0,0) #Beam amplitude at end mirror
    win_fun2 = abs(np.diff(np.diff(win_fun,axis=0),axis=0)[:,:-2]) + abs(np.diff(np.diff(win_fun,axis=1),axis=1)[:-2,:])
    u_window = (abs(u0[:-2,:-2])/abs(u0[:-2,:-2]).max())**2 + win_fun2/win_fun2.max() #a factor to present the window function 

    plt.figure(figsize=(8,6))
    plt.contourf(x_zp[:-2]*1e6,x_zp[:-2]*1e6,u_window)
    plt.xlabel('x (um)')
    plt.ylabel('y (um)')
    plt.title('Beam inside the window')
    plt.colorbar()
    plt.show()

    #===================main =====================
    #es.solve(15,w0,k0,FSR, X,Y,prop_phi,mirra_phi,mirrb_phi)






