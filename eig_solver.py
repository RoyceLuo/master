import numpy as np
import math
from scipy.optimize import minimize
from scipy.special import hermite
import matplotlib.pyplot as plt
import phonon_disp as pd


def HG_Fun(X,Y,k,w0,z,m,n): #ROW is r/w, zeta is z/z0
    z0 = k*w0**2/2 #in case z0 = 0
    qz = z+1j*z0
    wz = np.sqrt(-2/k/np.imag(1/qz)) #or w0*np.sqrt(1+(z/z0)**2)
    WaveFun = 2**(-(m+n-1)/2)/wz/np.sqrt(math.pi*math.factorial(m)*math.factorial(n))\
    *hermite(m)(np.sqrt(2)*X/wz)*hermite(n)(np.sqrt(2)*Y/wz)*np.exp(-1j*k*(X**2+Y**2)/(2*qz)+1j*(m+n+1)*np.arctan(z/z0))
    #total_area = xy_reso**2 * sum(abs(WaveFun)**2)
    return WaveFun #/np.sqrt(total_area)

def opt_M00(w0_init,k,X,Y, prop_phi,mirra_phi,mirrb_phi):
    xy_reso = X[1,1]-X[0,0]
    print("Resolution %.2fum"%(xy_reso*1e6))
    def single_prop(w):
        u0 = HG_Fun(X,Y,k,w,0,0,0) #initialize beam amplitude
        u = beamp(beamp(u0, prop_phi)*mirrb_phi, prop_phi)*mirra_phi
        overl = 1 - abs(xy_reso**2 * np.sum(np.conj(u0)*u))**2
        print("Beam waist %.5fum, diffraction loss %.9f." % (w*1e6, overl))
        return overl
    
    w0_bounds = ((10e-6, 100e-6),)
    res = minimize(single_prop, w0_init, method = 'L-BFGS-B', bounds = w0_bounds)
    return res.x[0]

def beamp(u, phi):
    u_fft = np.fft.ifft2(np.fft.ifftshift(u), norm="ortho")
    u_fft_1 = u_fft*phi
    u_fft_prop = np.fft.fftshift(np.fft.fft2(u_fft_1, norm="ortho"))
    return u_fft_prop


def s_gen(mode_num,w0,k, X,Y,win_fun,prop_phi,mirra_phi,mirrb_phi):#generate S matrix
    xy_reso = X[1,1] - X[0,0]
    N_span = mirrb_phi.shape[0]
    ltemp = np.repeat(np.arange(mode_num),np.arange(mode_num)+1) #[0,1,1,2,2,2,...], mode# list 
    n_list = []
    for m in np.arange(mode_num):
        n_list += np.arange(m+1).tolist()
    n_list = np.array(n_list)
    m_list = ltemp - n_list

    HerFun = np.zeros((len(ltemp),N_span**2))+1j*np.zeros((len(ltemp),N_span**2)) # beam amplitude at z=0
    HerFun_refl = np.zeros((len(ltemp),N_span**2))+1j*np.zeros((len(ltemp),N_span**2)) # beam amplitude at z=L

    for s in np.arange(len(ltemp)):
        u0temp = HG_Fun(X,Y,k,w0,0,n_list[s],m_list[s])
        HerFun[s,:] = np.ravel(u0temp)
        HerFun_refl[s,:]= np.ravel(beamp(beamp(u0temp, prop_phi)*mirrb_phi*win_fun, prop_phi)*mirra_phi*win_fun)

    M = xy_reso**2 * HerFun_refl @ np.conj(HerFun).T
    return M, HerFun


def solve(mode_num,w0,k, X,Y,win_fun,prop_phi,mirra_phi,mirrb_phi):
    N_span = mirrb_phi.shape[0]
    print("Generating scattering matrix and mode basis...")
    S, HerFun = s_gen(mode_num,w0,k, X,Y,win_fun,prop_phi,mirra_phi,mirrb_phi)
    print("done!")
    #=================Eigenvalue problem======================
    eigvalue,v = np.linalg.eig(S)
    eig_modes = (v.T @ HerFun).reshape(S.shape[0],N_span,N_span)
    return eigvalue,v,eig_modes


def sol_conv(mode_num,w0,k,FSR, X,Y,win_fun, prop_phi,mirra_phi,mirrb_phi):
    eigF0 = [] #eigenfrequency of the fundamental mode
    eigv0 = [] #|a0|^2 fir fundamental mode
    print("simulation w0=%.2fum"%(w0*1e6))
    #=============generate the coordinates (mode indices) for M matrix=============
    mode_num_list = np.arange(5,mode_num,2) #mode number list
    print("Convergence test of m+n<%d"%(mode_num_list.max()-1))
    
    S = s_gen(mode_num_list.max(),w0,k, X,Y,win_fun,prop_phi,mirra_phi,mirrb_phi)[0]

    for mode_num in mode_num_list:
        mode_ind = mode_num*(mode_num+1)//2  #mode number included for m+n=mode_num-1
        eigvalue,v = np.linalg.eig(S[:mode_ind,:mode_ind])
        #eig_modes = (v.T @ HerFun[:mode_ind,:]).reshape(mode_ind,N_span,N_span) #einsum('ij,ikl->jkl',v,HerFun)
        eig_F = 2*math.pi/(1-abs(eigvalue)**2) #finesses
        eigf = (np.angle(eigvalue) + 0)/2/math.pi*FSR #eigenfrequency, need to adjust 
        for m in np.arange(len(eigf)): #Some might have one FSR off because of the np.arctan() operation
            if eigf[m] < 0:
                eigf[m] += FSR
 
        #============pick the fundamental mode============
        eig_index = []
        for i in np.arange(len(eig_F)): #filter out super lossy modes
            if abs(eig_F[i]) > eig_F.max()/2:
                eig_index.append(i)
        eig_F1 = eig_F[eig_index]
        #eigf1 = eigf[eig_index]

        mode_of_interest = 0 #mode of interest being the 0th mode

        v1 = abs(v[mode_of_interest,eig_index])**2 #|a0|^2
        eigF0.append(max(eig_F1*(v1>=v1.max()))) #find the 0th mode, can adjust to any modes
        eigv0.append(v1.max()) #|a0|^2 of the 0th mode
        print("#m+n=%d, finesse=%d, |a0|^2=%.3f"%(mode_num-1,eigF0[-1],eigv0[-1]))
    eigF_data = np.vstack((mode_num_list-1, np.array(eigF0), np.array(eigv0))).T
    print("Convergence test finished!")
    
    return eigF_data

def sim_ring():
    return 1
