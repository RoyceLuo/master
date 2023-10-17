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
    print("resolution %.2fum"%(xy_reso*1e6))
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



def solve(mode_num,w0,k,FSR, X,Y,win_fun,prop_phi,mirra_phi,mirrb_phi):
    xy_reso = X[1,1] - X[0,0]
    N_span = mirrb_phi.shape[0]

    ltemp = np.repeat(np.arange(mode_num),np.arange(mode_num)+1) #[0,1,1,2,2,2,...], mode# list 
    n_list = []
    for m in np.arange(mode_num):
        n_list += np.arange(m+1).tolist()
    n_list = np.array(n_list)
    m_list = ltemp - n_list

    '''
    m_list = []
    n_list = []
    for n in range(mode_num):
        n_temp = list(np.arange(n+1))
        n_list = n_list+n_temp
        m_temp = list(n-np.arange(n+1))
        m_list = m_list+m_temp

    n_list = tuple(n_list)
    m_list = tuple(m_list)
    ModeNum = len(n_list)
    '''
    #============generate those Hermite-Gaussian modes=============
    HerFun = np.zeros((len(ltemp),N_span**2))+1j*np.zeros((len(ltemp),N_span**2)) # beam amplitude at z=0
    HerFun_refl = np.zeros((len(ltemp),N_span**2))+1j*np.zeros((len(ltemp),N_span**2)) # beam amplitude at z=L

    for s in range(ltemp):
        if s%100==0:
            print(s)
        u0temp = HG_Fun(X,Y,k,w0,0,n_list[s],m_list[s])
        HerFun[s,:] = np.ravel(u0temp)
        HerFun_refl[s,:]= np.ravel(beamp(beamp(u0temp, prop_phi)*mirrb_phi*win_fun, prop_phi)*mirra_phi*win_fun)

    #=================generate the M matrix========================
    M = xy_reso**2 * HerFun_refl @ np.conj(HerFun).T
    #=================Eigenvalue problem======================
    eigvalue,v = np.linalg.eig(M)
    eig_modes = (v.T @ HerFun).reshape(len(ltemp),N_span,N_span) #einsum('ij,ikl->jkl',v,HerFun)

    eig_F = 2*math.pi/(1-abs(eigvalue)**2)
    eigf = np.angle(eigvalue + 0)/2/math.pi*FSR
    for m in np.arange(len(eigf)): ### np.angle() reaturns (-math.pi,math.pi]
        if eigf[m] < 0:
            eigf[m] += FSR

    #==============reorder it in terms of their frequencies=================
    sort_ind = np.argsort(eigf)
    eigf = eigf[sort_ind] 
    eig_F = eig_F[sort_ind]
    eig_modes = eig_modes[sort_ind,:,:]
    v = v[:,sort_ind]

    #============pick the mode of interest============
    eig_index = [] #mode index of interest
    for i in np.arange(len(eig_F)):
        if abs(eig_F[i]) > eig_F.max()/2:
            eig_index.append(i)
    eig_F1 = eig_F[eig_index]
    eigf1 = eigf[eig_index]
    v1 = abs(v[0,eig_index])**2

    return eigf1,eig_F1,v1

def sol_conv(mode_num,w0,k,FSR, X,Y,win_fun, prop_phi,mirra_phi,mirrb_phi):
    xy_reso = X[1,1] - X[0,0]
    N_span = mirrb_phi.shape[0]
    eigF0 = [] #eigenfrequency of the fundamental mode
    eigv0 = [] #|a0|^2 fir fundamental mode
    print("simulation w0=%dum"%(w0*1e6))
    #=============generate the coordinates (mode indices) for M matrix=============
    mode_num_list = np.arange(5,mode_num,2) #mode number list
    N = mode_num_list.max() #m+n+1, maximum
    ltemp = np.repeat(np.arange(N),np.arange(N)+1) #[0,1,1,2,2,2,...], mode# list 
    n_list = []
    for m in np.arange(N):
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
    print("M size is %d by %d."%(M.shape[0],M.shape[1]))
    #savetxt(r"Z:\Data\Royce\Yale Facilities\Zygo\20230609_BC005_BC004_BC001C_xcut\BC004\Convergence_test\BC004_Lens01_ScatteringMatrix_r%dum_SimW0%dum.csv"%(r*1e6,w0*1e6), M, delimiter=',', header="Scattering Matrix\n%d rows %d cols\nm+n=%d"%(M.shape[0],M.shape[1],N-1))

    for mode_num in mode_num_list:
        print("#m+n is ",mode_num-1)
        mode_ind = len(ltemp)#mode_num*(mode_num+1)//2 #mode number included
        
        eigvalue,v = np.linalg.eig(M[:mode_ind,:mode_ind])
        eig_modes = (v.T @ HerFun[:mode_ind,:]).reshape(mode_ind,N_span,N_span) #einsum('ij,ikl->jkl',v,HerFun)

        eig_F = 2*math.pi/(1-abs(eigvalue)**2) #finesses
        eigf = (np.angle(eigvalue) + 0)/2/math.pi*FSR #eigenfrequency, need to adjust 
        for m in np.arange(len(eigf)): #Some might have one FSR off because of the np.arctan() operation
            if eigf[m] < 0:
                eigf[m] += FSR

        #==========reoder with reference to frequency=============
        sort_ind = np.argsort(eigf)
        eigf = eigf[sort_ind]
        eig_F = eig_F[sort_ind]
        eig_modes = eig_modes[sort_ind,:,:]
        v = v[:,sort_ind]

        #============pick the mode of interest============
        eig_index = []
        for i in np.arange(len(eig_F)):
            if abs(eig_F[i]) > eig_F.max()/2:
                eig_index.append(i)
        eig_F1 = eig_F[eig_index]
        eigf1 = eigf[eig_index]
        v1 = abs(v[0,eig_index])**2
        
        eigF0.append(max(eig_F1*(v1>=v1.max()))) #find the fundamental mode, can adjust to any modes
        eigv0.append(v1.max())
    
    eigF_data = np.vstack((mode_num_list-1, np.array(eigF0), np.array(eigv0))).T
    
    return M, eigF_data, 