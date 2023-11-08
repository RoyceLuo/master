import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm

import phonon_disp as pd
import z_profile as zp
import eig_solver as es



ph_lamda = 1.55e-6/2/1.54 #phonon wavelength
k0 = 2*math.pi/ph_lamda
L = 0.5e-3 #Crystal thickness
#===========load mirror profiles=================
fp = r"Z:\Data\Royce\Yale Facilities\Zygo\20230609_BC005_BC004_BC001C_xcut\BC004\\"
fn1 = fp + r"Lens00_FOV0p4mm_Stitch3x3.datx"
fn2 = fp + r"Flat01_SideB_FOV0p4mm_Stitch3x3.datx"
ZB,ZA,xy_reso,r = zp.profile_load(fn1,fn2,4) #ZB dome, ZA flat, downsample factor
#ZA, xy_reso = zp.down_sample(ZA,xy_reso,4)
#ZB, xy_reso = zp.down_sample(ZB,xy_reso,4)

N_span = ZB.shape[0]
print("Mirror size %d by %d. Leteral resolution is %.2fum"%(N_span,N_span, xy_reso*1e6))
print("Mirror radius is %.2fum. Sim radius is %.2fum"%(r*1e6, xy_reso*N_span/2*1e6))
#=========phonon dispersion=======================
popt = pd.disp_cal('z') #get fitted parameters of slowness surface
P_aniso = -1/(2*popt[0]*popt[1])
fB = 1/ph_lamda/popt[0] #Brillouin frequency
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
w0_init = 31e-6
w0 = es.opt_M00(w0_init,k0, X,Y, prop_phi,mirra_phi,mirrb_phi)
print("Optimized beam waist is %.2fum"%(w0*1e6))

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


#===============convergence test==================
mode_conv = 25
w0_list = np.array([20,25,35,40,45,50])*1e-6
F_list = []
for m in np.arange(len(w0_list)):
    w0_sim = w0_list[m]
    conv_test = es.sol_conv(mode_conv,w0_sim,k0,FSR, X,Y,win_fun, prop_phi,mirra_phi,mirrb_phi)
    F_list.append(conv_test[:,1])
#plt.plot(conv_test[:,0],conv_test[:,1])
F_list = np.vstack((conv_test[:,0],np.array(F_list))).T

fig = plt.figure()
ax = fig.gca()
for m in np.arange(len(w0_list)):
    plt.plot(F_list[:,0], F_list[:,m+1], label='w0=%dum'%(w0_list[m]*1e6))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('m + n')
plt.ylabel('Finesse')
plt.title('sim w0=%.2fum'%(w0_sim*1e6))
plt.legend()
plt.tight_layout()
plt.savefig(fp + r"Convergence_test2_20231018\Lens00_r%dum_simw0%dum.pdf")
plt.show()

np.savetxt(fp + r"Convergence_test2_20231018\Lens00_r%dum_simw0%dum.csv"\
%(r*1e6,w0*1e6), F_list, delimiter=',', fmt='%.2f', header='Mode#(m+n), Finesse\n')


'''
#===================main solver part=====================
mode_num = 11
eigvalue,v,eig_modes = es.solve(mode_num,w0,k0, X,Y,win_fun, prop_phi,mirra_phi,mirrb_phi)

eig_F = 2*math.pi/(1-abs(eigvalue)**2)
eigf = np.angle(eigvalue + 0)/2/math.pi*FSR
for m in np.arange(len(eigf)): ### np.angle() returns (-math.pi,math.pi]
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
v1 = abs(v[:,eig_index])**2

#================plot=============
fig = plt.figure('Eigenvalues',figsize=(12,10))
ax1 = fig.add_subplot(221,projection='polar')
ax1.plot(np.angle(eigvalue),abs(eigvalue),color='k', linestyle=':', linewidth=1,\
        marker='o', markersize=8, markeredgecolor='black', markerfacecolor='C3')
plt.rgrids((np.arange(6)+1)*0.2)

ax2 = fig.add_subplot(222)
ax2_ = ax2.twinx()
ax2.plot((eigf-eigf[0])/1e6,color='k', linestyle=':', linewidth=1,\
        marker='o', markersize=8, markeredgecolor='black', markerfacecolor='C3')
#ax2.set_xlabel('Peak order',fontsize=14)
ax2.set_ylabel('Resonant frequecy(MHz)',color='C3',fontsize=14)
ax2_.plot(eig_F/1e3,color='k', linestyle=':', linewidth=1,\
        marker='o', markersize=8, markeredgecolor='black', markerfacecolor='C0')
ax2_.set_ylabel(r'Finesse($\times10^3$)',color='C0',fontsize=14)

m = 0
Nt = 150
ax3 = fig.add_subplot(223)
cf3 = ax3.contourf(x_zp[:-2]*1e6, x_zp[:-2]*1e6, (abs(eig_modes[m,:-2,:-2])/abs(eig_modes[m,:,:]).max())**2 + win_fun2/win_fun2.max(), cmap=cm.hot)
ax3.set_xlim([-Nt,Nt])
ax3.set_ylim([-Nt,Nt])
ax3.set_title('Mode %d'%m,fontsize=14)
ax3.set_xlabel('x(um)',fontsize=14)
ax3.set_ylabel('y(um)',fontsize=14)
fig.colorbar(cf3, ax=ax3)

ax4 = fig.add_subplot(224)
ax4.bar(np.arange(v.shape[0]),abs(v[:,m])**2)
#ax4.set_xlim([-0.5,10.5])
ax4.set_title('Coefficients of Mode %d'%m,fontsize=14)
ax4.text(5,0.1,'Finesse is %d'%(eig_F[m]),color='r',fontsize=14)
ax4.text(5,0.01,'Q is %.2f million!'%(eig_F[m]*k0*L/math.pi/1e6),color='r',fontsize=14)
ax4.set_yscale('log')
ax4.set_xlabel('Peak order',fontsize=14)
plt.show()
'''




