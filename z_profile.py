import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (9, 6),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

import afm


#==============define functions==============

def load_file(fn, Sfactor):
    if fn.split('.')[-1] == 'ibw':
        X,Y,Z = afm.ibwread(fn)
    elif fn.split('.')[-1] == 'datx':
        X,Y,Z = afm.datxread(fn)
    elif fn.split('.')[-1] == 'xyz':
        X,Y,Z = afm.xyzread(fn)
    elif fn.split('.')[-1] == 'csv':
        X,Y,Z = afm.gwydread(fn)
    else: 
        print('Unknow file detected!')    
    print("Finished!")
    X = down_sample(X, Sfactor)
    Y = down_sample(Y, Sfactor)
    Z = down_sample(Z, Sfactor)
    return X,Y,Z

def del_nan(Z): #get rid of the nan data points
    for m in np.arange(Z.shape[0]):
        for n in np.arange(Z.shape[1]):
            if np.isnan(Z[m,n]) or abs(Z[m,n])>1:
                Z[m,n] = Z[m,n-1]
            else: 
                pass
    return Z


def level_cut(X,Y,Z, start_x,start_y,len_cut, convex): #for phononic resonators, convexity = 1
    Num1 = 10
    pos1 = (Num1,Num1)
    pos2 = (-Num1,Num1)
    pos3 = (Num1,-Num1)
    pos4 = (-Num1,-Num1)
    P1 = np.array([X[pos1],Y[pos1],Z[pos1]])
    P2 = np.array([X[pos2],Y[pos2],Z[pos2]])
    P3 = np.array([X[pos3],Y[pos3],Z[pos3]])
    P4 = np.array([X[pos4],Y[pos4],Z[pos4]])

    if convex == 1:
        Z1 = genplane(X,Y,P1,P3,P4) - Z
    elif convex == 0:
        Z1 = Z - genplane(X,Y,P1,P3,P4)
    else:
        print("Give the right convexity!")

    X1 = X[start_y:start_y+len_cut, start_x:start_x+len_cut]
    Y1 = Y[start_y:start_y+len_cut, start_x:start_x+len_cut]
    Z2 = Z1[start_y:start_y+len_cut, start_x:start_x+len_cut]

    return X1, Y1, Z2

def down_sample(ZM,Sfactor):
    if not isinstance(Sfactor, int):
        print("Downsampling factor has to be integer!")
    elif Sfactor == 1:
        ZM1 = ZM
    else:
        N_sample1 = ZM.shape[0]//Sfactor
        ZM1 = np.zeros((N_sample1, N_sample1))
        #reso1 = reso*Sfactor
        for m in np.arange(N_sample1):
            for n in np.arange(N_sample1):
                ZM1[m,n] = ZM[m*Sfactor,n*Sfactor]
    return ZM1 #,reso1

def genplane(X,Y,p1,p2,p3):
    v1 = p2-p1
    v2 = p3-p1
    v_m = np.array([[v1[0],v1[1]],[v2[0],v2[1]]])
    v_y = -1*np.array([v1[2],v2[2]])
    n_vec = np.dot(np.linalg.inv(v_m),v_y)
    Z_plane = p1[2]-(n_vec[0]*(X-p1[0]) + n_vec[1]*(Y-p1[1])) 
    return Z_plane


def center(X,Y,Z): #center the profile
    X_temp = X*(Z<=Z.min()) # has value only at the minimumals, 0 elsewhere
    Y_temp = Y*(Z<=Z.min())
    xc = np.sum(X_temp)/np.sum(X_temp!=0)
    yc = np.sum(Y_temp)/np.sum(Y_temp!=0)
    return xc,yc


def trim_zeros_2d(X,mask): #require both X & Y are centered, output are suqare matrices with even rank, centered at the minimum point
    mask_x = np.all(mask==0,1) #find all-zero rows
    mask_y = np.all(mask==0,0) #find all-zero cols
    X1 = np.delete(np.delete(X,mask_x,0),mask_y,1)
    return X1


def profile_load(fn,fn1, Sfactor):
    print("Loading convex surface profile...")
    #fn = r"Z:\Data\Royce\Yale Facilities\Zygo\20230609_BC005_BC004_BC001C_xcut\BC004\Lens00_FOV0p4mm_Stitch3x3.datx"
    X,Y,Z = load_file(fn, Sfactor)
    Z = del_nan(Z)

    '''
    plt.figure()
    plt.imshow((Z-Z.min())*1e9)
    plt.colorbar(label='z(nm)')
    plt.show()
    '''
    start_x = 0 #int(input("Start from x0 = "))
    start_y = 0 #int(input("Start from y0 = "))
    cut_len = -1#int(input("Cut length = "))

    X,Y,Z = level_cut(X,Y,Z, start_x, start_y, cut_len, 1)

    xc,yc = center(X,Y,Z) #need to set the center manually if cannot find the minimum point
    X = X-xc 
    Y = Y-yc

    r = min(abs(X[0,0]),abs(X[-1,-1]),abs(Y[0,0]),abs(Y[-1,-1])) #boundary of the effective area
    mask = (abs(X)<r) * (abs(Y)<r)

    #=======tailor dome surface========
    XA = trim_zeros_2d(X,mask)
    YA = trim_zeros_2d(Y,mask)
    ZA = trim_zeros_2d(Z,mask)

    N_sample = max(ZA.shape[0],ZA.shape[1])
    xy_reso = X[1,1]-X[0,0]   #lateral resolution

    plt.figure()
    plt.contourf(XA*1e6, YA*1e6, ZA*1e9, cmap=cm.coolwarm)
    plt.axvline(x=0,linestyle='-.',color='y')
    plt.axhline(y=0,linestyle='-.',color='y')
    plt.xlabel('x (um)')
    plt.ylabel('y (um)')
    plt.title(fn.split('\\')[-1])
    plt.colorbar(label='z(nm)')
    plt.show()


    #==============flat surface================
    print("Load flat surface profile...")
    #fn1 = r"Z:\Data\Royce\Yale Facilities\Zygo\20230609_BC005_BC004_BC001C_xcut\BC005\Flat01_SideB_FOV0p4mm_Stitch3x3.datx"
    X,Y,Z = load_file(fn1,Sfactor)

    if abs(X[1,1]-X[0,0]-xy_reso) > 1e-15:
        print("Backside resolution: %.2fum, dome side: %.2fum."%((X[1,1]-X[0,0])*1e6, xy_reso*1e6))
    else: 
        X1,Y1,Z1 = level_cut(X,Y,Z,0,0,N_sample,1)

    XB,YB,ZB = level_cut(X1,Y1,Z1, 0,0,-1, 1)

    plt.figure()
    plt.contourf(XB*1e6, YB*1e6, ZB*1e9, cmap=cm.coolwarm)
    plt.xlabel('x (um)')
    plt.ylabel('y (um)')
    plt.title('$\sigma$=%.2fnm'%(np.std(ZB)*1e9))
    plt.colorbar(label='z(nm)')
    plt.show()

    #========extend the simulation area==============
    m = 0
    while 2**m <= N_sample:
        m += 1
    N_span = 2**m #N_sample + N_sample//10*2 #2**m
    N1 = (N_span-N_sample)//2

    ZM_A = np.pad(ZA, ((N1,N_span-N1-ZA.shape[0]),(N1,N_span-N1-ZA.shape[1])), 'edge') #lens side
    ZM_B = np.pad(ZB, ((N1,N_span-N1-ZB.shape[0]),(N1,N_span-N1-ZB.shape[1])), 'edge') #backside
    
    return ZM_A, ZM_B, xy_reso, r


def arti_profile():
    return 1

