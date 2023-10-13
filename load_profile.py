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

import os
os.sys.path.append("Z:\Code")
os.sys.path

from RoycePython import afm

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
        np.error("Give the right convexity!")

    X1 = X[start_y:start_y+len_cut, start_x:start_x+len_cut]
    Y1 = Y[start_y:start_y+len_cut, start_x:start_x+len_cut]
    Z2 = Z1[start_y:start_y+len_cut, start_x:start_x+len_cut]
    return X1,Y1,Z2

def down_sample(Z, factor):
    return 1 

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

    plt.figure()
    plt.contourf((X-xc)*1e6, (Y-yc)*1e6, Z*1e6)
    plt.axvline(x=0,linestyle='-.',color='y')
    plt.axhline(y=0,linestyle='-.',color='y')
    plt.xlabel('x (um)')
    plt.ylabel('y (um)')
    plt.colorbar(label='z(um)')
    plt.show()
    return xc,yc


def tailor(X,Y,Z): #require both X & Y are centered matrices, output are suqare matrices
    #r = min(abs(x_cut[0]-xc), abs(x_cut[-1]-xc),abs(x_cut[0]-yc),abs(x_cut[-1]-yc)) 
    r = min(abs(X[0,0]),abs(X[-1,-1]),abs(Y[0,0]),abs(Y[-1,-1])) #boundary of the effective area
    xdiff = np.diff(1*(abs(X[0,:])<r)).tolist() #list.index() doesn't work for array
    x1 = xdiff.index(1)+1
    x2 = xdiff.index(-1)+1
    ydiff = np.diff(1*(abs(Y[:,0])<r)).tolist()
    y1 = ydiff.index(1)+1
    y2 = ydiff.index(-1)+1
    print("The tailored the matrix has %d rows and %d columns" % (x2-x1+1, y2-y1+1))
    N_sample = min(x2-x1, y2-y1)
    if N_sample%2 == 1:
        N_sample-=1

    Z1 = Z[y1:y1+N_sample, x1:x1+N_sample]
    Z2 = Z1 - Z1.min()
    X1 = X[y1:y1+N_sample, x1:x1+N_sample]
    Y1 = Y[y1:y1+N_sample, x1:x1+N_sample]
    return X1,Y1,Z2

def load_file(fn):
    if fn.split('.')[-1] == 'ibw':
        X,Y,Z = afm.ibwread(fn)
    elif fn.split('.')[-1] == 'datx':
        X,Y,Z = afm.datxread(fn)
    elif fn.split('.')[-1] == 'xyz':
        X,Y,Z = afm.xyzread(fn)
    elif fn.split('.')[-1] == 'csv':
        X,Y,Z = afm.gwydread(fn)
    else: 
        np.error('Unknow file detected!')    
    print("Finished!")
    return X,Y,Z

def profile_load():
    print("Loading convex surface profile...")
    fn = r"Z:\Data\Royce\Yale Facilities\Zygo\20230609_BC005_BC004_BC001C_xcut\BC004\Lens00_FOV0p4mm_Stitch3x3.datx"
    X,Y,Z = load_file(fn)

    plt.figure()
    plt.imshow(Z)
    plt.show()

    start_x = input("Start from x0 = ")
    start_y = input("Start from x0 = ")
    cut_len = input("Cut length = ")

    X,Y,Z = level_cut(X,Y,Z, start_x, start_y, cut_len)
    xc,yc = center(X,Y,Z)
    X,Y,Z = tailor(X-xc,Y-yc,Z)
    N_sample = Z.shape[0]
    xy_reso = X[1,1]-X[0,0]   #lateral resolution

    plt.figure()
    plt.contourf((X-xc)*1e6, (Y-yc)*1e6, Z*1e9, cmap=cm.coolwarm)
    plt.axvline(x=0,linestyle='-.',color='y')
    plt.axhline(y=0,linestyle='-.',color='y')
    plt.xlabel('x (um)')
    plt.ylabel('y (um)')
    plt.title(fn.split('\\')[-1])
    plt.colorbar(label='z(nm)')
    plt.show()

    print("Load flat surface profile...")
    fn1 = r"Z:\Data\Royce\Yale Facilities\Zygo\20230609_BC005_BC004_BC001C_xcut\BC005\Flat01_SideB_FOV0p4mm_Stitch3x3.datx"
    X,Y,Z = load_file(fn1)

    if xy_reso != X[1,1]-X[0,0]:
        np.error("Backside resolution is different from dome side!")
    else: 
        X1,Y1,Z1 = level_cut(X,Y,Z,0,0,N_sample,0)

    plt.figure()
    plt.contourf((X-xc)*1e6, (Y-yc)*1e6, Z1*1e9, cmap=cm.coolwarm)
    plt.axvline(x=0,linestyle='-.',color='y')
    plt.axhline(y=0,linestyle='-.',color='y')
    plt.xlabel('x (um)')
    plt.ylabel('y (um)')
    plt.title('$\sigma$=%.2fnm'%(np.std(Z1)*1e9))
    plt.colorbar(label='z(nm)')
    plt.show()

    #========extend the simulation area==============
    m = 0
    while 2**m <= N_sample:
        m += 1
    N_span = 2**m #N_sample + N_sample//10*2 #2**m
    N1 = (N_span-N_sample)//2

    ZM_A = np.pad(Z, ((N1,N_span-N1-N_sample),(N1,N_span-N1-N_sample)), 'edge') #lens side
    ZM_B = np.pad(Z1, ((N1,N_span-N1-N_sample),(N1,N_span-N1-N_sample)), 'edge') #backside
    


    return ZM_A,ZM_B



