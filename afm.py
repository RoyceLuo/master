# -*- coding: utf-8 -*-
"""
Spyder Editor

This program read various type of AFM profile data and 
export them in 3 matrices, all in the unit of m.
"""

import numpy as np
import h5py
#import igor.igorpy as igorp
#import igor.binarywave as igorb

def gwydread(path): #Read .csv or .txt file from Gwyddion 
    zz = np.loadtxt(path, delimiter='\t', skiprows=4)
    f = open(path, 'r')
    f.readline()
    lx = float(f.readline().split(' ')[2])
    ly = float(f.readline().split(' ')[2])
    f.close()
    
    xs = np.linspace(0,lx,zz.shape[0])
    ys = np.linspace(0,ly,zz.shape[1])
    xx,yy = np.meshgrid(xs,ys) 
    return xx,yy,zz
    
'''
def ibwread(path): #Read .ibw file from Asylum AFM
    data = igorb.load(path)
    xy_reso = data['wave']['wave_header']['sfA'][0]
    zz = data['wave']['wData'][:,:,0]
    ly,lx = np.shape(zz)
    xs = np.arange(lx)*xy_reso
    ys = np.arange(ly)*xy_reso
    xx,yy = np.meshgrid(xs,ys)  
    return xx,yy,zz

'''

def xyzread(path): #Read .xyz file from Zygo
    f = open(path, 'r')
    for i in np.arange(7):
        f.readline()
    xy_reso = float(f.readline().split(' ')[-2])
    [m,n] = [int(x) for x in f.readline().split(' ')[0:2]] ### sometimes it's not square matrix
    f.close()

    data = np.genfromtxt(path, delimiter=' ', skip_header=13, usecols=(2,))
    xs = np.arange(m) * xy_reso
    ys = np.arange(n) * xy_reso
    xx,yy = np.meshgrid(xs,ys)
    zz = data.reshape(n,m)*1e-6 #Original data is in micrometer
    return xx,yy,zz



def datxread(path): #Read .datx file from Zygo
    f = datx2py(path)
    zdata = f['Measurement']['Surface']
    zz = zdata['vals']*1e-9 #Original data is in nanometer
    xunit = zdata['attrs']['X Converter']['Parameters'][1]
    yunit = zdata['attrs']['Y Converter']['Parameters'][1]
    ly,lx = np.shape(zz)
    xs = np.arange(lx)*xunit
    ys = np.arange(ly)*yunit
    xx,yy = np.meshgrid(xs,ys)
    return xx,yy,zz


def datx2py(file_name):
    # unpack an h5 group into a dict
    def _group2dict(obj):
        return {k: _decode_h5(v) for k, v in zip(obj.keys(), obj.values())}

    # unpack a numpy structured array into a dict
    def _struct2dict(obj):
        names = obj.dtype.names
        return [dict(zip(names, _decode_h5(record))) for record in obj]

    # decode h5py.File object and all of its elements recursively
    def _decode_h5(obj):
        # group -> dict
        if isinstance(obj, h5py.Group):
            d = _group2dict(obj)
            if len(obj.attrs):
                d['attrs'] = _decode_h5(obj.attrs)
            return d
        # attributes -> dict
        elif isinstance(obj, h5py.AttributeManager):
            return _group2dict(obj)
        # dataset -> numpy array if not empty
        elif isinstance(obj, h5py.Dataset):
            d = {'attrs': _decode_h5(obj.attrs)}
            try:
                d['vals'] = obj[()]
            except (OSError, TypeError):
                pass
            return d
        # numpy array -> unpack if possible
        elif isinstance(obj, np.ndarray):
            if np.issubdtype(obj.dtype, np.number) and obj.shape == (1,):
                return obj[0]
            elif obj.dtype == 'object':
                return _decode_h5([_decode_h5(o) for o in obj])
            elif np.issubdtype(obj.dtype, np.void):
                return _decode_h5(_struct2dict(obj))
            else:
                return obj
        # dimension converter -> dict
        elif isinstance(obj, np.void):
            return _decode_h5([_decode_h5(o) for o in obj])
        # bytes -> str
        elif isinstance(obj, bytes):
            return obj.decode()
        # collection -> unpack if length is 1
        elif isinstance(obj, list) or isinstance(obj, tuple):
            if len(obj) == 1:
                return obj[0]
            else:
                return obj
        # other stuff
        else:
            return obj

    # open the file and decode it
    with h5py.File(file_name, 'r') as f:
        h5data = _decode_h5(f)

    return h5data


