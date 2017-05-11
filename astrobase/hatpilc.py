import os.path
import gzip

import numpy as np

coldefs = [('rjd',float),
           ('stf',str),
           ('hat',str),
           ('xcc',float),
           ('ycc',float),
           ('xic',float),
           ('yic',float),
           ('fsv',float),
           ('fdv',float),
           ('fkv',float),
           ('bgv',float),
           ('bge',float),
           ('ifl1',float),
           ('ife1',float),
           ('irm1',float),
           ('ire1',float),
           ('irq1',str),
           ('ifl2',float),
           ('ife2',float),
           ('irm2',float),
           ('ire2',float),
           ('irq2',str),
           ('ifl3',float),
           ('ife3',float),
           ('irm3',float),
           ('ire3',float),
           ('irq3',str),
           ('iep1',float),
           ('iep2',float),
           ('iep3',float)]

def read_hatpi_txtlc(lcfile):                              

    if 'TF1' in lcfile:
        thiscoldefs = coldefs + [('itf1',float)]
    elif 'TF2' in lcfile:
        thiscoldefs = coldefs + [('itf2',float)]
    elif 'TF3' in lcfile:
        thiscoldefs = coldefs + [('itf3',float)]
        
    print('reading %s' % lcfile)

    with gzip.open(lcfile,'r') as infd:

        lclines = infd.read().decode().split('\n')
        lclines = [x.split() for x in lclines if ('#' not in x and len(x) > 0)]
        print('ndet: %s' % len(lclines))                
        
        if len(lclines) > 0:
            
            lccols = list(zip(*lclines))     
            print('ncols: %s' % len(lccols))
            lcdict = {x[0]:y for (x,y) in zip(thiscoldefs, lccols)}
            
            # convert to ndarray
            for col in thiscoldefs:
                lcdict[col[0]] = np.array([col[1](x) for x in lcdict[col[0]]])

        else:
            lcdict = None
            
    return lcdict
