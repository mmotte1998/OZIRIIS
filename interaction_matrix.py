
import numpy as np
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False  # ADIEU LaTeX
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False


from plot_functions.plot_func import *

import time
from .tools_zwfs import *
from OOPAO.calibration.CalibrationVault import CalibrationVault
import tqdm
def interaction_matrix(stroke, dm, cam,nmodes = None, precropping = [20,-20,20,-20], nmeasurement:int = 1):
    if nmodes is None:
        M2C = dm.M2C
        nmodes = M2C.shape[1] #define the nbr of modes
    # nsignals = [submasks[0][submasks[0]].size, submasks[1][submasks[1]].size]
    
    out = 0
    for k_measure in range(nmeasurement):
        interactionmatrix=[]
        for i in tqdm.tqdm(range(nmodes)):
            dm.poke_mode(stroke, i)
            time.sleep(0.01)

            image = cam.get()
            image = image[precropping[0]:precropping[1],precropping[2]:precropping[3]]
            normalisation = np.sum(image)
            image /=normalisation
        
            signal_plus = np.copy(image)   
            #print(f'the shape is {signals_plus[1].shape}')
            dm.poke_mode(-stroke,i)
            time.sleep(0.01)
    
            image = cam.get()
            image = image[precropping[0]:precropping[1],precropping[2]:precropping[3]]
            image/=image.sum()
            signal_minus = np.copy(image)  
            #print(f'the shape is {signals_minus[1].shape}')
            interactionmatrix.append(0.5*(signal_plus-signal_minus)/stroke)
        dm.set_flat_surf()
        out += np.array(interactionmatrix)
    out /= nmeasurement
    
    # Masque uniquement de cette région
    return out
def interaction_matrix_Zonal(stroke, dm, cam, nmodes = None, precropping = [20,-20,20,-20], nmeasurement:int = 1):

    if nmodes is None:
        M2C = dm.M2C
        nmodes = M2C.shape[0] #define the nbr of modes
    # nsignals = [submasks[0][submasks[0]].size, submasks[1][submasks[1]].size]
    
    out = 0
    for k_measure in range(nmeasurement):
        interactionmatrix=[]
        for i in tqdm.tqdm(range(nmodes)):
            vector = np.zeros(nmodes)
            vector[i] = stroke
            dm.poke_vector(vector)
            time.sleep(0.1)

            image = cam.get()
            image = image[precropping[0]:precropping[1],precropping[2]:precropping[3]]
            normalisation = np.sum(image)
            image /=normalisation
            
            signal_plus = np.copy(image)   
            #print(f'the shape is {signals_plus[1].shape}')
            vector[i] = -stroke
            dm.poke_vector(vector)
            time.sleep(0.1)

            image = cam.get()
            image = image[precropping[0]:precropping[1],precropping[2]:precropping[3]]
            image/=image.sum()
            signal_minus = np.copy(image)  
            #print(f'the shape is {signals_minus[1].shape}')
            interactionmatrix.append(0.5*(signal_plus-signal_minus)/stroke)
        dm.set_flat_surf()
        out += np.array(interactionmatrix)
    out /= nmeasurement

    # Masque uniquement de cette région
    return out
def inversion_IM(IM, filtering = 15):
    outs = CalibrationVault(IM,invert=True, nTrunc = filtering)
    return outs
def interaction_matrix2(stroke, dm, cam, position_of_signal, submasks, invert = True):
    
    M2C = dm.M2C
    nmodes = M2C.shape[1] #define the nbr of modes
    nsignals = [submasks[0][submasks[0]].size, submasks[1][submasks[1]].size]
    intmats = [np.zeros((nsignals[0], nmodes)), np.zeros((nsignals[1], nmodes))] #predefine IM
    for i in tqdm.tqdm(range(nmodes)):
        dm.poke_mode(stroke, i)
        time.sleep(0.1)
        sub_images = []
        image = cam.get()[precropping[0]:precropping[1],precropping[2]:precropping[3]]
        normalisation = np.sum(image)
        image /=normalisation
     
        for j in range(2):
            minr, minc, maxr, maxc = position_of_signal[j]
            sub_images.append(image[minr:maxr, minc:maxc])
        sub_images[0]= pad_to_square(sub_images[0])*submasks[0]#np.pad(pupilles[0], pad_width=((1, 1), (0, 0)), mode='constant', constant_values=0)
        sub_images[1]= pad_to_square(sub_images[1])*submasks[1]#np.pad(pupilles[1], pad_width=((1, 1), (0, 0)), mode='constant', constant_values=0)
        signals_plus = []
        signals_plus.append(sub_images[0][submasks[0]].ravel())
        
        signals_plus.append(sub_images[1][submasks[1]].ravel())
        #print(f'the shape is {signals_plus[1].shape}')
        dm.poke_mode(-stroke,i)
        time.sleep(0.1)
        sub_images = []
        image = cam.get()[precropping[0]:precropping[1],precropping[2]:precropping[3]]
        for j in range(2):
            minr, minc, maxr, maxc = position_of_signal[j]
            sub_images.append(image[minr:maxr, minc:maxc])
        sub_images[0]= pad_to_square(sub_images[0])*submasks[0]#np.pad(pupilles[0], pad_width=((1, 1), (0, 0)), mode='constant', constant_values=0)
        sub_images[1]= pad_to_square(sub_images[1])*submasks[1]#np.pad(pupilles[1], pad_width=((1, 1), (0, 0)), mode='constant', constant_values=0)
        signals_minus = []
        signals_minus.append(sub_images[0][submasks[0]].ravel())
        signals_minus.append(sub_images[1][submasks[1]].ravel())
        #print(f'the shape is {signals_minus[1].shape}')
        intmats[0][:,i]=0.5*(signals_plus[0]-signals_minus[0])/stroke
        intmats[1][:,i]=0.5*(signals_plus[1]-signals_minus[1])/stroke
    outs = []
    outs.append(CalibrationVault(intmats[0],invert=invert))
    outs.append(CalibrationVault(intmats[1],invert=invert))
    dm.set_flat_surf()
    # Masque uniquement de cette région
    return outs

def interaction_matrix_OOPAO_2D(stroke, dm, tel, wfs, src, M2C):
    

    nmodes = M2C.shape[1] #define the nbr of modes
    # nsignals = [submasks[0][submasks[0]].size, submasks[1][submasks[1]].size]
    interactionmatrix=[]
    
    for i in tqdm.tqdm(range(nmodes)):
        vector = np.zeros(M2C.shape[1])
        vector[i] = stroke
        dm.coefs = M2C@vector
        src*tel*dm
        tel*wfs
        signal_plus = wfs.img_ZWFS
        #print(f'the shape is {signals_plus[1].shape}')
        vector[i] = -stroke
        dm.coefs = M2C@vector
        src*tel*dm
        tel*wfs
        signal_minus = wfs.img_ZWFS
        
        #print(f'the shape is {signals_minus[1].shape}')
        interactionmatrix.append(0.5*(signal_plus-signal_minus)/stroke)
    # out = np.array(interactionmatrix)
 
    # Masque uniquement de cette région
    return interactionmatrix

def interaction_matrix_OOPAO(stroke, dm, tel, wfs, src, M2C):
    

    nmodes = M2C.shape[1] #define the nbr of modes
    # nsignals = [submasks[0][submasks[0]].size, submasks[1][submasks[1]].size]
    interactionmatrix=[]
    
    for i in tqdm.tqdm(range(nmodes)):
        vector = np.zeros(M2C.shape[1])
        vector[i] = stroke
        dm.coefs = M2C@vector
        src*tel*dm
        tel*wfs
        signal_plus = wfs.signal
        #print(f'the shape is {signals_plus[1].shape}')
        vector[i] = -stroke
        dm.coefs = M2C@vector
        src*tel*dm
        tel*wfs
        signal_minus = wfs.signal
        
        #print(f'the shape is {signals_minus[1].shape}')
        interactionmatrix.append(0.5*(signal_plus-signal_minus)/stroke)
    # out = np.array(interactionmatrix)
 
    # Masque uniquement de cette région
    return interactionmatrix


# def interaction_matrix_zonal(stroke, dm, tel, wfs, src, M2C):
    

#     nmodes = M2C.shape[0] #define the nbr of modes
#     # nsignals = [submasks[0][submasks[0]].size, submasks[1][submasks[1]].size]
#     interactionmatrix=[]
    
#     for i in tqdm.tqdm(range(nmodes)):
#         vector = np.zeros(M2C.shape[0])
#         vector[0] = stroke
#         dm.coefs = vector
#         src*tel*dm
#         tel*wfs
#         signal_plus = wfs.signal
#         #print(f'the shape is {signals_plus[1].shape}')
#         vector[i] = -stroke
#         dm.coefs = vector
#         src*tel*dm
#         tel*wfs
#         signal_minus = wfs.signal
        
#         #print(f'the shape is {signals_minus[1].shape}')
#         interactionmatrix.append(0.5*(signal_plus-signal_minus)/stroke)
#     # out = np.array(interactionmatrix)
 
#     # Masque uniquement de cette région
#     return interactionmatrix