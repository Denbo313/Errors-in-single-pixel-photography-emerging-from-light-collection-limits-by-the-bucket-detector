# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:04:44 2024

@author: D S
"""

from _helper_functions import *

def sublist(ls,key):
    """
    helper function to extract a list from a list that matches a certain criterion
    """
    return [match for match in ls if key in match]

def SPI_rec(vec,y):
    """
    reconstruct the sampled vector y of the image u with the Hadamard matrix by y = H u, according to:
    x = H y = H H u = N u
    """
    return vec @ y

direc_in = 'experimental_data/'
direc_out = 'results/experiment/'

direcs = ['complex_hologram/','tilted_lens/']

roi_sizes  = np.arange(0,33)*8 + 1

N = 1024 # number of measurements
n = 32 # = sqrt(N), dimensions in x and y
roi_max = 200
size = 384
spixel = 12

dats = []

# iterate over directories
for direc in direcs:
    print(direc)
    filelist = os.listdir(direc_in + direc)
    # set parameters depending on the evaluation
    if 'complex' in direc:
        cases1 = ['Car_half','Car_zernike','Car_boat']
        cases2 = ['','']
        end = 1
    elif 'tilted_lens' in direc:
        cases1 = ['aligned','slightly','strongly']
        cases2 = ['Gauss_','Car_']
        end = 2
    # iterate over the different possibilities
    for CASE in cases1:
        files1 = sublist(filelist,CASE)
        for iterator in range(end):    
            files = sublist(files1,cases2[iterator])
            # iterate over the different roi sizes
            for roi in roi_sizes:
                # loading of the data
                ROI = '_'+str(roi)+'_'
                file = sublist(files,ROI)
                print(file)
                data = loadmat(direc_in + direc + file[0])
                M = data['vecs'] # sampling matrix
                data = data['data'].T # sampled data
                dats.append(data)
                #image reconstruction
                rec_img = SPI_rec(M,data)
                rec_img = vec_2_mask(rec_img, (size,size), (spixel,spixel))
                plt.imsave(direc_out+direc+cases2[iterator]+CASE+'_'+str(round((roi)/roi_max*100,1))+'%_frequency_sampling.png',rec_img,cmap = 'gray')
                
                