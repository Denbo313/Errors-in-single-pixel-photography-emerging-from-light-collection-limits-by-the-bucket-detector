# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 12:08:19 2024

This script executes the simulation

@author: Dennis Scheidt
"""
from _helper_functions import *


#%% setting of parameters 

# reading in the first data set is sufficient for this application. loop over files to get whole dataset
file_direc = r'C:\Users\D S\PhD\simulation_iswf\cifar-10-batches-py\data_batch_'
files = [str(x) for x in range(1,6)]
# outbound directory
res_direc = 'results/simulation/'


off = 1024

n = 32; m = 32


dict1 = unpickle(file_direc+files[0])

# data is stored in RGB channels
data = dict1[b'data']
r = data[:][:,:off]
g = data[:][:,off:2*off]
b = data[:][:,2*off:]
wr = 0.2126
wg = 0.7152
wb = 0.0722
# convert RGB to gray scale, applying the weigths of the human eye
gray = wr * r + wg * g + wb * b
# already reshape to 2d --> easier handle later
gray = gray.reshape((10000,n,m)).astype(float)

wid = 1.5
ns = 64 # make 
ns = 384
n = 32
x = np.linspace(-wid,wid,ns)
y = x.copy()
X,Y = np.meshgrid(x,y)

kx = 4; ky = 4
xm = 18; ym = 18
xm = 34; ym = 34
kx = 0; ky = 0
xm = ns//2; ym = ns//2
prism = prism_phase(X, Y, kx, ky)

wids = 0
x0 = np.arange(xm-wids,xm+wids+1); x0 = x0[0]+1
y0 = np.arange(ym-wids,ym+wids+1); y0 = y0[0]+1

#%% create different amplitude and phase distributions for the fields
n = ns
x = np.arange(n)
y = np.arange(n)
XX,YY = np.meshgrid(x,y)
star = spiral_mask(XX,YY)
amp_star = (star + 1)/2
phas_star  = abs(star-1)*np.pi
XX = XX.copy()
XX[XX<=ns//2-1] = 0
XX[XX>1] = 1
phase_half = XX.copy()

X,Y = np.meshgrid(x,y)

angle = np.arctan2(X-n//2,Y-n//2)
r = np.sqrt((X-n//2)**2+(Y-n//2)**2)
r[r<= ns//2] = 1
r[r>1] = 0


amp_car = gray[226].copy()/255

# boat phase
phase = gray[192].copy()/255 * 2*pi

n = ns
amp_car = np.asarray(Image.fromarray(amp_car).resize((n,n)))

gauss = np.exp(-((X-n//2)**2+(Y-n//2)**2)/1e5)
plt.imshow(gauss); plt.title('Gauss amplitude'); plt.show()

phase_32 = np.asarray(Image.fromarray(phase).resize((32,32),0).resize((n,n),0))
phase_z1 = zernike(1,1,64,outside = 1)
phase_z2 = zernike(2,2,64,outside = 1)

phase_zern = np.pi * (( 3 *pi * phase_z1/phase_z1.max() + pi * phase_z2/phase_z2.max()) % (1))
phase_zern = np.asarray(Image.fromarray(phase_zern).resize((32,32),0).resize((n,n),0))
plt.imshow(phase_zern); plt.title('Zernike phase distribution'); plt.show()

a_names = ['_gauss_','_car_']
a_names = ['_car_']
p_names = ['const_','half_','zernike_','boat_']
data_name = 'dec_size_'


amps = [np.ones((n,n)), amp_car] 
amps = [amp_car]
phases = [np.ones((n,n)), phase_half, phase_zern, phase_32]
wids = np.arange(0,ns//2+2,2)
WIDS = wids//2
M = len(wids)

WID_max = ns//4 # maximal size of the spectrum -> spread according to the size of the grid ns
#%%
N = 1024 #number of basis elements

# iterate over amplitudes
for amp, a_name in zip(amps[:],a_names):
    amp = amp * gauss
    plt.imshow(amp, cmap = 'gray'); plt.show()
    # iterate over phases
    for phase, p_name in zip(phases[:],p_names):
        name = a_name + p_name 
        plt.imshow(phase,cmap = 'bone'); plt.title('applied phase'); plt.show()
        
        data = np.zeros((N,M))
        data_b = np.zeros(N)
        
        vecs = hadamard(N)
        vecn = abs(vecs - 1) // 2 # negative part of the Hadamard basis
        vecp = abs(vecs + 1) //2 # positive part of the Hadamard basis

        # parameters for unwrapping the measurement vector and reconstructed amp and phase
        nn = int(np.sqrt(N))
        sx = n//nn
        
        # measurement over the basis vectors
        for it in range(N):
            mask = vec_2_mask(vecn[it,:], (n,n), (sx,sx)) # make 1d vector to 2d mask; negative part
            mask2 = vec_2_mask(vecp[it,:],(n,n),(sx,sx)) # positive part
            
            # sampling of the positive Hadamard basis
            new_wave = amp * mask2 *np.exp(1j*mask2 *((phase + prism)%(2*pi)))
            intensity3 = abs(fftshift(fft2(new_wave)))
            # sampling of the negative Hadamard basis
            new_wave_2 = amp * mask * np.exp(1j * mask *( (phase + prism) % (2*pi)) )
            intensity4 = abs(fftshift(fft2(new_wave_2)))
            # take the sum over different window sizes defined by wid
            for wid, m in zip(wids,range(M)):
                if wid == 0:
                    wid = 1
                data[it,m] = (intensity3[x0-wid:x0+wid,y0-wid:y0+wid]**2).sum() - (intensity4[x0-wid:x0+wid,y0-wid:y0+wid]**2).sum()
                if wid == ns//2:
                    data[it,m] = (intensity3**2).sum() - (intensity4**2).sum()
        
        for wid, m in zip(wids,range(M)):
            amps_rec = [] # save all reconstructed amplitudes, if required
            mat = vecs;
            dat = data[:,m] # choose correct reconstructed amplitude
            dat_name = data_name + '_'+str(round(wid/WID_max * 100,1))+'%_frequency_sampling_'

            rec_amp = mat@dat    # reconstruction
                     
            amplitude = vec_2_mask(rec_amp, (n,n), (sx,sx)); #nearest-neighbour upsampling to initial grid size
            amps_rec.append(amplitude)
            
            amplitude = amplitude/amplitude.max() # normalization
            
            plt.imshow(amplitude); plt.title(dat_name); plt.show()
            plt.imsave(res_direc +r'\amp'+r'/'+name + '_'+dat_name+'_'+'amp.png',amplitude,cmap = 'gray',dpi = 150,vmin = 0)
           
            