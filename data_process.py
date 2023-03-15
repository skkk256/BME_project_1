from CS_mask import cartesian_mask
import numpy as np

def get_aliased(dataset):
    mask = cartesian_mask(shape=(1, 20, 192, 192), acc=10, sample_n=10, centred=True)
    cine = dataset
    cine_kspace = np.fft.fftshift(np.fft.fft2(cine))
    cine_kspace = cine_kspace *  mask
    cine_rec = np.abs(np.fft.ifft2(cine_kspace))
    return cine_rec