import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from tqdm import tqdm

### DATA #######################################################################################################

### read results.fits ###
data_e = Table.read('SDSS_env.fits', format='fits')
data_e = data_e.to_pandas()

ID = np.array(data_e.loc[:, "id"])

density = np.array(data_e.loc[:, "dens_05"])

### read spettroscopy.fits ###
data_s = Table.read('SDSS_LAB2024_spectroscopy.fits', format='fits')
data_s = data_s.to_pandas()

ra = np.array(data_s.loc[:, "Ra"])
dec = np.array(data_s.loc[:, "Dec"])

#%%
################################################################################################################

density_cut = density[density<30]

plt.figure()
plt.hist(density_cut, density=True, bins=25, color='deepskyblue')
plt.title("Environnemental density")
plt.xlabel("number density")
plt.show()

plt.figure(figsize=(12,10))
plt.scatter(ra, dec, c=density, s=0.1, cmap='plasma')
plt.xlabel("ra")
plt.ylabel("dec")
plt.colorbar(label="density")
plt.xlim(min(ra), max(ra))
plt.ylim(min(dec), max(dec))
plt.show()