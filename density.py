import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from tqdm import tqdm

### DATA #######################################################################################################

### read results.fits ###
data = Table.read('SDSS_env.fits', format='fits')
data = data.to_pandas()

ID = np.array(data.loc[:, "id"])

density = np.array(data.loc[:, "dens_05"])

density_cut = density[density<30]

plt.figure()
plt.hist(density_cut, density=True, bins=25, color='deepskyblue')
plt.title("Boh")
plt.xlabel("number density")
plt.show()