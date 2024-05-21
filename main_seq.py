import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

### DATA #######################################################################################################

### read results.fits ###
data = Table.read('out/results.fits', format='fits')
data = data.to_pandas()

mass = np.array(data.loc[:, "best.stellar.m_star"])

SFR10 = np.array(data.loc[:, "bayes.sfh.sfr10Myrs"])
err_SFR10 = np.array(data.loc[:, "bayes.sfh.sfr10Myrs_err"])

age = np.array(data.loc[:, "best.sfh.age"])

tau = np.array(data.loc[:, "best.sfh.tau_main"])

dust_old = np.array(data.loc[:, "best.attenuation.E_BVs.stellar.old"])
dust_young = np.array(data.loc[:, "best.attenuation.E_BVs.stellar.young"])

#%%
### SFR VS MASS ################################################################################################

### age plot ###
plt.figure()
plt.scatter(mass, SFR10, c=age, cmap='rainbow', s=0.7, alpha=0.5)
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**7, 10**12)
plt.ylim(10**(-3), 10**2)
plt.title("Age")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("SFR ($M_{\odot}$ / yrs)")
plt.colorbar(label="age")
plt.show()

### age/tau plot ###
plt.figure()
plt.scatter(mass, SFR10, c=age/tau, cmap='rainbow', s=0.7, alpha=0.5)
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**7, 10**12)
plt.ylim(10**(-3), 10**2)
plt.title("Age / $\\tau$")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("SFR ($M_{\odot}$ / yrs)")
plt.colorbar(label="age / $\\tau$")
plt.show()

### attenuation plots ###
plt.figure()
plt.scatter(mass, SFR10, c=dust_old, cmap='rainbow', s=0.7, alpha=0.5)   #with old dust
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**7, 10**12)
plt.ylim(10**(-3), 10**2)
plt.title("E(B-V) old")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("SFR ($M_{\odot}$ / yrs)")
plt.colorbar(label="E(B-V) old")
plt.show()

plt.figure()
plt.scatter(mass, SFR10, c=dust_young, cmap='rainbow', s=0.7, alpha=0.5)   #with young dust
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**7, 10**12)
plt.ylim(10**(-3), 10**2)
plt.title("E(B-V) young")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("SFR ($M_{\odot}$ / yrs)")
plt.colorbar(label="E(B-V) young")
plt.show()

#%%
### SPECIFIC SFR VS MASS #######################################################################################

#definig the log of the specific star formation rate
log_sSFR = np.log10(SFR10) - np.log10(mass)
log_mass = np.log10(mass)

### specific star formation rate plot ###
plt.figure()
plt.scatter(log_mass, log_sSFR, c=age/tau, cmap='rainbow', s=0.7, alpha=0.5)   #sSFR
plt.xlim(7, 12)
plt.ylim(-14, -8)
plt.title("Specific star formation rate")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("sSFR (1 / yrs)")
plt.colorbar(label="age / $\\tau$")
plt.show()

#showing the grid used for the histograms
plt.figure()
plt.scatter(log_mass, log_sSFR, c=age/tau, cmap='rainbow', s=0.7, alpha=0.5)   #sSFR
plt.xlim(7, 12)
plt.ylim(-14, -8)
plt.title("Specific star formation rate")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("sSFR (1 / yrs)")
plt.colorbar(label="age / $\\tau$")
plt.axhline(-9.73, color='red')
plt.axvline(8, color='black', ls='dotted')
plt.axvline(8.75, color='black', ls='dotted')
plt.axvline(9.5, color='black', ls='dotted')
plt.axvline(10.25, color='black', ls='dotted')
plt.axvline(11, color='black', ls='dotted')
plt.show()

#indexes associated with the mass intervals
bin1 = (log_mass>=8) & (log_mass<8.75)
bin2 = (log_mass>=8.75) & (log_mass<9.5)
bin3 = (log_mass>=9.5) & (log_mass<10.25)
bin4 = (log_mass>=10.25) & (log_mass<11)

### bimodal plot ###
plt.figure()
plt.axvline(-9.73, color='red')
plt.hist(log_sSFR[bin1], bins=50, density=True, color='royalblue', label="$10^{8} < m <10^{8.75}$", alpha=0.5)
plt.hist(log_sSFR[bin4], bins=50, density=True, color='crimson', label="$10^{10.25} < m <10^{11}$", alpha=0.5)
plt.title("Bimodal plot")
plt.xlabel("$log_{10}(~sSFR (1 / yrs)~)$")
plt.ylabel("#galaxies")
plt.legend()
plt.xlim(-12, -8)
plt.show()

### plot of all the mass intervals ###
plt.figure()
plt.axvline(-9.73, color='red')
plt.hist(log_sSFR[bin1], bins=50, density=True, color='royalblue', label="$10^{8} < m <10^{8.75}$", alpha=0.5)
plt.hist(log_sSFR[bin2], bins=50, density=True, color='limegreen', label="$10^{8.75} < m <10^{9.5}$", alpha=0.5)
plt.hist(log_sSFR[bin3], bins=50, density=True, color='gold', label="$10^{9.5} < m <10^{10.25}$", alpha=0.5)
plt.hist(log_sSFR[bin4], bins=50, density=True, color='crimson', label="$10^{10.25} < m <10^{11}$", alpha=0.5)
plt.title("All sections")
plt.xlabel("$log_{10}(~sSFR (1 / yrs)~)$")
plt.ylabel("#galaxies")
plt.legend()
plt.xlim(-12, -8)
plt.show()