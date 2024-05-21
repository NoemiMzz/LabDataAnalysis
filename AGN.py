import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.optimize import curve_fit
from tqdm import tqdm

### DATA #######################################################################################################

### read spettroscopy.fits ###
data_s = Table.read('SDSS_LAB2024_spectroscopy.fits', format='fits')
data_s = data_s.to_pandas()

index = np.array(data_s.loc[:, "id"])

OIII = np.array(data_s.loc[:, "oiii_5007_flux"])
err_OIII = np.array(data_s.loc[:, "oiii_5007_flux_err"])

Hbeta = np.array(data_s.loc[:, "h_beta_flux"])
err_Hbeta = np.array(data_s.loc[:, "h_beta_flux_err"])

Halpha = np.array(data_s.loc[:, "h_alpha_flux"])
err_Halpha = np.array(data_s.loc[:, "h_alpha_flux_err"])

NII = np.array(data_s.loc[:, "nii_6584_flux"])
err_NII = np.array(data_s.loc[:, "nii_6584_flux_err"])

### read photometry.fits ###
data_p = Table.read('SDSS_LAB2024_photometry.fits', format='fits')
data_p = data_p.to_pandas()

z_data = np.array(data_p.loc[:, "redshift"])

### read CIGALEresults.fits ###
data_r = Table.read('SDSS_LAB2024_CIGALEresults.fits', format='fits')
data_r = data_r.to_pandas()

mass_data = np.array(data_r.loc[:, "bayes_stellar_m_star"])
SFR_data = np.array(data_r.loc[:, "bayes_sfh_sfr"])

N = len(OIII)

### FUNCTIONS ##################################################################################################

def kauffmann(x):
    k = np.where(x<0, 0.61 / (x - 0.05) + 1.3, -np.inf)
    return k

def line(x, a, b):
    return a*x + b 


#%%
### AGN OR STAR ionization #####################################################################################

### select the galaxies ###
SNR_OIII = OIII / err_OIII
SNR_Hbeta = Hbeta / err_Hbeta
SNR_Halpha = Halpha / err_Halpha
SNR_NII = NII / err_NII

x = []
y = []
mass = []
SFR = []
for j in tqdm(range(N)):
    #selecting galaxies with signal-to-noise ratio high enough
    if SNR_OIII[j]>5 and SNR_Hbeta[j]>5 and SNR_Halpha[j]>5 and SNR_NII[j]>5:
        x.append( np.log10(NII[j] / Halpha[j]) )
        y.append( np.log10(OIII[j] / Hbeta[j]) )
        mass.append( mass_data[j] )
        SFR.append( SFR_data[j] )
x = np.array(x)
y = np.array(y)
mass = np.array(mass)
SFR = np.array(SFR)

### plotting the selected galaxies ###
plt.figure()
plt.scatter(x, y, s=2, alpha=0.2, color='navy')
#plt.plot(np.sort(x), kauffmann(np.sort(x)), color='crimson')
plt.xlim(-2, 0.5)
plt.ylim(-2, 1.5)
plt.xlabel("$log([NII]~\\lambda 6584~/~H_{\\alpha})$")
plt.ylabel("$log([OIII]~\\lambda 5007~/~H_{\\beta})$")
plt.show()


#%%
### dividing ionization processes ###
AGN = np.array(np.where(y>=kauffmann(x))).T.flatten()   #selecting indices
star = np.array(np.where(y<kauffmann(x))).T.flatten()

x_AGN = []
y_AGN = []
z_AGN = []
mass_AGN = []
SFR_AGN = []
x_star = []
y_star = []
z_star = []
mass_star = []
SFR_star = []
for j in tqdm(AGN):   #saving arrays for AGN ionization
    x_AGN.append(x[j])
    y_AGN.append(y[j])
    mass_AGN.append(mass[j])
    SFR_AGN.append(SFR[j])
for k in tqdm(star):   #saving arrays for stellar ionization
    x_star.append(x[k])
    y_star.append(y[k])
    mass_star.append(mass[k])
    SFR_star.append(SFR[k])


###plotting the divided galaxies ###
plt.figure()
plt.scatter(x_AGN, y_AGN, s=2, alpha=0.2, color='orange', label="AGN ionization")
plt.scatter(x_star, y_star, s=2, alpha=0.2, color='royalblue', label="stellar ionization")
plt.plot(np.sort(x), kauffmann(np.sort(x)), color='crimson', label="Kauffmann et al. 2003")
plt.xlim(-2, 0.5)
plt.ylim(-2, 1.5)
plt.xlabel("$log([NII]~\\lambda 6584~/~H_{\\alpha})$")
plt.ylabel("$log([OIII]~\\lambda 5007~/~H_{\\beta})$")
plt.legend()
plt.show()


#%%
### STAR FORMATION RATE ########################################################################################

### mass and star formation rate ###
plt.figure()
plt.scatter(mass_star, SFR_star, s=2, alpha=0.2, color='royalblue', label="stellar ionization")
plt.scatter(mass_AGN, SFR_AGN, s=2, alpha=0.2, color='orange', label="AGN ionization")
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**7, 10**12)
plt.ylim(10**(-3), 10**2)
plt.title("Star formation rate")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("SFR ($M_{\odot}$ / yrs)")
plt.legend()
plt.show()


### mass and  specific star formation rate ###
log_sSFR_AGN = np.log10(SFR_AGN) - np.log10(mass_AGN)
log_mass_AGN = np.log10(mass_AGN)
log_sSFR_star = np.log10(SFR_star) - np.log10(mass_star)
log_mass_star = np.log10(mass_star)

plt.figure()
plt.scatter(log_mass_star, log_sSFR_star, s=2, alpha=0.2, color='royalblue', label="stellar ionization")
plt.scatter(log_mass_AGN, log_sSFR_AGN, s=2, alpha=0.2, color='orange', label="AGN ionization")
plt.axhline(-9.73, color='crimson', label="Main sequence of Star Forming Galaxies")
plt.xlim(7, 12)
plt.ylim(-14, -8)
plt.title("Specific star formation rate")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("sSFR (1 / yrs)")
plt.legend()
plt.show()


#%%
### FIT ########################################################################################################

log_sSFR = np.log10(SFR) - np.log10(mass)

### dividing main sequence ###
MS = np.array(np.where(log_sSFR>=-9.73)).T.flatten()   #selecting indices
red = np.array(np.where(log_sSFR<-9.73)).T.flatten()

mass_MS = []
SFR_MS = []
mass_red = []
SFR_red = []
for j in tqdm(MS):   #saving arrays for AGN ionization
    mass_MS.append(mass[j])
    SFR_MS.append(SFR[j])
for k in tqdm(red):   #saving arrays for stellar ionization
    mass_red.append(mass[k])
    SFR_red.append(SFR[k])


### mass and  specific star formation rate ###
log_sSFR_MS = np.log10(SFR_MS) - np.log10(mass_MS)
log_mass_MS = np.log10(mass_MS)
log_sSFR_red = np.log10(SFR_red) - np.log10(mass_red)
log_mass_red = np.log10(mass_red)

plt.figure()
plt.scatter(log_mass_red, log_sSFR_red, s=2, alpha=0.2, color='gainsboro', label="red passive galaxies")
plt.scatter(log_mass_MS, log_sSFR_MS, s=2, alpha=0.2, color='mediumslateblue', label="main sequence")
plt.axhline(-9.73, color='crimson', label="Main sequence of Star Forming Galaxies")
plt.xlim(7, 12)
plt.ylim(-14, -8)
plt.title("Specific star formation rate")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("sSFR (1 / yrs)")
plt.legend()
plt.show()


### fit the main sequence ###
param, err = curve_fit(line, log_mass_MS, log_sSFR_MS)
print("Fit: log(sSFR) = " + str(np.round(param[0], 2)) + " log(mass) + " + str(np.round(param[1], 2)))

x_axis = np.linspace(7, 12, 1000)

plt.figure()
plt.scatter(log_mass_MS, log_sSFR_MS, s=2, alpha=0.2, color='mediumslateblue', label="main sequence")
plt.plot(x_axis, line(x_axis, param[0], param[1]), color='gold', lw=2, label=("Fit: log(sSFR) = " + str(np.round(param[0], 2)) + " log(mass) + " + str(np.round(param[1], 2))))
plt.xlim(7, 12)
plt.ylim(-12, -6)
plt.title("Specific star formation rate")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("sSFR (1 / yrs)")
plt.legend()
plt.show()


### main sequence binning ###
#indexes associated with the mass intervals
n_bins = 25
bin_margins = np.linspace(min(log_mass_MS), max(log_mass_MS), n_bins+1)
bins = []
for b in range(n_bins):
    bins.append((log_mass_MS >= bin_margins[b]) & (log_mass_MS < bin_margins[b+1]))
plt.hist(log_mass_MS, bins=bins)
    






