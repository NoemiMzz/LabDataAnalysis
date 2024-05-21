import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

### DATA #######################################################################################################

### read CIGALEresults.fits ###
data_r = Table.read('SDSS_LAB2024_CIGALEresults.fits', format='fits')
data_r = data_r.to_pandas()

age_by = np.array(data_r.loc[:, "bayes_sfh_age"]) * 10**6   #rescaled in yr
tau_by = np.array(data_r.loc[:, "bayes_sfh_tau_main"]) * 10**6   #rescaled in yr
mass_by = np.array(data_r.loc[:, "bayes_stellar_m_star"])
SFR_by = np.array(data_r.loc[:, "bayes_sfh_sfr"])

age_ml = np.array(data_r.loc[:, "best_sfh_age"]) * 10**6   #rescaled in yr
tau_ml = np.array(data_r.loc[:, "best_sfh_tau_main"]) * 10**6   #rescaled in yr
mass_ml = np.array(data_r.loc[:, "best_stellar_m_star"])
SFR_ml = np.array(data_r.loc[:, "best_sfh_sfr"])

### parameters ###
epsilon = 0.02
t_dyn = 2.*10**7
eps_prime = epsilon / t_dyn

Mgas_0 = 10**10   

#%%
### FUNCTIONS ##################################################################################################

def t_dyn(R, vel):
    return 2.*10**7 * (R/4) * (200/vel)   #years * (kpc) * (km/s)

def sSFR_model(t):
    y = np.exp(- eps_prime * t)
    return eps_prime * y / (1-y)

def m_star_model(t):
    y = np.exp(- eps_prime * t)
    return Mgas_0 * (1-y)

#%%
### BAYES OR BEST? #############################################################################################

log_mass_by = np.log10(mass_by)
log_sSFR_by = np.log10(SFR_by) - np.log10(mass_by)

log_mass_ml = np.log10(mass_ml)
log_sSFR_ml = np.log10(SFR_ml) - np.log10(mass_ml)

plt.figure()
plt.scatter(log_mass_by, log_sSFR_by, c=age_by/tau_by, cmap='rainbow', s=2, alpha=0.2)
plt.axhline(-9.73, color='crimson', label="Main sequence of Star Forming Galaxies")
plt.xlim(7, 12)
plt.ylim(-14, -8)
plt.title("Specific star formation rate")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("sSFR (1 / yrs)")
plt.colorbar(label="age / $\\tau$")
plt.legend()
plt.show()

plt.figure()
plt.scatter(log_mass_ml, log_sSFR_ml, c=age_ml/tau_ml, cmap='rainbow', s=2, alpha=0.2)
plt.axhline(-9.73, color='crimson', label="Main sequence of Star Forming Galaxies")
plt.xlim(7, 12)
plt.ylim(-14, -8)
plt.title("Specific star formation rate")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("sSFR (1 / yrs)")
plt.colorbar(label="age / $\\tau$")
plt.legend()
plt.show()

#%%
### FIRST CLOSED BOX MODEL #####################################################################################

t = np.linspace(1, 13, 1000) * 10**9

plt.figure()
plt.plot(t, sSFR_model(t), color='royalblue')
plt.title("sSFR - closed box model")
plt.xlabel("t")
plt.ylabel("sSFR")
plt.show()

plt.figure()
plt.plot(t, m_star_model(t), color='royalblue')
plt.title("Star mass - closed box model")
plt.xlabel("t")
plt.ylabel("Star mass")
plt.show()

selected_ages = [sSFR_model(0.5 * 10**9), sSFR_model(1 * 10**9), sSFR_model(2 * 10**9)]
styles = ['-', '-.', '--']
label = ['time = 0.5 Gyr', 'time = 1 Gyr', 'time = 2 Gyr']

plt.figure()
plt.scatter(log_mass_by, log_sSFR_by, s=2, alpha=0.2, color='royalblue')
plt.axhline(-9.73, color='crimson', label="Main sequence of Star Forming Galaxies")
for i in range(len(selected_ages)):
    plt.axhline(np.log10(selected_ages[i]), color='orange', ls=styles[i], label=label[i])
plt.xlim(7, 12)
plt.ylim(-14, -8)
plt.title("Specific star formation rate")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("sSFR (1 / yrs)")
plt.legend()
plt.show()

#%%
### EVOLUTION OF THE STAR MASS #################################################################################

time = np.array([0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 3, 4, 5, 6, 7]) * 10**9
time_young = np.array([0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8]) * 10**9
time_old = np.array([2, 3, 4, 5, 6, 7]) * 10**9

plt.figure()
plt.axhline(-9.73, color='crimson', label="Main sequence of Star Forming Galaxies")
plt.scatter(np.log10(m_star_model(time_young)), np.log10(sSFR_model(time_young)), color='darkgreen')
plt.scatter(np.log10(m_star_model(time_old)), np.log10(sSFR_model(time_old)), color='limegreen')
plt.axvline(10, color='black', ls='dotted')
plt.ylim(-14, -8)
plt.title("Specific star formation rate")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("sSFR (1 / yrs)")
plt.legend()
plt.show()

plt.figure()
plt.axhline(-9.73, color='crimson', label="Main sequence of Star Forming Galaxies")
plt.scatter(np.log10(m_star_model(time)), np.log10(sSFR_model(time)), c=time, cmap='Greens_r')
plt.axvline(10, color='black', ls='dotted')
plt.ylim(-14, -8)
plt.title("Specific star formation rate")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("sSFR (1 / yrs)")
plt.legend()
plt.show()






