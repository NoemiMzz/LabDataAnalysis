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
Mgas_0 = 10**10  

v_ref = 242 #km/s 
a = 3.60
b = 10.5

A = np.exp(0.8)
alpha = 0.25

#%%
### FUNCTIONS ##################################################################################################

def v_circ(mass):
    return v_ref * 10**((1/a) * np.log10(mass) - (b/a))

def radius(mass):
    return A * (mass / 5*10**10)**alpha

def t_dyn(mass):
    return 2.*10**7 * (radius(mass)/4) * (200/v_circ(mass))   #years * (kpc) * (km/s)

def eps_prime(mass):
    return epsilon / t_dyn(mass)

def sSFR_model(t, mass):
    y = np.exp(- eps_prime(t_dyn(mass)) * t)
    return eps_prime(t_dyn(mass)) * y / (1-y)

def m_star_model(t, mass):
    y = np.exp(- eps_prime(t_dyn(mass)) * t)
    return Mgas_0 * (1-y)


#%%
### VARYING DYNAMICAL TIME #####################################################################################

log_mass_by = np.log10(mass_by)
log_sSFR_by = np.log10(SFR_by) - np.log10(mass_by)

selected_ages = [sSFR_model(0.5 * 10**9, Mgas_0), sSFR_model(1 * 10**9, Mgas_0), sSFR_model(2 * 10**9, Mgas_0)]
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
### VARYING Mgas_0 #############################################################################################

masses_0 = [10**9, 10**10, 10**11]
label = ['$10^9 M_{gas0}$  -  t$dyn$='+str(np.format_float_scientific(t_dyn(masses_0[0]), precision=2)),
         '$10^{10} M_{gas0}$  -  t$dyn$='+str(np.format_float_scientific(t_dyn(masses_0[1]), precision=2)),
         '$10^{11} M_{gas0}$  -  t$dyn$='+str(np.format_float_scientific(t_dyn(masses_0[2]), precision=2)) ]
c = ['limegreen', 'orange', 'gold']

plt.figure()
plt.scatter(log_mass_by, log_sSFR_by, s=2, alpha=0.2, color='royalblue')
plt.axhline(-9.73, color='crimson', label="Main sequence of Star Forming Galaxies")
for m in range(3):
    plt.axhline(np.log10(sSFR_model(1 * 10**9, masses_0[m])), color=c[m], ls='-.', label=label[m])
plt.xlim(7, 12)
plt.ylim(-14, -8)
plt.title("Specific star formation rate")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("sSFR (1 / yrs)")
plt.legend()
plt.show()






