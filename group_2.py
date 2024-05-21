import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.optimize import curve_fit

### DATA #######################################################################################################

### read CIGALEresults.fits ###
data = Table.read('SDSS_LAB2024_CIGALEresults_2.0.fits', format='fits')
data = data.to_pandas()

mass = np.array(data.loc[:, "bayes.stellar.m_star"])
err_mass = np.array(data.loc[:, "bayes.stellar.m_star_err"])
SFR = np.array(data.loc[:, "bayes.sfh.sfr"])
err_SFR = np.array(data.loc[:, "bayes.sfh.sfr_err"])

age = np.array(data.loc[:, "best.sfh.age"]) * 10**6   #rescaled in yr
tau = np.array(data.loc[:, "best.sfh.tau_main"]) * 10**6   #rescaled in yr

### parameters ###
epsilon = 0.02
t_dyn = 2.*10**7
eps_prime = epsilon / t_dyn

Mgas_0 = 10**10

T = -10   #main sequence treshold

N = len(mass)


#%%
### FUNCTIONS ##################################################################################################

def t_dyn(R, vel):
    return 2.*10**7 * (R/4) * (200/vel)   #years * (kpc) * (km/s)

def sSFR_model(t):
    y = np.exp(- eps_prime * t)
    return eps_prime * y / (1-y)

def line(x, a, b):
    return a*x + b 


#%%
### SFR VS MASS ################################################################################################

#definig the log of the specific star formation rate
log_sSFR = np.log10(SFR) - np.log10(mass)
log_mass = np.log10(mass)

#prova a marginalizzare se hai tempo e voglia per trovare T

plt.figure()
plt.scatter(log_mass, np.log10(SFR), c=age, cmap='rainbow', s=2, alpha=0.2)
plt.xlim(7, 12)
plt.ylim(-4, 2)
plt.title("Star formation rate")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("SFR (1 / yrs)")
plt.colorbar(label="age")
plt.show()

plt.figure()
plt.scatter(log_mass, log_sSFR, c=age, cmap='rainbow', s=2, alpha=0.2)
plt.axhline(T, color='crimson', label="Main sequence of Star Forming Galaxies")
plt.xlim(7, 12)
plt.ylim(-14, -8)
plt.title("Specific star formation rate")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("sSFR (1 / yrs)")
plt.colorbar(label="age")
plt.legend()
plt.show()


#%%
### CLOSED BOX MODEL ###########################################################################################

plt.figure()
plt.scatter(log_mass, np.log10( sSFR_model(age) ), c=age, cmap='rainbow', s=2, alpha=0.2)
plt.axhline(T, color='crimson', label="Main sequence of Star Forming Galaxies")
plt.xlim(7, 12)
plt.ylim(-14, -8)
plt.title("Specific star formation rate")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("sSFR (1 / yrs)")
plt.colorbar(label="age")
plt.legend()
plt.show()


#%%
### FIT MS #####################################################################################################

log_sSFR_computed = np.log10( sSFR_model(age) )

MS_data = np.array(np.where(log_sSFR>=T)).T.flatten()   #selecting indices of MS galaxies
MS_model = np.array(np.where(log_sSFR_computed>=T)).T.flatten()


### fit the main sequence - model ###
param_m, err_m = curve_fit(line, log_mass[MS_model], log_sSFR_computed[MS_model])
print("\nsSFR computed with model")
print("log(sSFR) = " + str(np.round(param_m[0], 2)) + " log(mass) + " + str(np.round(param_m[1], 2)))
print("slope:", np.round(param_m[0], 3), "error:", np.round(np.sqrt(err_m[0,0]), 3))
print("intercept:", np.round(param_m[1], 3), "error:", np.round(np.sqrt(err_m[1,1]), 3))

x_axis = np.linspace(7, 12, 1000)

plt.figure()
plt.scatter(log_mass, log_sSFR_computed, color='gainsboro', s=2, alpha=0.2)
plt.scatter(log_mass[MS_model], log_sSFR_computed[MS_model], s=2, alpha=0.2, color='navy', label="main sequence")
plt.plot(x_axis, line(x_axis, param_m[0], param_m[1]), color='gold', lw=2, label=("Fit: log(sSFR) = " + str(np.round(param_m[0], 2)) + " log(mass) + " + str(np.round(param_m[1], 2))))
plt.xlim(7, 12)
plt.ylim(-12, -6)
plt.title("Specific star formation rate - model")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("sSFR (1 / yrs)")
plt.legend()
plt.show()

#%%
### fit the main sequence - data ###
param_d, err_d = curve_fit(line, log_mass[MS_data], log_sSFR[MS_data])
print("\nsSFR from dataset")
print("log(sSFR) = " + str(np.round(param_d[0], 2)) + " log(mass) + " + str(np.round(param_d[1], 2)))
print("slope:", np.round(param_d[0], 4), "error:", np.round(np.sqrt(err_d[0,0]), 4))
print("intercept:", np.round(param_d[1], 3), "error:", np.round(np.sqrt(err_d[1,1]), 3))

x_axis = np.linspace(7, 12, 1000)

plt.figure()
plt.scatter(log_mass, log_sSFR, color='gainsboro', s=2, alpha=0.2)
plt.scatter(log_mass[MS_data], log_sSFR[MS_data], s=2, alpha=0.2, color='navy', label="main sequence")
plt.plot(x_axis, line(x_axis, param_d[0], param_d[1]), color='gold', lw=2, label=("Fit: log(sSFR) = " + str(np.round(param_d[0], 2)) + " log(mass) + " + str(np.round(param_d[1], 2))))
plt.xlim(7, 12)
plt.ylim(-12, -6)
plt.title("Specific star formation rate - data")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("sSFR (1 / yrs)")
plt.legend()
plt.show()


### plot fits ###
plt.figure()
plt.plot(x_axis, line(x_axis, param_m[0], param_m[1]), color='gold', lw=2, label=("Fit: log(sSFR) = " + str(np.round(param_m[0], 2)) + " log(mass) + " + str(np.round(param_m[1], 2))))
plt.plot(x_axis, line(x_axis, param_d[0], param_d[1]), color='gold', lw=2, label=("Fit: log(sSFR) = " + str(np.round(param_d[0], 2)) + " log(mass) + " + str(np.round(param_d[1], 2))))
plt.title("Specific star formation rate - data")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("sSFR (1 / yrs)")
plt.legend()
plt.show()

















