import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.optimize import curve_fit
import astropy.units as u
from astropy.cosmology import Planck13, z_at_value
from tqdm import tqdm

### DATA #######################################################################################################

### read CIGALEresults.fits ###
data = Table.read('SDSS_LAB2024_CIGALEresults_2.0.fits', format='fits')
data = data.to_pandas()

#mass = np.array(data.loc[:, "bayes.stellar.m_star"])
#err_mass = np.array(data.loc[:, "bayes.stellar.m_star_err"])
#SFR = np.array(data.loc[:, "bayes.sfh.sfr"])
#err_SFR = np.array(data.loc[:, "bayes.sfh.sfr_err"])

mass = np.array(data.loc[:, "best.stellar.m_star"])
SFR = np.array(data.loc[:, "best.sfh.sfr"])

age = np.array(data.loc[:, "best.sfh.age"]) * 10**6   #rescaled in yr
tau = np.array(data.loc[:, "best.sfh.tau_main"]) * 10**6   #rescaled in yr

### parameters ###
epsilon = 0.02
t_dyn = 2.*10**7
eps_prime = epsilon / t_dyn

Mgas_0 = 10**10

T = -10.25   #main sequence treshold

### other parameters ###
N = len(mass)
alpha = 0.05

# set global parameters 
dt= 0.01 #Gyr
eta = 1.0
R = 0.1
f_b= 0.15
M_h_min = 1.e9
M_h_max = 10.0**11.6
#M_h_form = 5.e8 #dm halo mass at t_form
M_h_form_isfixed = False


#%%
### FUNCTIONS ##################################################################################################

def t_dyn(R, vel):
    return 2.*10**7 * (R/4) * (200/vel)   #years * (kpc) * (km/s)

def sSFR_model(t):
    y = np.exp(- eps_prime * t)
    return eps_prime * y / (1-y)

def line(x, a, b):
    return a*x + b 

def evolve_galaxy(t_form, t_obs, M_h_in):
    #initialize arrays
    M_h = t*0.
    M_g = t*0.
    M_s = t*0.
    SFR_open = t*0.
    i = 0 
    while i < len(t):
        if t[i] < t_form:
            M_h[i] = M_h_in
        elif t[i] < t_obs:
            M_dot_h = 42.0 * ((M_h[i-1]/1.e12)**1.127) * (1+1.17 * z[i])*(0.3*(1+z[i])**3+0.7)**0.5 #M_sun/yr
            M_h[i] = M_h[i-1] + M_dot_h*dt*1.e9
            if M_h[i]< M_h_min:
                csi=0.0
            elif M_h[i]> M_h_min and M_h[i]< M_h_max:
                csi=1.0
            else:
                csi=0.0        #csi= (M_h_max/M_h[i])
            M_g[i]= M_g[i-1] + dt*1.e9*(f_b* csi* M_dot_h -epsilon*(1+eta-R)*M_g[i-1]/t_dyn[i])
            SFR_open[i] = epsilon * M_g[i] /t_dyn[i]
            M_s[i] = M_s[i-1] + SFR_open[i]*dt*1.e9
        else:
            M_h[i] = M_h[i-1]
            M_g[i] = M_g[i-1]
            M_s[i] = M_s[i-1]
            SFR_open[i] = SFR_open[i-1]
        i=i+1
        
    return M_h[-1], M_g[-1], M_s[-1], SFR_open[-1]


#%%
### SFR VS MASS ################################################################################################

#definig the log of the specific star formation rate
log_sSFR = np.log10(SFR) - np.log10(mass)
log_mass = np.log10(mass)

#plotting the SFR from CIGALE
plt.figure()
p = plt.scatter(log_mass, np.log10(SFR), c=age, cmap='rainbow', s=2, alpha=1)
plt.xlim(7, 12)
plt.ylim(-4, 2)
plt.title("Star formation rate")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("SFR (1 / yrs)")
plt.colorbar(label="age")
p.set_alpha(alpha)
plt.show()

#plotting the sSFR from CIGALE
plt.figure()
p = plt.scatter(log_mass, log_sSFR, c=age, cmap='rainbow', s=2, alpha=1)
plt.axhline(T, color='crimson', label="Main sequence of Star Forming Galaxies")
plt.xlim(7, 12)
plt.ylim(-14, -8)
plt.title("Specific star formation rate")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("sSFR (1 / yrs)")
plt.colorbar(label="age")
p.set_alpha(alpha)
plt.legend()
plt.show()


#%%
### CLOSED BOX MODEL ###########################################################################################

log_sSFR_computed = np.log10( sSFR_model(age) )

#computing the sSFR with the closed box model
plt.figure()
p = plt.scatter(log_mass, log_sSFR_computed, c=age, cmap='rainbow', s=2, alpha=1)
plt.axhline(T, color='crimson', label="Main sequence of Star Forming Galaxies")
plt.xlim(7, 12)
plt.ylim(-14, -8)
plt.title("Closed box model")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("sSFR (1 / yrs)")
plt.colorbar(label="age")
p.set_alpha(alpha)
plt.legend()
plt.show()


#%%
### OPEN BOX MODEL #############################################################################################

#set global arrays
t = np.arange(dt,13.0,dt) #Gyr
z = t*0.
t_dyn = t*0.
print("\nInitialize arrays:")
for i in tqdm(range(len(t))):
    z[i] = z_at_value(Planck13.age, t[i] * u.Gyr)
    t_dyn[i] = 2.e7*(1+z[i])**(-0.75) #yr

#ranges in the functions parameters
t_form = np.concatenate((np.linspace(0.2, 0.3, 2000), np.linspace(0.3, 1, 500)))
M_h_form = np.logspace(7, 8.7, 6) #DM halo mass at t_form

Y = []
if M_h_form_isfixed:
    print("\nEvolving the galaxies:")
    for j in tqdm(range(len(t_form))):
        Y.append( evolve_galaxy(t_form[j], 13, M_h_form) )
else:
    print("\nEvolving the galaxies:")
    for k in range(len(M_h_form)):
        for j in tqdm(range(len(t_form))):
            Y.append( evolve_galaxy(t_form[j], 13, M_h_form[k]) )
Y = np.array(Y)

#definig the log of the specific star formation rate
log_mass_open = np.log10(Y[:,2])
log_sSFR_open = (np.log10(Y[:,3]) - log_mass_open) + 0.4

#computing the sSFR with the open box model
plt.figure()
plt.scatter(log_mass_open, log_sSFR_open, color='navy', s=2, alpha=1)
plt.axhline(T, color='crimson', label="Main sequence of Star Forming Galaxies")
plt.xlim(7, 12)
plt.ylim(-14, -8)
plt.title("Open box model")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("sSFR (1 / yrs)")
plt.legend()
plt.show()


#%%
### FIT MS #####################################################################################################

MS_data = np.array(np.where(log_sSFR>=T)).T.flatten()   #selecting indices of MS galaxies
MS_model = np.array(np.where((log_sSFR_computed>=T) & (log_sSFR_computed<-8.1))).T.flatten()
MS_open = np.array(np.where(log_sSFR_open>=T)).T.flatten()


### fit the main sequence - data ###
param_d, err_d = curve_fit(line, log_mass[MS_data], log_sSFR[MS_data])
print("\nsSFR from dataset")
print("log(sSFR) = " + str(np.round(param_d[0], 2)) + " log(mass) + " + str(np.round(param_d[1], 2)))
print("slope:", np.round(param_d[0], 4), "error:", np.round(np.sqrt(err_d[0,0]), 4))
print("intercept:", np.round(param_d[1], 3), "error:", np.round(np.sqrt(err_d[1,1]), 3))

x_axis = np.linspace(7, 12, 1000)

plt.figure()
plt.scatter(log_mass, log_sSFR, color='gainsboro', s=2, alpha=alpha)
plt.scatter(log_mass[MS_data], log_sSFR[MS_data], s=2, alpha=alpha, color='navy', label="main sequence")
plt.plot(x_axis, line(x_axis, param_d[0], param_d[1]), color='gold', lw=2, label=("Fit: log(sSFR) = " + str(np.round(param_d[0], 2)) + " log(mass) + " + str(np.round(param_d[1], 2))))
plt.xlim(7, 12)
plt.ylim(-12, -6)
plt.title("Main sequence - CIGALE data")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("sSFR (1 / yrs)")
plt.legend()
plt.show()

#%%
### fit the main sequence - open box ###
param_m, err_m = curve_fit(line, log_mass[MS_model], log_sSFR_computed[MS_model], p0=[0, -8])
print("\nsSFR computed with closed model")
print("log(sSFR) = " + str(np.round(param_m[0], 2)) + " log(mass) + " + str(np.round(param_m[1], 2)))
print("slope:", np.round(param_m[0], 3), "error:", np.round(np.sqrt(err_m[0,0]), 3))
print("intercept:", np.round(param_m[1], 3), "error:", np.round(np.sqrt(err_m[1,1]), 3))

x_axis = np.linspace(7, 12, 1000)

plt.figure()
plt.scatter(log_mass, log_sSFR_computed, color='gainsboro', s=2, alpha=alpha)
plt.scatter(log_mass[MS_model], log_sSFR_computed[MS_model], s=2, alpha=alpha, color='navy', label="main sequence")
plt.plot(x_axis, line(x_axis, param_m[0], param_m[1]), color='orange', lw=2, label=("Fit: log(sSFR) = " + str(np.round(param_m[0], 2)) + " log(mass) + " + str(np.round(param_m[1], 2))))
plt.xlim(7, 12)
plt.ylim(-12, -6)
plt.title("Main sequence - closed box")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("sSFR (1 / yrs)")
plt.legend()
plt.show()

#%%
### fit the main sequence - closed box ###
param_o, err_o = curve_fit(line, log_mass_open[MS_open], log_sSFR_open[MS_open], p0=[0, -8])
print("\nsSFR computed with open box model")
print("log(sSFR) = " + str(np.round(param_o[0], 2)) + " log(mass) + " + str(np.round(param_o[1], 2)))
print("slope:", np.round(param_o[0], 3), "error:", np.round(np.sqrt(err_o[0,0]), 3))
print("intercept:", np.round(param_o[1], 3), "error:", np.round(np.sqrt(err_o[1,1]), 3))

x_axis = np.linspace(7, 12, 1000)

plt.figure()
plt.scatter(log_mass_open, log_sSFR_open, color='gainsboro', s=2, alpha=alpha)
plt.scatter(log_mass_open[MS_open], log_sSFR_open[MS_open], s=2, alpha=alpha, color='navy', label="main sequence")
plt.plot(x_axis, line(x_axis, param_o[0], param_o[1]), color='coral', lw=2, label=("Fit: log(sSFR) = " + str(np.round(param_o[0], 2)) + " log(mass) + " + str(np.round(param_o[1], 2))))
plt.xlim(7, 12)
plt.ylim(-12, -6)
plt.title("Main sequence - open box")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("sSFR (1 / yrs)")
plt.legend()
plt.show()

#%%
### plot fits comparison ###
plt.figure()
plt.plot(x_axis, line(x_axis, param_m[0], param_m[1]), color='orange', lw=2, label=("Closed box model"))
plt.plot(x_axis, line(x_axis, param_o[0], param_o[1]), color='coral', lw=2, label=("Open box model"))
plt.plot(x_axis, line(x_axis, param_d[0], param_d[1]), color='gold', lw=2, label=("CIGALE data"))
plt.title("Main sequence - comparison")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("sSFR (1 / yrs)")
plt.legend()
plt.show()


#%%
### BIMODAL PLOT ###############################################################################################

### CIGALE dataset ###
#showing the grid
plt.figure()
plt.scatter(log_mass, log_sSFR, color='navy', s=0.7, alpha=alpha)
plt.xlim(7, 12)
plt.ylim(-14, -8)
plt.title("Binning - CIGALE data")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("sSFR (1 / yrs)")
plt.axhline(T, color='red')
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
plt.axvline(T, color='red')
plt.hist(log_sSFR[bin1], bins=150, density=True, color='royalblue', label="$10^{8} < m <10^{8.75}$", alpha=0.5)
plt.hist(log_sSFR[bin4], bins=150, density=True, color='crimson', label="$10^{10.25} < m <10^{11}$", alpha=0.5)
plt.title("Bimodal plot - CIGALE data")
plt.xlabel("$log_{10}(~sSFR (1 / yrs)~)$")
plt.ylabel("#galaxies")
plt.legend()
plt.xlim(-12, -8)
plt.show()

#%%
### closed box model ###
#showing the grid used for the histograms
plt.figure()
plt.scatter(log_mass, log_sSFR_computed, color='navy', s=0.7, alpha=alpha)   #sSFR
plt.xlim(7, 12)
plt.ylim(-14, -8)
plt.title("Binning - closed box")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("sSFR (1 / yrs)")
plt.axhline(T, color='red')
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

#bimodal plot
plt.figure()
plt.axvline(T, color='red')
plt.hist(log_sSFR_computed[bin1], bins=50, density=True, color='royalblue', label="$10^{8} < m <10^{8.75}$", alpha=0.5)
plt.hist(log_sSFR_computed[bin4], bins=50, density=True, color='crimson', label="$10^{10.25} < m <10^{11}$", alpha=0.5)
plt.title("Bimodal plot - closed box")
plt.xlabel("$log_{10}(~sSFR (1 / yrs)~)$")
plt.ylabel("#galaxies")
plt.legend()
plt.xlim(-12, -8)
plt.show()

#%%
### open box model ###
#showing the grid used for the histograms
plt.figure()
plt.scatter(log_mass_open, log_sSFR_open, color='navy', s=0.7, alpha=alpha)   #sSFR
plt.xlim(7, 12)
plt.ylim(-14, -8)
plt.title("Binning - open box")
plt.xlabel("mass ($M_{\odot}$)")
plt.ylabel("sSFR (1 / yrs)")
plt.axhline(T, color='red')
plt.axvline(8, color='black', ls='dotted')
plt.axvline(8.75, color='black', ls='dotted')
plt.axvline(9.5, color='black', ls='dotted')
plt.axvline(10.25, color='black', ls='dotted')
plt.axvline(11, color='black', ls='dotted')
plt.show()

#indexes associated with the mass intervals
bin1 = (log_mass_open>=8) & (log_mass_open<8.75)
bin2 = (log_mass_open>=8.75) & (log_mass_open<9.5)
bin3 = (log_mass_open>=9.5) & (log_mass_open<10.25)
bin4 = (log_mass_open>=10.25) & (log_mass_open<11)

#bimodal plot
plt.figure()
plt.axvline(T, color='red')
plt.hist(log_sSFR_open[bin1], bins=15, density=True, color='royalblue', label="$10^{8} < m <10^{8.75}$", alpha=0.5)
plt.hist(log_sSFR_open[bin4], bins=15, density=True, color='crimson', label="$10^{10.25} < m <10^{11}$", alpha=0.5)
plt.title("Bimodal plot - open box")
plt.xlabel("$log_{10}(~sSFR (1 / yrs)~)$")
plt.ylabel("#galaxies")
plt.legend()
plt.xlim(-12, -8)
plt.show()



