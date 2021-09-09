#Scripts evaluating gathered AoA data for various scenarios
#including: high acitivity in the wireless channel, effects
#of bandpass filtration and movement during measurment
#%%
#import modules
import struct, os, ctypes
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
from matplotlib import rc
import numpy.polynomial.polynomial as poly
from matplotlib.ticker import MaxNLocator

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=CB_color_cycle)

rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{gensymb}')
#!python numbers=disable
#fig_width_pt = 469.75502  # Get this from LaTeX using \showthe\columnwidth result: 
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0      # Aesthetic ratio
fig_width = 2.3622  # width in inches
fig_height =fig_width*golden_mean*2.0       # height in inches
fig_size = [fig_width,fig_height]

params = {'backend': 'ps',
          'axes.labelsize': 8,
          'font.size': 8,
          'legend.fontsize': 8,
          'xtick.labelsize': 7,
          'ytick.labelsize': 7,
          'text.usetex': True,
          'figure.figsize': fig_size}

mpl.rcParams.update(params)
mpl.rcParams["font.family"] = ["Latin Modern Roman"]
mpl.rcParams['lines.linewidth'] = 1

#!python numbers=disable
#pylab.axes([0.125,0.2,0.95-0.125,0.95-0.2])
plt.rcParams['path.simplify'] = True
print("\nFinished importing modules!\n")

#!python numbers=disable
#pylab.axes([0.125,0.2,0.95-0.125,0.95-0.2])
print("\nFinished importing modules!\n")

#%%
#declare functions
def read_data_file(filename, sampl_rate, decim):
    #fread binary file containing only floats (no other info)
    f_size = os.path.getsize(filename)
    print("Reading file: %s" %(filename))
    val_size = ctypes.sizeof(ctypes.c_float)
    n_vals = int(f_size/val_size)

    print("Number of samples: %d"%(n_vals))
    samples = open(filename, "rb")
    sampl_arr = struct.unpack('f'*n_vals, samples.read(4*n_vals))# %%
    
    resultant_sample_rate = sampl_rate/decim
    sampl_spacing = 1/(resultant_sample_rate)
    print("Resultant sample rate: %d"%(resultant_sample_rate))

    rec_time_length = n_vals*sampl_spacing
    print("Record time length: %s \n"%(datetime.timedelta(seconds=round(rec_time_length))))
    return sampl_arr

def trim_data(sampl_arr,num_samples):
    return sampl_arr[num_samples : (len(sampl_arr)-num_samples)]

def fw_replace_nan(sampl_arr):
    #replace inf if any
    mask = np.logical_or(np.isinf(sampl_arr),np.isnan(sampl_arr))
    idx = np.where(~mask,np.arange(mask.size),0)
    np.maximum.accumulate(idx, out=idx)
    return sampl_arr[idx]
def arr_get_hh(seconds_arr):
    ret_str_arr = list()
    for idx in range(len(seconds_arr)):
        ret_str_arr.append(str(datetime.timedelta(seconds=seconds_arr[idx]))[:-6])
    return ret_str_arr

print("\nFinished declaring functions!\n")

# %%
sampl_rate = int(1e6)
decim = int(1000)
resultant_sample_rate = sampl_rate/decim
sampl_spacing = 1/(resultant_sample_rate)
dir = "/home/marcin/Desktop/ref_phase_drift/"
ref_phase = read_data_file(dir+"phase_mav_100k_bp_filtered_f_2440000000_round_0.bin",sampl_rate,decim)
#decimate obtained sample array
decim_rate = 1000
ref_phase = np.degrees(ref_phase[::decim_rate])
sampl_spacing = sampl_spacing*decim


#%%
#Histogram comparsion group plot
fig, axs = plt.subplots(2,1)
props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
n_bins = 100
#axs[0].set_title("a)")
axs[0].set_xlabel("Time [h]")
axs[0].set_ylabel('Phase difference [$\degree$]')

timebase = np.linspace(0,sampl_spacing*len(ref_phase),len(ref_phase))
axs[0].plot(timebase, ref_phase, linewidth=0.1)

decimation_rate = 10
timebase_decim = timebase[0:len(timebase):decimation_rate]
sampl_arr_decim = ref_phase[0:len(ref_phase):decimation_rate]

poly_deg = 5
poly_coeff = poly.polyfit(timebase_decim, sampl_arr_decim,poly_deg)
ffit = poly.polyval(timebase_decim, poly_coeff)
axs[0].plot(timebase_decim,ffit)
axs[0].grid(zorder=0)
axs[0].legend(['__nolegend__','Polynomial fit',])

x_ticks_loc = (np.arange(min(timebase),max(timebase),3600))
x_ticks_val = arr_get_hh(x_ticks_loc)

axs[0].set_xticks(x_ticks_loc)
axs[0].set_xticklabels(x_ticks_val)
axs[0].set_xlim([min(x_ticks_loc),4*3600])
axs[0].set_ylim([8.7,9.2])
#axs[0].text(0.5,-0.2, "a)", transform=axs[0].transAxes, horizontalalignment = 'center', fontsize = 10)
axs[0].yaxis.set_major_locator(MaxNLocator(6)) 

#axs[0].set_title("b)")
axs[1].set_xlabel('Phase difference [$\degree$]')
axs[1].set_ylabel('Probability')
axs[1].grid(zorder=0)
#skip initial 3h from avg calculation
n_last_samples = int(40496 * 8/11)
no_drift = ref_phase[-n_last_samples:]
axs[1].hist(no_drift, bins=n_bins, weights=np.ones_like(no_drift)/(len(no_drift)), zorder=3, label="__nolegend__")
axs[1].vlines(np.mean(no_drift), 0, 0.04, color='#f781bf', zorder=4, label=r"$\overline{\Delta\varphi}=%1.2f,\ \sigma=%1.2f\degree$" %(np.mean(no_drift), np.std(no_drift)))
#axs[1].text(0.5, -0.2, "b)", transform=axs[1].transAxes, horizontalalignment = 'center', fontsize = 10)
#axs[1].text(0.79, 0.92,(r"$\overline{\Delta\varphi} = %1.2f \pm %1.4f \degree$" % (np.mean(ref_phase), np.mean(ref_phase)*1.96*np.std(ref_phase)/np.sqrt(len(ref_phase)))), transform=axs[1].transAxes, fontsize=8, verticalalignment='bottom', bbox=props)
axs[1].set_ylim([0,0.05])
axs[1].set_xlim([8.7, 9.1])
axs[1].yaxis.set_major_locator(MaxNLocator(5)) 
axs[1].xaxis.set_major_locator(MaxNLocator(5)) 
axs[1].legend()


plt.tight_layout()
plt.savefig("ref_phase_meas.pdf",dpi=600,bbox_inches = 'tight')
plt.show()


# %%
