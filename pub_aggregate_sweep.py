#Script reading, analysing and plotting correcitng coefficients
#%%
#import modules
import struct, os, ctypes
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
from matplotlib import rc
from matplotlib.ticker import MaxNLocator
 
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=CB_color_cycle)
mpl.rcParams['lines.linewidth'] = 1

rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{gensymb}')
#!python numbers=disable
#fig_width_pt = 469.75502  # Get this from LaTeX using \showthe\columnwidth result: 
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0      # Aesthetic ratio
fig_width = 2.3622  # width in inches
fig_height =fig_width*golden_mean*1.2       # height in inches
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

#!python numbers=disable
#pylab.axes([0.125,0.2,0.95-0.125,0.95-0.2])
plt.rcParams['path.simplify'] = True
print("\nFinished importing modules!\n")

#%%
#declare functions
def read_data_file(filename, sampl_rate, decimation):
    #fread binary file containing only floats (no other info)
    f_size = os.path.getsize(filename)
    print("Reading file: %s" %(filename))
    val_size = ctypes.sizeof(ctypes.c_float)
    n_vals = int(f_size/val_size)

    print("Number of samples: %d"%(n_vals))
    samples = open(filename, "rb")
    sampl_arr = struct.unpack('f'*n_vals, samples.read(4*n_vals))# %%
    
    resultant_sample_rate = sampl_rate/decimation
    sampl_spacing = 1/(resultant_sample_rate)
    print("Resultant sample rate: %d"%(resultant_sample_rate))

    rec_time_length = n_vals*sampl_spacing
    print("Record time length: %s \n"%(datetime.timedelta(seconds=round(rec_time_length))))
    return sampl_arr, sampl_spacing

def fw_replace_nan(sampl_arr):
    #replace inf if any
    mask = np.logical_or(np.isinf(sampl_arr),np.isnan(sampl_arr))
    idx = np.where(~mask,np.arange(mask.size),0)
    np.maximum.accumulate(idx, out=idx)
    return sampl_arr[idx]

def trim_data(sampl_arr,num_samples):
    return sampl_arr[num_samples : (len(sampl_arr)-num_samples)]

print("\nFinished declaring functions!\n")

#%%
sampl_rate = int(1e6)
decimation = int(1e3)
n_skip_samples = int(2000)

n_meas = list(range(50))
dir = "/home/marcin/Desktop/freq_sweep_eval/"

freq_start = 350e6
freq_stop = 6e9
freq_step = 50e6
meas_time_s = 10

freq_vec = np.round(np.arange(freq_start, freq_stop, freq_step))

n_files_each = len(freq_vec)
pha_coeff = list()
pha_std = list()


for round_idx in n_meas:
    tmp_pha_coeff = list()
    tmp_pha_std = list()
    for freq in freq_vec:
        pha_sampl_arr, pha_ss_arr = read_data_file(dir + "phase_mav_100k_bp_filtered_f_%d_round_%d.bin" % (freq, round_idx),sampl_rate,decimation)
        tmp_pha_coeff.append(np.average(fw_replace_nan(np.array(pha_sampl_arr))))
        tmp_pha_std.append(np.std(fw_replace_nan(np.array(pha_sampl_arr))))

    pha_coeff.append(np.degrees(tmp_pha_coeff))
    pha_std.append(np.degrees(tmp_pha_std))

print("\nFinished reading!\n")

#%%
# fig, ax = plt.subplots()
# plt.grid()
pha_coeff_mat = np.vstack(pha_coeff)
pha_coeff_avg = np.mean(pha_coeff, axis=0)
pha_std_col = np.std(pha_coeff, axis=0)
z_score_val = 1.96
conf_interval = z_score_val * pha_std_col / np.sqrt(len(pha_coeff))

fig, ax = plt.subplots()
ax.plot(freq_vec/1e9, pha_coeff_avg)
#ax.errorbar(freq_vec/1e9, pha_coeff_avg, conf_interval, alpha=0.5,ecolor='black', capsize=8)
ax.fill_between(freq_vec/1e9, (pha_coeff_avg-conf_interval), (pha_coeff_avg+conf_interval), alpha=0.5, color = '#ff7f00')
ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel('Phase difference [$\degree$]')
ax.set_xlim([350e6/1e9, 6e9/1e9])
ax.set_ylim([0,28])
ax.yaxis.set_major_locator(MaxNLocator(6)) 
ax.xaxis.set_major_locator(MaxNLocator(6)) 
ax.grid()

plt.tight_layout()
plt.savefig("agg_phase_cal_meas.pdf", dpi=600, bbox_inches = 'tight')
plt.show()

# %%
#HOW MUCH A SINGLE MM contributes to the total phase shift at 2440: 63mm/64mm
c = 299792458
f = 2440000000
s = 0.001 # 1mm
var_phi  = s/(c/f)*2*np.pi
print("Radian shift: %f" %var_phi)
print("Degree shift: %f" %(var_phi/(2*np.pi)*360))

print("Half wavelength array spacin at 2.44GHz: %f" %(c/f/2))# %%

