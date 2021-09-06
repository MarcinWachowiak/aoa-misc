#Script reading, analysing and plotting correcitng coefficients
#%%
#import modules
import struct, os, ctypes
from matplotlib import lines
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
fig_height =fig_width*golden_mean*1.5       # height in inches
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

n_meas = list(range(1))
meas_dirs = ["/home/marcin/Desktop/usrp_dev_comparison/freq_sweep_eval_usrp5/", "/home/marcin/Desktop/usrp_dev_comparison/freq_sweep_eval_usrp3/", "/home/marcin/Desktop/usrp_dev_comparison/freq_sweep_eval_usrp4/", "/home/marcin/Desktop/usrp_dev_comparison/freq_sweep_eval_usrp6/", "/home/marcin/Desktop/usrp_dev_comparison/freq_sweep_eval_usrp7/", "/home/marcin/Desktop/usrp_dev_comparison/freq_sweep_eval_usrp8/"]

freq_start = 350e6
freq_stop = 6e9
freq_step = 50e6
meas_time_s = 10

freq_vec = np.round(np.arange(freq_start, freq_stop, freq_step))
n_files_each = len(freq_vec)
pha_coeff = list()
pha_std = list()

for single_dir in meas_dirs:
    for round_idx in n_meas:
        tmp_pha_coeff = list()
        tmp_pha_std = list()
        for freq in freq_vec:
            pha_sampl_arr, pha_ss_arr = read_data_file(single_dir + "phase_mav_100k_bp_filtered_f_%d_round_%d.bin" % (freq, round_idx),sampl_rate,decimation)
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

freq_start = 350e6
freq_stop = 6e9
freq_step = 50e6
meas_time_s = 10

new_freq_vec = np.round(np.arange(freq_start, freq_stop, freq_step))

fig, ax = plt.subplots()

for dev_pha_sweep in pha_coeff:
    ax.plot(new_freq_vec/1e9, dev_pha_sweep[0:len(new_freq_vec)])

ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel('Phase difference [$\degree$]')
ax.set_xlim([freq_start/1e9, freq_stop/1e9])
ax.set_ylim([-16,28])
ax.yaxis.set_major_locator(MaxNLocator(8)) 
ax.xaxis.set_major_locator(MaxNLocator(6)) 

ax.vlines(2.2, -18, 30, color = "gray", linestyles='dashed', zorder = 3)
ax.vlines(4.0, -18, 30, color = "gray", linestyles='dashed', zorder = 3)
ax.grid()

ax.annotate(
    '', xy=(2.2, -12), xycoords='data',
    xytext=(0.35, -12), textcoords='data',
    arrowprops={'arrowstyle': '<->', 'shrinkA':0, 'shrinkB':0},bbox=dict(pad=0, facecolor="none", edgecolor="none"))
ax.annotate(
    'Chain A', xy=(1.1-0.6, -12), xycoords='data',
    xytext=(0, 5), textcoords='offset points')
ax.annotate(
    '', xy=(4.0, -12), xycoords='data',
    xytext=(2.2, -12), textcoords='data',
    arrowprops={'arrowstyle': '<->', 'shrinkA':0, 'shrinkB':0},bbox=dict(pad=0, facecolor="none", edgecolor="none"))
ax.annotate(
    'Chain B', xy=(2.9-0.55, -12), xycoords='data',
    xytext=(0, 5), textcoords='offset points')
ax.annotate(
    '', xy=(6.0, -12), xycoords='data',
    xytext=(4.0, -12), textcoords='data',
    arrowprops={'arrowstyle': '<->', 'shrinkA':0, 'shrinkB':0 }, bbox=dict(pad=0, facecolor="none", edgecolor="none"))
ax.annotate(
    'Chain C', xy=(4.8-0.55, -12), xycoords='data',
    xytext=(0, 5), textcoords='offset points')
 
plt.tight_layout()
plt.savefig("multiple_usrp_phase_freq_sweep.pdf", dpi=600, bbox_inches = 'tight')
plt.show()


# fig, ax = plt.subplots()

# for dev_pha_std in pha_std:
#     ax.plot(new_freq_vec/1e9, dev_pha_std[0:len(new_freq_vec)])

# ax.set_xlabel('Frequency [GHz]')
# ax.set_ylabel('Phase difference between RX channels STD [$\degree$]')
# ax.set_xlim([freq_start/1e9, freq_stop/1e9])
# ax.yaxis.set_major_locator(MaxNLocator(8)) 
# #ax.xaxis.set_major_locator(MaxNLocator(6)) 
# ax.grid()

# plt.tight_layout()
# plt.savefig("multiple_usrp_phase_std_freq_sweep.pdf", dpi=600, bbox_inches = 'tight')
# plt.show()

# %%
