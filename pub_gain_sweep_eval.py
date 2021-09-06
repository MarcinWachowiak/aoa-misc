#%%
#import modules
import struct, array, os, sys, ctypes
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib as mpl
import matplotlib.pyplot as plt
import array
import datetime
from matplotlib import rc
import time
import subprocess
import sys
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
fig_height =fig_width*golden_mean*1.4       # height in inches
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
# %%
#read data
sampl_rate = int(1e6)
decimation = int(1e3)

dir = "/home/marcin/Desktop/gain_sweep_eval/"

tx_gain = 40 #does not cause saturation at 76, increased variance below 20dB...
center_freq = 2.44e+9

rx0_gain = 10
gain_start = 0
gain_stop = 76.0
gain_step = 1.0

rx0_gain_vec = [20, 37]
rx1_gain_vec = np.arange(gain_start, gain_stop+gain_step, gain_step)

pha_coeff_uut = list()
pha_std_uut = list()

meas_range = list(range(1))

for round_idx in meas_range:
    for rx0_gain in rx0_gain_vec:
        if(rx0_gain == 8.0 or rx0_gain == 20.0 or rx0_gain == 55.0):
            tx_gain = 55.0
        else:
            tx_gain = 40.0

        tmp_pha_coeff = list()
        tmp_pha_std = list()
        for rx1_gain in rx1_gain_vec:
            pha_sampl_arr, pha_ss_arr = read_data_file(dir + "phase_mav_100k_bp_filtered_rx0_%2.0f_rx1_%2.0f_tx_%2.1f_freq_%d_round_%d.bin" % (rx0_gain, rx1_gain, tx_gain, center_freq, round_idx),sampl_rate,decimation)
            tmp_pha_coeff.append(np.average(fw_replace_nan(np.array(pha_sampl_arr))))
            tmp_pha_std.append(np.std(fw_replace_nan(np.array(pha_sampl_arr))))

        pha_coeff_uut.append(tmp_pha_coeff)
        pha_std_uut.append(tmp_pha_std)

print("\nFinished reading data!\n")

#%%
#plot data and check 
#phase
fig, ax1 = plt.subplots()
plt.grid()
plt.tight_layout()

plot1 =  ax1.plot(rx1_gain_vec, np.degrees(pha_coeff_uut[0]), label = "20")
plot2 =  ax1.plot(rx1_gain_vec, np.degrees(pha_coeff_uut[1]), label = "37")

ax1.set_xlabel("RX1 Gain [dB]")
ax1.set_ylabel('Phase difference [$\degree$]')
ax1.set_ylim([-190,190])
ax1.set_xlim([0,76])
ax1.legend(rx0_gain_vec,title="RX0 [dB]", loc = 2)
ax1.yaxis.set_major_locator(MaxNLocator(7))
ax1.xaxis.set_major_locator(MaxNLocator(6)) 

ilna_gain = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 14, 14, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21]
mixer_gain = [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
ax2 = ax1.twinx()
plot3 = ax2.plot(rx1_gain_vec, ilna_gain, '--', label = "iLNA", color = "k")
plot4 = ax2.plot(rx1_gain_vec, mixer_gain, '--', label = "Mixer", color = "g")
ax2.set_ylabel("IC gain [dB]")

plots = plot1+plot2+plot3+plot4
labs = [l.get_label() for l in plots]
leg = ax1.legend(plots, labs,loc=(0.12,-0.6) ,title="RX0 [dB]", ncol= 2)
leg._legend_box.align = "left"

ax2.yaxis.set_major_locator(MaxNLocator(6))

plt.savefig("gain_sweep.pdf",dpi=600,bbox_inches = 'tight')
plt.show()

# #phase meas std
# fig, ax = plt.subplots()
# plt.grid()
# for pha_vec_std in pha_std_uut:
#     ax.plot(rx1_gain_vec, np.degrees(pha_vec_std))
# ax.legend()
# ax.set_xlabel("RX1 Gain [dB]")
# ax.set_ylabel('Phase difference between RX channels STD [$\degree$]')

# plt.savefig("gain_eval_test_std.pdf")
# plt.show()

# %%
