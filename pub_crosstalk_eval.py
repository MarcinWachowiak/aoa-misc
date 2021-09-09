#%%
#import modules
from math import inf
import struct, array, os, sys, ctypes
import numpy as np
from numpy.core.numeric import Inf
import numpy.polynomial.polynomial as poly
import matplotlib as mpl
import matplotlib.pyplot as plt
import array
import datetime
from matplotlib import rc
import time
import subprocess
import sys

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

att_vec = [0, 10, 20, 31, 40, 50, 60]

dir = "/home/marcin/Desktop/crosstalk_eval/tx0/"
pha_coeff_uut0 = list()
pha_std_uut0 = list()
mag_uut0 = list()
for attenuation in att_vec:
        pha_sampl_arr, pha_ss_arr = read_data_file(dir + "phase_mav_100k_bp_filtered_rx0_40_rx1_40_tx_45.0_freq_2440000000_att_%d_round_0.bin" %(int(attenuation)),sampl_rate,decimation)

        pha_coeff_uut0.append(np.average(fw_replace_nan(np.array(pha_sampl_arr))))
        pha_std_uut0.append(np.std(fw_replace_nan(np.array(pha_sampl_arr))))
for attenuation in att_vec:
        mag_sampl_arr, pha_ss_arr = read_data_file(dir + "mag_ch0_mav_100k_bp_filtered_rx0_40_rx1_40_tx_45.0_freq_2440000000_att_%d_round_0.bin" %(int(attenuation)),sampl_rate,decimation)
        mag_uut0.append(np.average(fw_replace_nan(np.array(mag_sampl_arr))))

dir = "/home/marcin/Desktop/crosstalk_eval/tx1/"
pha_coeff_uut1 = list()
pha_std_uut1 = list()
mag_uut1 = list()
for attenuation in att_vec:
        pha_sampl_arr, pha_ss_arr = read_data_file(dir + "phase_mav_100k_bp_filtered_rx0_40_rx1_40_tx_45.0_freq_2440000000_att_%d_round_0.bin" %(int(attenuation)),sampl_rate,decimation)

        pha_coeff_uut1.append(np.average(fw_replace_nan(np.array(pha_sampl_arr))))
        pha_std_uut1.append(np.std(fw_replace_nan(np.array(pha_sampl_arr))))
for attenuation in att_vec:
        mag_sampl_arr, pha_ss_arr = read_data_file(dir + "mag_ch0_mav_100k_bp_filtered_rx0_40_rx1_40_tx_45.0_freq_2440000000_att_%d_round_0.bin" %(int(attenuation)),sampl_rate,decimation)
        mag_uut1.append(np.average(fw_replace_nan(np.array(mag_sampl_arr))))

att_vec = np.array([0, 10, 20, 30, 40, 50, 60])
dir = "/home/marcin/Desktop/crosstalk_eval/indep_tx/"
indep_pha = list()
for attenuation in att_vec:
        pha_sampl_arr, pha_ss_arr = read_data_file(dir + "phase_mav_100k_bp_filtered_rx0_40_rx1_40_tx_45.0_freq_2440000000_att_%d_round_0.bin" %(int(attenuation)),sampl_rate,decimation)
        indep_pha.append(np.average(fw_replace_nan(np.array(pha_sampl_arr))))

print("Finished reading!")
#%%
att_vec = np.array([0, 10, 20, 30, 40, 50, 60])

ref = np.average(indep_pha[:4])

mag_samp_array0, pha_ss_arr = read_data_file("/home/marcin/Desktop/crosstalk_eval/tx0/mag_ch0_mav_100k_bp_filtered_rx0_40_rx1_40_tx_45.0_freq_2440000000_att_99_round_0.bin",sampl_rate,decimation)
cross_mag0 = np.average(fw_replace_nan(np.array(mag_samp_array0)))
print("cross_mag0")
print(cross_mag0)
print("ref_mag_uut0")
print(mag_uut0[0])
mag_uut0 = mag_uut0[0]*np.power(10.0, -att_vec/20)
print(mag_uut0)
mag_uut0[mag_uut0<cross_mag0] = 0
print("mag_uut0")
print(mag_uut0)

print("\n")

mag_samp_array1, pha_ss_arr = read_data_file("/home/marcin/Desktop/crosstalk_eval/tx1/mag_ch1_mav_100k_bp_filtered_rx0_40_rx1_40_tx_45.0_freq_2440000000_att_99_round_0.bin",sampl_rate,decimation)
cross_mag1 = np.average(fw_replace_nan(np.array(mag_samp_array1)))
print("cross_mag1")
print(cross_mag1)
print("ref_mag_uut1")
print(mag_uut1[0])
mag_uut1 = mag_uut1[0]*np.power(10.0, -att_vec/20)
print(mag_uut1)
mag_uut1[mag_uut1<cross_mag1] = 0
print("mag_uut1")
print(mag_uut1)

ref = np.average(fw_replace_nan(np.array(pha_sampl_arr)))
print("\nFinished reading data!\n")

#%%
#plot data and check
att_vec = [0, 10, 20, 30, 40, 50, 60]
dot_size = 7
fig, ax = plt.subplots()
ax.grid()
ref_deg = np.degrees(ref)

ref_plot = ax.hlines(ref_deg,0,60, 'k', label = "Ref.")
scatt3 = ax.scatter(att_vec, np.degrees(indep_pha), zorder = 5, label = "Sep. TX",s =dot_size, color = "#4daf4a" )

scatt1 = ax.scatter(att_vec, np.degrees(pha_coeff_uut0), zorder = 5, label = "TX0", s=dot_size)
scatt2 = ax.scatter(att_vec, np.degrees(pha_coeff_uut1), zorder = 6, label = "TX1",s =dot_size)


print("Calibration error:")
print(np.degrees(pha_coeff_uut0-ref))
print(np.degrees(pha_coeff_uut1-ref))

leg1 = ax.legend(loc='lower left',ncol=2, columnspacing=0.5)
ba0 =np.divide(cross_mag0,mag_uut0)
print(ba0)
err0 = np.where(ba0 == np.inf, 2, np.arctan(ba0))
err0_deg = np.degrees(err0)
print(err0_deg)

fill1 = ax.fill_between(att_vec, (ref_deg-err0_deg),(ref_deg+err0_deg), alpha=0.5, color = '#377eb8', zorder = 4, label = "BD0")

ba1 =np.divide(cross_mag1,mag_uut1)
print(ba1)
err1 = np.where(ba1 == np.inf, 2, np.arctan(ba1))
err1_deg = np.degrees(err1)
print(err1_deg)

fill2 = ax.fill_between(att_vec, (ref_deg-err1_deg), (ref_deg+err1_deg), alpha=0.5, color = '#ff7f00', zorder = 3, label = "BD1")
ax.set_xlabel("Attenuation [dB]")
ax.set_ylabel('Measured phase difference [$\degree$]')

leg2 = ax.legend(["__nolegend__","__nolegend__","__nolegend__","__nolegend__","BD0", "BD1"],loc='upper left')
ax.add_artist(leg1)

ax.set_xlim([0,60])
ax.set_yticks([-90,-60,-30,0,30,60,90])
ax.set_ylim([-90,90])

plt.tight_layout()
plt.savefig("crosstalk_eval.pdf")
plt.show()

# %%
