#Script reading, analysing and plotting correcitng coefficients
#%%
#import modules
import struct, os, ctypes
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib as mpl
import matplotlib.pyplot as plt
import array
import datetime
from matplotlib import rc

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=CB_color_cycle)

rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{gensymb}')
#!python numbers=disable
fig_width_pt = 469.75502  # Get this from LaTeX using \showthe\columnwidth result: 
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0      # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height =fig_width*golden_mean       # height in inches
fig_size = [fig_width,fig_height]

params = {'backend': 'ps',
          'axes.labelsize': 10,
          'font.size': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
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
def read_data_file(filename, sampl_rate, decim):
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

def trim_data(sampl_arr,num_samples):
    return sampl_arr[num_samples : (len(sampl_arr)-num_samples)]

def fw_replace_nan(sampl_arr):
    #replace inf if any
    mask = np.logical_or(np.isinf(sampl_arr),np.isnan(sampl_arr))
    idx = np.where(~mask,np.arange(mask.size),0)
    np.maximum.accumulate(idx, out=idx)
    return sampl_arr[idx]

def rmse(predictions, targets):
    differences = predictions - targets                       #the DIFFERENCEs
    differences_squared = differences ** 2                    #the SQUAREs of ^
    mean_of_differences_squared = differences_squared.mean()  #the MEAN of ^
    rmse_val = np.sqrt(mean_of_differences_squared)           #ROOT of ^
    return rmse_val                                           #get the ^

print("\nFinished declaring functions!\n")

#%%
sampl_rate = int(1e6)
decimation = int(1000)
n_skip_samples = int(0)

ref_1 = "/home/marcin/Desktop/wi_coeff_eval/ref_cal/phase_mav_100k_bp_filtered_f_2440000000_chan_0_round_0.bin"
ref_2 = "/home/marcin/Desktop/wi_coeff_eval/ref_cal/phase_mav_100k_bp_filtered_f_2440000000_chan_1_round_0.bin"
ref_3 = "/home/marcin/Desktop/wi_coeff_eval/los_cal_ref/phase_mav_100k_bp_filtered_f_2440000000_chan_0_round_0.bin"
ref_4 = "/home/marcin/Desktop/wi_coeff_eval/los_cal_ref/phase_mav_100k_bp_filtered_f_2440000000_chan_1_round_0.bin"

ph_ref_1, pha_ss_arr = read_data_file(ref_1,sampl_rate,decimation)
ph_ref_2, pha_ss_arr = read_data_file(ref_2,sampl_rate,decimation)
ph_ref_3, pha_ss_arr = read_data_file(ref_3,sampl_rate,decimation)
ph_ref_4, pha_ss_arr = read_data_file(ref_4,sampl_rate,decimation)

avg_ref_1 = np.degrees(np.mean(ph_ref_1))
avg_ref_2 = np.degrees(np.mean((ph_ref_2)))
avg_ref_3 = np.degrees(np.mean((ph_ref_3)))
avg_ref_4 = np.degrees(np.mean((ph_ref_4)))

avg_tot = (avg_ref_1+avg_ref_2+avg_ref_3+avg_ref_4)/4

print(avg_ref_1)
print(avg_ref_2)
print(avg_ref_3)
print(avg_ref_4)
print(avg_tot)

#%%
n_rounds = list(range(50))
close_dir = "/home/marcin/Desktop/wi_coeff_eval/close_cal_50/"

close_ph = list()
close_ph_std = list()

far_ph = list()
far_ph_std = list()

los_ph = list()
los_ph_std = list()

tmp_pha_ch0 = list()
tmp_pha_ch1 = list()
tmp_pha_avg = list()

tmp_pha_ch0_std = list()
tmp_pha_ch1_std = list()
tmp_pha_avg_std = list()

for round_idx in n_rounds:
    pha_ch0, pha_ss_arr = read_data_file(close_dir + "phase_mav_100k_bp_filtered_f_2440000000_chan_0_round_%d.bin" % (round_idx),sampl_rate,decimation)
    pha_ch0 = pha_ch0[-10000:]
    pha_ch1, pha_ss_arr = read_data_file(close_dir + "phase_mav_100k_bp_filtered_f_2440000000_chan_1_round_%d.bin" % (round_idx),sampl_rate,decimation)
    pha_ch1 = pha_ch1[-10000:]

    pha_ch_avg = np.mean([pha_ch0, pha_ch1], axis = 0)

    tmp_pha_ch0.append(np.mean(pha_ch0))
    tmp_pha_ch1.append(np.mean(pha_ch1))
    tmp_pha_avg.append(np.mean(pha_ch_avg))

    tmp_pha_ch0_std.append(np.std(pha_ch0))
    tmp_pha_ch1_std.append(np.std(pha_ch1))
    tmp_pha_avg_std.append(np.std(pha_ch_avg))

close_ph.append(np.degrees(tmp_pha_ch0))
close_ph.append(np.degrees(tmp_pha_ch1))
close_ph.append(np.degrees(tmp_pha_avg))

close_ph_std.append(np.degrees(tmp_pha_ch0_std))
close_ph_std.append(np.degrees(tmp_pha_ch1_std))
close_ph_std.append(np.degrees(tmp_pha_avg_std))

far_dir = "/home/marcin/Desktop/wi_coeff_eval/far_cal_50/"
tmp_pha_ch0 = list()
tmp_pha_ch1 = list()
tmp_pha_avg = list()

tmp_pha_ch0_std = list()
tmp_pha_ch1_std = list()
tmp_pha_avg_std = list()

for round_idx in n_rounds:
    pha_ch0, pha_ss_arr = read_data_file(far_dir + "phase_mav_100k_bp_filtered_f_2440000000_chan_0_round_%d.bin" % (round_idx),sampl_rate,decimation)
    pha_ch0 = pha_ch0[-10000:]
    pha_ch1, pha_ss_arr = read_data_file(far_dir + "phase_mav_100k_bp_filtered_f_2440000000_chan_1_round_%d.bin" % (round_idx),sampl_rate,decimation)
    pha_ch1 = pha_ch1[-10000:]

    #invert phase coeff from ch 1
    pha_ch_avg = np.mean([pha_ch0, pha_ch1], axis = 0)

    tmp_pha_ch0.append(np.mean(pha_ch0))
    tmp_pha_ch1.append(np.mean(pha_ch1))
    tmp_pha_avg.append(np.mean(pha_ch_avg))

    tmp_pha_ch0_std.append(np.std(pha_ch0))
    tmp_pha_ch1_std.append(np.std(pha_ch1))
    tmp_pha_avg_std.append(np.std(pha_ch_avg))


far_ph.append(np.degrees(tmp_pha_ch0))
far_ph.append(np.degrees(tmp_pha_ch1))
far_ph.append(np.degrees(tmp_pha_avg))

far_ph_std.append(np.degrees(tmp_pha_ch0_std))
far_ph_std.append(np.degrees(tmp_pha_ch1_std))
far_ph_std.append(np.degrees(tmp_pha_avg_std))


los_dir = "/home/marcin/Desktop/wi_coeff_eval/los_cal/"
tmp_pha_ch0 = list()
tmp_pha_ch1 = list()
tmp_pha_avg = list()

tmp_pha_ch0_std = list()
tmp_pha_ch1_std = list()
tmp_pha_avg_std = list()

for round_idx in n_rounds:
    pha_ch0, pha_ss_arr = read_data_file(los_dir + "phase_mav_100k_bp_filtered_f_2440000000_chan_0_round_%d.bin" % (round_idx),sampl_rate,decimation)
    pha_ch0 = pha_ch0[-10000:]
    pha_ch1, pha_ss_arr = read_data_file(los_dir + "phase_mav_100k_bp_filtered_f_2440000000_chan_1_round_%d.bin" % (round_idx),sampl_rate,decimation)
    pha_ch1 = pha_ch1[-10000:]

    #invert phase coeff from ch 1
    pha_ch_avg = np.mean([pha_ch0, pha_ch1], axis = 0)

    tmp_pha_ch0.append(np.mean(pha_ch0))
    tmp_pha_ch1.append(np.mean(pha_ch1))
    tmp_pha_avg.append(np.mean(pha_ch_avg))

    tmp_pha_ch0_std.append(np.std(pha_ch0))
    tmp_pha_ch1_std.append(np.std(pha_ch1))
    tmp_pha_avg_std.append(np.std(pha_ch_avg))


los_ph.append(np.degrees(tmp_pha_ch0))
los_ph.append(np.degrees(tmp_pha_ch1))
los_ph.append(np.degrees(tmp_pha_avg))

los_ph_std.append(np.degrees(tmp_pha_ch0_std))
los_ph_std.append(np.degrees(tmp_pha_ch1_std))
los_ph_std.append(np.degrees(tmp_pha_avg_std))

print("\nFinished reading!\n")

#%%
fig, axs = plt.subplots(1, 1)
for data in close_ph:
    axs.hist(data, weights=np.ones_like(data)/(len(data)))
axs.vlines(avg_tot, 0, 0.3, color='k', linewidth=3)
axs.set_ylim([0,0.3])
axs.set_xlim([-40,40])
axs.legend(["Reference value = %1.1f$\degree$" %(avg_tot), "TX1 calibration", "TX2 calibration", "Average = %1.1f$\degree$, RMSE = %1.1f$\degree$" %(np.mean(close_ph[2]), rmse(avg_tot,close_ph[2]))])
axs.set_xlabel('Phase difference between RX channels [$\degree$]')
axs.set_ylabel('Probability')
axs.grid()

plt.tight_layout()
plt.savefig("close_wi_cal.pdf",dpi=600,bbox_inches = 'tight')
plt.show()

fig, axs = plt.subplots(1, 1)
for data in far_ph:
    axs.hist(data, weights=np.ones_like(data)/(len(data)))
axs.vlines(avg_tot, 0, 0.3, color='k', linewidth=3)
axs.set_ylim([0,0.3])
axs.set_xlim([-40,40])
axs.legend(["Reference value = %1.1f$\degree$" %(avg_tot), "TX1 calibration", "TX2 calibration", "Average = %1.1f$\degree$, RMSE = %1.1f$\degree$" %(np.mean(far_ph[2]), rmse(avg_tot,far_ph[2]))])
axs.set_xlabel('Phase difference between RX channels [$\degree$]')
axs.set_ylabel('Probability')
axs.grid()

plt.tight_layout()
plt.savefig("far_wi_cal.pdf",dpi=600,bbox_inches = 'tight')
plt.show()

fig, axs = plt.subplots(1, 1)
for data in los_ph:
    axs.hist(data, weights=np.ones_like(data)/(len(data)))
axs.vlines(avg_tot, 0, 0.3, color='k', linewidth=3)
axs.set_ylim([0,0.3])
axs.set_xlim([-40,40])
axs.legend(["Reference value = %1.1f$\degree$" %(avg_tot), "TX1 calibration", "TX2 calibration", "Average = %1.1f$\degree$, RMSE = %1.1f$\degree$" %(np.mean(los_ph[2]), rmse(avg_tot,los_ph[2]))])
axs.set_xlabel('Phase difference between RX channels [$\degree$]')
axs.set_ylabel('Probability')
axs.grid()

plt.tight_layout()
plt.savefig("los_wi_cal.pdf",dpi=600,bbox_inches = 'tight')
plt.show()



#%%
#HOW MUCH A SINGLE MM contributes to the total phase shift at 2440: 63mm/64mm
c = 299792458
f = 2440000000
s = 0.001 # 1mm
var_phi  = s/(c/f)*2*np.pi
print("Radian shift: %1.2f" %var_phi)
print("Degree shift: %1.2f" %(var_phi/(2*np.pi)*360))

print("Half wavelength array spacin at 2.44GHz: %2.2f [mm]" %(1000*c/f/2))# %%

# %%
#c/f1 normal movement 
#c/f2 movement with metal plate

c1_ch0_mov, pha_ss_arr = read_data_file("/home/marcin/Desktop/wi_coeff_eval/close_cal_mov/phase_mav_100k_bp_filtered_f_2440000000_chan_0_round_51.bin",sampl_rate,decimation)
c1_ch1_mov, pha_ss_arr = read_data_file("/home/marcin/Desktop/wi_coeff_eval/close_cal_mov/phase_mav_100k_bp_filtered_f_2440000000_chan_1_round_51.bin",sampl_rate,decimation)

f1_ch0_mov, pha_ss_arr = read_data_file("/home/marcin/Desktop/wi_coeff_eval/far_cal_mov_low_tx/phase_mav_100k_bp_filtered_f_2440000000_chan_0_round_51.bin",sampl_rate,decimation)
f1_ch1_mov, pha_ss_arr = read_data_file("/home/marcin/Desktop/wi_coeff_eval/far_cal_mov_low_tx/phase_mav_100k_bp_filtered_f_2440000000_chan_1_round_51.bin",sampl_rate,decimation)

l1_ch0_mov, pha_ss_arr = read_data_file("/home/marcin/Desktop/wi_coeff_eval/los_cal_mov/phase_mav_100k_bp_filtered_f_2440000000_chan_0_round_51.bin",sampl_rate,decimation)
l1_ch1_mov, pha_ss_arr = read_data_file("/home/marcin/Desktop/wi_coeff_eval/los_cal_mov/phase_mav_100k_bp_filtered_f_2440000000_chan_1_round_51.bin",sampl_rate,decimation)


print("\nFinished reading!\n")

# %%
c1_pha = np.mean([c1_ch0_mov, c1_ch1_mov], axis = 0)
print(np.mean(c1_pha))
print(np.std(c1_pha))
print("\n")

f1_pha = np.mean([f1_ch0_mov, f1_ch1_mov], axis = 0)
print(np.mean(f1_pha))
print(np.std(f1_pha))
print("\n")

l1_pha = np.mean([l1_ch0_mov, l1_ch1_mov], axis = 0)
print(np.mean(l1_pha))
print(np.std(l1_pha))
print("\n")


#MOVEMENT DOES NOT CAUSE PHASE COEFFICIENT TO AVERAGE TO TRUE VALUE...

# %%
