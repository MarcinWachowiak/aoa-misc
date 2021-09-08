#Scripts evaluating gathered AoA data for various scenarios
#including: high acitivity in the wireless channel, effects
#of bandpass filtration and movement during measurment
#%%
#import modules
import struct, os, ctypes
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import array
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
fig_width = 2.3622*2  # width in inches 2*6cm double column
fig_height =fig_width*golden_mean       # height in inches
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

print("\nFinished declaring functions!\n")

# %%
#Read AoA values from files
sampl_rate = int(1e6)
decim = int(1024)
dir = "/home/marcin/Desktop/aoa_variance_eval/"
ref_err = read_data_file(dir+"ref_90_deg.bin",sampl_rate,decim)
movement_err = read_data_file(dir+"movement_90_deg.bin",sampl_rate,decim)
act_filter_err = read_data_file(dir+"filter_90_deg.bin",sampl_rate,decim)
act_no_filter_err = read_data_file(dir+"no_filter_90_deg.bin",sampl_rate,decim)

#remove DC offset to expose variabiity (remove constant error)
print('ref_err %1.2f' % np.mean(ref_err))
ref_err = ref_err - np.mean(ref_err)
print('movement_err %1.2f'% np.mean(movement_err))
movement_err = movement_err - np.mean(movement_err)
print('act_filter_err %1.2f'% np.mean(act_filter_err))
act_filter_err = act_filter_err - np.mean(act_filter_err)
print('act_no_filter_err %1.2f' % np.mean(act_no_filter_err))
act_no_filter_err = act_no_filter_err - np.mean(act_no_filter_err)
time_vals = np.linspace(0,60,len(ref_err))
#%%
#Histogram comparsion group plot
fig, axs = plt.subplots(2,2)
props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
n_bins = 200
txt_x = 0.68
txt_y = 0.80
axs[1][0].set_title("c) Low RF activity",fontsize=8)
axs[1][0].set_ylabel('Probability')
axs[1][0].hist(ref_err, bins=n_bins, weights=np.ones_like(ref_err)/(len(ref_err)),zorder=3, log=True)
axs[1][0].set_xlim([-15,15])
#axs[1][0].set_ylim([0.0,0.6])
axs[1][0].text(txt_x, txt_y,("$\sigma = %1.2f\degree$" % np.std(ref_err)), transform=axs[1][0].transAxes, fontsize=8, verticalalignment='bottom', bbox=props)
axs[1][0].set_xlabel('AoA error [$\degree$]')
#axs[1][0].yaxis.set_major_locator(MaxNLocator(4))
axs[1][0].set_yticks([1e-5,1e-4, 1e-3, 1e-2, 1e-1, 1])
axs[1][0].xaxis.set_major_locator(MaxNLocator(6)) 
axs[1][0].grid(zorder=0)

axs[1][1].set_title("d) Low RF activity with movement",fontsize=8)
axs[1][1].hist(movement_err, bins=n_bins, weights=np.ones_like(movement_err)/(len(movement_err)),zorder=3, log=True)
axs[1][1].set_xlim([-15,15])
#axs[1][1].set_ylim([0.0,0.04])
#axs[1][1].set_ylabel('Probability')
axs[1][1].text(txt_x, txt_y,("$\sigma = %1.2f\degree$" % np.std(movement_err)), transform=axs[1][1].transAxes, fontsize=8, verticalalignment='bottom', bbox=props)
axs[1][1].set_xlabel('AoA error [$\degree$]')
#axs[1][1].yaxis.set_major_locator(MaxNLocator(4))
axs[1][1].set_yticks([1e-5,1e-4, 1e-3, 1e-2, 1e-1, 1])
 
axs[1][1].xaxis.set_major_locator(MaxNLocator(6)) 
axs[1][1].grid(zorder=0)


axs[0][0].set_title("a) High RF activity",fontsize=8)
axs[0][0].set_ylabel('Probability')
#axs[0][0].set_xlabel('AoA error [$\degree$]')
axs[0][0].hist(act_no_filter_err, bins=n_bins, weights=np.ones_like(act_no_filter_err)/(len(act_no_filter_err)),zorder=3, log=True)
axs[0][0].text(txt_x, txt_y,("$\sigma = %1.2f\degree$" % np.std(act_no_filter_err)), transform=axs[0][0].transAxes, fontsize=8, verticalalignment='bottom', bbox=props)
axs[0][0].set_xlim([-90,90])
axs[0][0].set_yticks([1e-5,1e-4, 1e-3, 1e-2, 1e-1, 1])
axs[0][0].set_xticks([-90,-60,-30,0,30,60,90])
axs[0][0].grid(zorder=0)

axs[0][1].set_title("b) High RF activity with BP filter",fontsize=8)
axs[0][1].hist(act_filter_err, bins=n_bins, weights=np.ones_like(act_filter_err)/(len(act_filter_err)),zorder=3, log=True)
axs[0][1].set_xlim([-15,15])
#axs[0][1].set_xlabel('AoA error [$\degree$]')
axs[0][1].set_ylim([0.0,0.2])
#axs[0][1].set_ylabel('Probability')
axs[0][1].text(txt_x, txt_y,("$\sigma = %1.2f\degree$" % np.std(act_filter_err)), transform=axs[0][1].transAxes, fontsize=8, verticalalignment='bottom', bbox=props)
#axs[0][1].yaxis.set_major_locator(MaxNLocator(4))
axs[0][1].set_yticks([1e-5,1e-4, 1e-3, 1e-2, 1e-1, 1])
axs[0][1].xaxis.set_major_locator(MaxNLocator(6))

axs[0][1].grid(zorder=0)

plt.tight_layout()
plt.savefig("aoa_variance_eval.pdf",dpi=600,bbox_inches = 'tight')
plt.show()
# %%
