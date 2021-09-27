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
mpl.rcParams["legend.borderpad"] = 0.2
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

def plot_data_with_stats(sampl_arr,sampl_spacing,is_phase):
    #calculate stats
    coeff_mean = np.mean(sampl_arr,dtype=np.float64)
    coeff_std_dev = np.std(sampl_arr,dtype=np.float64)
    rel_std_var = np.abs(100*coeff_std_dev/coeff_mean)
    
    stats_str = r"$\sigma$ = %1.3f" % coeff_std_dev
    y_label_str ="Power correction coefficent value [-]"
    leg_loc = 4
    txt_x = 0.855
    txt_y = 0.15

    if (is_phase):
        angle_std_variation = np.rad2deg(coeff_std_dev)
        stats_str = r"$\sigma$ = %1.4f $\approx$ %1.2f$\degree$" % (coeff_std_dev,angle_std_variation)
        y_label_str ="Coefficient value [rad]"
        leg_loc = 1
        txt_x = 0.74
        txt_y = 0.8
        
    props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
    fig, ax = plt.subplots()
    # place a text box in upper left in axes coords
    ax.text(txt_x, txt_y, stats_str, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', bbox=props)
    #simple data plot
    plt.xlabel("Time [h]")
    plt.ylabel(y_label_str)

    timebase = np.linspace(0,sampl_spacing*len(sampl_arr),len(sampl_arr))
    ax.plot(timebase, sampl_arr, linewidth=0.1)

    decimation_rate = 1000
    timebase_decim = timebase[0:len(timebase):decimation_rate]
    sampl_arr_decim = sampl_arr[0:len(sampl_arr):decimation_rate]

    poly_deg = 7
    poly_coeff = poly.polyfit(timebase_decim, sampl_arr_decim,poly_deg)
    ffit = poly.polyval(timebase_decim, poly_coeff)
    ax.plot(timebase_decim,ffit)
    
    ax.legend(['_nolegend_', "Polynomial fit"],loc=leg_loc)

    x_ticks_loc = (np.arange(min(timebase),max(timebase),3600))
    x_ticks_val = arr_get_hh(x_ticks_loc)
    #y_ticks_loc = ((np.round(np.linspace(min(sampl_arr),3),np.round(max(sampl_arr),3),8)))
    plt.xticks(x_ticks_loc,x_ticks_val)
    plt.locator_params(axis='y',nbins=11)
    plt.xlim([min(x_ticks_loc),max(x_ticks_loc)])
    plt.grid()

def plot_data_with_stats_composite(fig, ax,sampl_arr_list,ss_l):
    
    #simple data plot
    plt.xlabel("Time [min]")
    plt.ylabel("Coefficient value [rad]")

    for arr_idx in range (len(sampl_arr_list)):
        timebase = np.linspace(0,ss_l[arr_idx]*len(sampl_arr_list[arr_idx]),len(sampl_arr_list[arr_idx]))
        ax.plot(timebase, sampl_arr_list[arr_idx], linewidth=1)

def trim_data_and_plot(sampl_arr,num_samples,sampl_spacing):
    data_begin = sampl_arr[ : num_samples]
    data_mid = sampl_arr[num_samples : (len(sampl_arr)-num_samples)]
    data_end = sampl_arr[(len(sampl_arr)-num_samples) : ]
    
    t_base_begin = np.linspace(0,num_samples*sampl_spacing,num_samples)
    t_base_mid = np.linspace(sampl_spacing*num_samples,sampl_spacing*(len(sampl_arr)-num_samples),len(sampl_arr)-2*num_samples)
    #t_base_end = np.linspace(sampl_spacing*(len(sampl_arr)-num_samples),sampl_spacing*len(sampl_arr),num_samples)

    ax1 = plt.subplot(212)
    ax1.plot(t_base_mid,data_mid)
    ax1.set_title('Middle')
    x_ticks_loc = (np.linspace(min(t_base_mid),max(t_base_mid),8).round())
    x_ticks_val = arr_get_h_m_format(x_ticks_loc)
    ax1.set_xticks(x_ticks_loc)
    ax1.set_xticklabels(x_ticks_val)
    ax1.set_xlabel("Time [hh:mm]")
    ax1.grid()

    ax2 = plt.subplot(221)
    ax2.plot(t_base_begin,data_begin)
    ax2.set_title('Beginning')
    ax2.set_xlabel("Time [s]")
    ax2.grid()

    ax3 = plt.subplot(222)
    ax3.plot(t_base_begin,data_end)
    ax3.set_title('Tail')
    ax3.set_xlabel("Time [s]")
    ax3.grid()

    plt.tight_layout()
    plt.show()

    return data_mid 

def trim_data(sampl_arr,num_samples):
    return sampl_arr[num_samples : (len(sampl_arr)-num_samples)]

def trim_data_spc(sampl_arr,num_s_begin, num_s_end):
    return sampl_arr[num_s_begin : (len(sampl_arr)-num_s_end)]

def arr_get_h_m_format(seconds_arr):
    ret_str_arr = list()
    for idx in range(len(seconds_arr)):
        ret_str_arr.append(str(datetime.timedelta(seconds=seconds_arr[idx]))[:-3])
    return ret_str_arr

def arr_get_hh(seconds_arr):
    ret_str_arr = list()
    for idx in range(len(seconds_arr)):
        ret_str_arr.append(str(datetime.timedelta(seconds=seconds_arr[idx]))[:-6])
    return ret_str_arr

def fw_replace_nan(sampl_arr):
    #replace inf if any
    mask = np.logical_or(np.isinf(sampl_arr),np.isnan(sampl_arr))
    idx = np.where(~mask,np.arange(mask.size),0)
    np.maximum.accumulate(idx, out=idx)
    return sampl_arr[idx]

print("\nFinished declaring functions!\n")

#%%
sampl_rate = int(1e6)
decimation = int(1024)
n_skip_samples = int(1000)

dir_name = "/home/marcin/Desktop/aoa_pub_files/aoa_meas/"

aoa_err = list()
aoa_err_mov = list()

props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)

aoa_err_files = ["angle_90_err.bin","angle_135_err.bin"]
aoa_mov_err_files = ["angle_90_mov_err.bin","angle_135_mov_err.bin"]

for file in aoa_err_files:
    aoa, _ = read_data_file(dir_name+ file,sampl_rate,decimation)
    aoa_err.append(fw_replace_nan(trim_data(np.array(aoa),n_skip_samples)))

for file in aoa_mov_err_files:
    aoa, _ = read_data_file(dir_name+ file,sampl_rate,decimation)
    aoa_err_mov.append(fw_replace_nan(trim_data(np.array(aoa),n_skip_samples)))
print("\nFinished reading!\n")

#%%
fig, ax1 = plt.subplots(1, 1)
n_bins = 100
ax1.hist(aoa_err[0], n_bins, alpha = 0.9, weights=np.ones_like(aoa_err[0])/(len(aoa_err[0])), zorder=4)
ax1.hist(aoa_err_mov[0], n_bins, alpha = 0.9, weights=np.ones_like(aoa_err_mov[0])/(len(aoa_err_mov[0])), zorder=3)
ax1.set_xlim([-10,10])
ax1.set_ylim([0.0,0.06])

ax1.set_title("Source at $90\degree$")
ax1.legend([r"$\overline{\theta}_{stat.}=%1.1f, \sigma=%1.1f\degree $" %(np.mean(aoa_err[0]), np.std(aoa_err[0])), r"$\overline{\theta}_{mov.}=%1.1f, \sigma=%1.1f \degree$" %(np.mean(aoa_err_mov[0]), np.std(aoa_err_mov[0]))])
ax1.set_ylabel("Probability")
ax1.set_xlabel('AoA error [$\degree$]')
ax1.grid(zorder=0)

plt.tight_layout()
plt.savefig("aoa_field_acc_90.pdf",dpi=600,bbox_inches = 'tight')
plt.show()

fig, ax2 = plt.subplots(1, 1)
ax2.hist(aoa_err[1], n_bins, alpha = 0.9, weights=np.ones_like(aoa_err[1])/(len(aoa_err[1])), zorder=4)
ax2.hist(aoa_err_mov[1], n_bins, alpha = 0.9, weights=np.ones_like(aoa_err_mov[1])/(len(aoa_err_mov[1])), zorder=3)
ax2.set_xlim([-10,10])
ax2.set_ylim([0.0,0.06])
ax2.grid(zorder=0)
ax2.set_ylabel("Probability")

ax2.set_title("Source at $135\degree$")
ax2.set_xlabel('AoA error [$\degree$]')
ax2.legend([r"$\overline{\theta}_{stat.}=%1.1f, \sigma=%1.1f\degree$" %(np.mean(aoa_err[1]), np.std(aoa_err[1])), r"$\overline{\theta}_{mov.}=%1.1f, \sigma= %1.1f\degree $" %(np.mean(aoa_err_mov[1]), np.std(aoa_err_mov[1]))])
plt.tight_layout()
plt.savefig("aoa_field_acc_135.pdf",dpi=600,bbox_inches = 'tight')
plt.show()
# %%
