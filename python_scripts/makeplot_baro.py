#!/usr/bin/env python
# coding: utf-8

# # ROMY - Barometer

# 

# In[1]:


import os
import obspy as obs
import matplotlib.pyplot as plt
import numpy as np

from obspy import read_inventory
from scipy.signal import hilbert


# In[2]:


from functions.trim_stream import trim_stream
from functions.read_sds import __read_sds
from functions.get_mean_promy_pressure import __get_mean_promy_pressure
from functions.get_mean_rmy_pressure import __get_mean_rmy_pressure
from functions.regressions import regressions


# In[4]:


if os.uname().nodename == 'lighthouse':
    root_path = '/home/andbro/'
    data_path = '/home/andbro/kilauea-data/'
    archive_path = '/home/andbro/freenas/'
    bay_path = '/home/andbro/bay200/'
elif os.uname().nodename == 'kilauea':
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'
    bay_path = '/bay200/'
elif os.uname().nodename == 'teide':
    root_path = '/home/sysopromy/'
    data_path = '/freenas-ffb-01/'
    archive_path = '/freenas-ffb-01/'
    bay_path = '/bay200/'
    lamont_path = '/lamont/'
elif os.uname().nodename in ['lin-ffb-01', 'ambrym', 'hochfelln']:
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'
    bay_path = '/bay200/'


# In[ ]:





# ## Configurations

# In[62]:


config = {}

# output path for figures
config['path_to_figs'] = archive_path+"romy_html_monitor/figures/"

# path to data archive
config['path_to_data'] = data_path+"romy_events/data/"

# path to data sds
config['path_to_sds'] = archive_path+"temp_archive/"


# adjust time interval
# config['t1'] = obs.UTCDateTime("2024-03-04 05:00")
# config['t2'] = obs.UTCDateTime("2024-03-04 07:00")
# config['toffset'] = 3*3600

config['t1'] = obs.UTCDateTime("2024-04-23 02:00")
config['t2'] = obs.UTCDateTime("2024-04-23 05:00")
config['toffset'] = 3*3600

# quasi live mode
config['toffset'] = 3*3600
config['t2'] = obs.UTCDateTime.now() - 86400
config['t1'] = config['t2'] - config['toffset']

# data
config['tbeg'] = config['t1'] - config['toffset']
config['tend'] = config['t2'] + config['toffset']

# frequency range
config['fmin'] = 0.0005
config['fmax'] = 0.01

# ROMY coordinates
config['sta_lon'] = 11.275501
config['sta_lat'] = 48.162941


# ## Load Data

# In[6]:


st0 = obs.Stream()
try:
    st0 += __read_sds(config['path_to_sds'], "BW.ROMY.00.BJZ", config['tbeg'], config['tend'])
    st0 += __read_sds(config['path_to_sds'], "BW.ROMY.00.BJN", config['tbeg'], config['tend'])
    st0 += __read_sds(config['path_to_sds'], "BW.ROMY.00.BJE", config['tbeg'], config['tend'])
    
    st0.merge(fill_value="interpolate")
except:
    pass


# In[7]:


ffbi0 = obs.Stream()

try:
    ffbi_inv = read_inventory(archive_path+"stationxml_ringlaser/station_BW_FFBI.xml")

    ffbi0 += __read_sds(bay_path+"mseed_online/archive/", "BW.FFBI..BDF", config['tbeg'], config['tend'])
    ffbi0 += __read_sds(bay_path+"mseed_online/archive/", "BW.FFBI..BDO", config['tbeg'], config['tend'])

    if len(ffbi0) != 2:
        ffbi0.merge();

    for tr in ffbi0:
        if "F" in tr.stats.channel:
            tr = tr.remove_response(ffbi_inv, water_level=10)
        if "O" in tr.stats.channel:
            # tr.data = tr.data /1.0 /6.28099e5 /100e-3   # gain=1 sensitivity_reftek=6.28099e5count/V; sensitivity = 100 mV/hPa
            tr.data = tr.data /1.0 /6.28099e5 /1e-5   # gain=1 sensitivity_reftek=6.28099e5count/V; sensitivity = 100 mV/hPa

    # ffbi0 = ffbi0.resample(1.0, no_filter=True)

    ffbi0.merge();

except Exception as e:
    print(e)


# In[8]:


# promy = obs.Stream()
# try:
#     promy = __get_mean_promy_pressure(["03", "04", "05", "07", "09"],
#                                     config['tbeg'],
#                                     config['tend'],
#                                     archive_path,
#                                     plot=True
#                                     )
# except Exception as e:
#     print(e)


# In[9]:


# rmy = obs.Stream()
# try:
#     rmy = __get_mean_rmy_pressure(["PROMY", "ALFT", "TON", "BIB", "GELB", "GRMB"],
#                               config['tbeg'], config['tend'],
#                               archive_path,
#                               plot=True,
#                              )
# except Exception as e:
#     print(e)


# In[10]:


# add hilbert transform of ffbi0
bdh = ffbi0.select(channel="BDO").copy()
bdh[0].data = np.imag(hilbert(bdh[0].data))
bdh[0].stats.channel = "BDH"

# rotations
rot1 = st0.select(station=f"ROMY", channel="*J*").copy()

# add tilt
tilt = rot1.copy()
tilt = tilt.integrate("spline")
for tr in tilt:
    tr.stats.channel = tr.stats.channel.replace("J", "T")

# make combined stream
stt = obs.Stream()
stt += tilt.copy()
stt += rot1.copy()
stt += ffbi0.copy()
stt += bdh.copy()

try:
    stt += promy.copy()
except Exception as e:
    print(e)

try:
    stt += rmy.copy()
except Exception as e:
    print(e)


# In[96]:


try:
    stt = obs.read("./test.mseed")
except:
    pass


# In[ ]:


for tr in stt:
    try:
        tr = tr.detrend("demean").detrend("linear").detrend("demean")
        tr = tr.taper(0.1)
        tr = tr.filter("lowpass", freq=1.0, corners=2, zerophase=True)
        tr = tr.filter("highpass", freq=1e-4, corners=2, zerophase=True)

        tr = tr.resample(1.0, no_filter=True)
    except:
        print(f"Resampling failed for {tr.stats.station}.{tr.stats.channel}...")
        stt.remove(tr)

# add rad BDO from ffbi
try:
    ffbi_raw = ffbi0.select(station="FFBI", channel="*DO").copy()
    ffbi_raw[0].stats.channel = "BDX"
    ffbi_raw[0].data /= 100 # to hPa
    stt += ffbi_raw.copy()
except:
    pass

stt = stt.trim(config['t1'], config['t2'])

stt = trim_stream(stt)

# stt.plot(equal_scale=False);

print(stt)


# In[98]:


def compute_regression_models_with_regressions(st, pressure_ch="BDO", hilbert_ch="BDH", reg_method="theilsen"):
    """
    Compute regression models for each J and T channel using pressure and its Hilbert transform
    using the regressions function from functions.regressions.
    
    Parameters:
    -----------
    st : obspy.Stream
        Stream containing J and T channels, as well as pressure channel
    pressure_ch : str
        Channel code for pressure data
    hilbert_ch : str
        Channel code for Hilbert transform of pressure data
    reg_method : str
        Regression method to use ('ols', 'ransac', 'theilsen', 'odr')
        
    Returns:
    --------
    dict : Dictionary containing regression results for each channel
    """
    import pandas as pd
    import numpy as np
    
    results = {}
    
    # Find all J and T channels
    j_channels = [tr.stats.channel for tr in st if tr.stats.channel.startswith('BJ')]
    t_channels = [tr.stats.channel for tr in st if tr.stats.channel.startswith('BT')]
    
    # Get pressure data and its Hilbert transform
    pressure = st.select(channel=pressure_ch)[0].data
    hilbert_pressure = st.select(channel=hilbert_ch)[0].data
    
    # Create time array
    time = st.select(channel=pressure_ch)[0].times()
    
    # Process all channels
    for channel in j_channels + t_channels:
        # Get channel data
        data = st.select(channel=channel)[0].data
        
        # Create DataFrame for regression
        df = pd.DataFrame({
            'time': time,
            'pressure': pressure,
            'hilbert': hilbert_pressure,
            'channel_data': data
        })
        
        # Perform regression
        reg_result = regressions(
            df, 
            features=['pressure', 'hilbert'], 
            target='channel_data',
            reg=reg_method,
            verbose=False
        )
        
        # Calculate model and residual
        model = np.array(reg_result['dp'])
        residual = data - model
        
        # Calculate variance reduction
        var_orig = np.var(data)
        var_resid = np.var(residual)
        var_reduction = (var_orig - var_resid) / var_orig * 100
        
        # Store results
        results[channel] = {
            'model': model,
            'residual': residual,
            'var_reduction': var_reduction,
            'regression_result': reg_result
        }
    
    return results


# In[111]:


def plot_waveforms(stt, fmin=None, fmax=None):
    """
    Plot barometric pressure and ROMY components in subplots.
    
    Parameters:
    -----------
    stt : obspy.Stream
        Stream containing FFBI and ROMY components
    fmin : float, optional
        Lower frequency for bandpass filter in Hz
    fmax : float, optional
        Upper frequency for bandpass filter in Hz
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plots
    """
    
    # Create a copy of the stream to avoid modifying the original
    st = stt.copy()
    
    # Apply bandpass filter if frequencies are provided
    if fmin is not None and fmax is not None:
        st = st.detrend("linear")
        st = st.detrend("demean")
        st = st.taper(0.1)
        st = st.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)

    # Compute regression models
    print(f"Computing regression models for {len(st.select(channel='*J*'))+len(st.select(channel='*T*'))} traces")
    regression_results = compute_regression_models_with_regressions(
        st, 
        pressure_ch="BDO", 
        hilbert_ch="BDH", 
        reg_method="ransac"
    )

    # scaling factor for ROMY components
    j_scale = 1e9

    # define font size for label
    fs = 12

    # Create figure with subplots
    fig, axes = plt.subplots(9, 1, figsize=(20, 12), sharex=True)
    plt.subplots_adjust(hspace=0.1)
    
    # Time vector in hours
    try:
        tscale = 1/3600  # convert to hours
    except:
        print("Error getting time array")
        return None
    
    # Plot absolute barometric pressure (FFBI DO)
    try:
        bdo = stt.select(station="FFBI", channel="*DX")[0]
        axes[0].plot(bdo.times()*tscale,
                     bdo.data,
                     color='k',
                     label=f'{bdo.stats.station} (absolute)'
                     )
        ymax = np.max(np.abs(bdo.data))
        axes[0].set_ylabel('P (hPa)', fontsize=fs)
        axes[0].legend(loc='upper right', fontsize=fs)
        axes[0].grid(True, linestyle=':')
    except:
        axes[0].text(0.5, 0.5, 'No BDO data available', 
                    ha='center', va='center', transform=axes[0].transAxes, fontsize=fs)
    
    # Plot bandpass filtered pressure
    try:
        bdo = st.select(channel="*DO")[0]
        axes[1].plot(bdo.times()*tscale,
                     bdo.data,
                     color='k',
                     label=f'{bdo.stats.station}.{bdo.stats.channel} (filtered)'
                     )
        ymax = np.max(np.abs(bdo.data))
        axes[1].set_ylim(-ymax, ymax)
        axes[1].set_ylabel('P (Pa)', fontsize=fs)
        axes[1].legend(loc='upper right', fontsize=fs)
        axes[1].grid(True, linestyle=':')
    except:
        axes[1].text(0.5, 0.5, 'No filtered BDO data available', 
                    ha='center', va='center', transform=axes[1].transAxes, fontsize=fs)
    
    # Plot Hilbert transform of DH channel
    try:
        bdh = st.select(channel="*DH")[0]
        axes[2].plot(bdh.times()*tscale, 
                     bdh.data, 
                     color='k', 
                     label=f'Hilbert[{bdo.stats.station}.{bdo.stats.channel}]'
                     )
        ymax = np.max(np.abs(bdh.data))
        axes[2].set_ylim(-ymax, ymax)
        axes[2].set_ylabel('P (Pa)', fontsize=fs)
        axes[2].legend(loc='upper right', fontsize=fs)
        axes[2].grid(True, linestyle=':')
    except:
        axes[2].text(0.5, 0.5, 'No Hilbert transform data available', 
                    ha='center', va='center', transform=axes[2].transAxes, fontsize=fs)
    
    # Plot ROMY components (Z, N, E)
    components = ['Z', 'N', 'E']
    for i, comp in enumerate(components):
        try:
            tr = st.select(channel=f"*J{comp}")[0]
            data = tr.data * j_scale  # Convert to nrad
            axes[i+3].plot(tr.times()*tscale,
                           data,
                           label=f'{tr.stats.station}.{tr.stats.channel}',
                           color='k'
                           )
            axes[i+3].plot(tr.times()*tscale,
                           regression_results[tr.stats.channel]['residual'] * j_scale,
                           color='red',
                           ls='--',
                           label=f'Residual (VR={round(regression_results[tr.stats.channel]["var_reduction"], 1)}%')
            ymax = np.max(np.abs(data))
            axes[i+3].set_ylim(-ymax, ymax)
            axes[i+3].set_ylabel('$\dot{\omega}$ (nrad)', fontsize=fs)
            axes[i+3].legend(loc='upper right', fontsize=fs)
            axes[i+3].grid(True, linestyle=':')
        except Exception as e:
            print(e)
            axes[i+3].text(0.5, 0.5, f'No ROMY.{comp} data available', 
                          ha='center', va='center', transform=axes[i+3].transAxes, fontsize=fs)
    
    # Plot ROMY Tilt components
    for i, comp in enumerate(components):
        try:
            tr = st.select(channel=f"*T{comp}")[0].copy()
            data = tr.data * j_scale  # Convert to nrad
            axes[i+6].plot(tr.times()*tscale,
                           data,
                           color='k',
                           label=f'{tr.stats.station}.{tr.stats.channel} (integrated)'
                           )
            axes[i+6].plot(tr.times()*tscale,
                           regression_results[tr.stats.channel]['residual'] * j_scale,
                           color='red',
                           ls='--',
                           label=f'Residual (VR={round(regression_results[tr.stats.channel]["var_reduction"], 1)}%'
                           )
            ymax = np.max(np.abs(data))
            axes[i+6].set_ylim(-ymax, ymax)
            axes[i+6].set_ylabel('$\omega$ (nrad)', fontsize=fs)
            axes[i+6].legend(loc='upper right', fontsize=fs)
            axes[i+6].grid(True, linestyle=':')
        except:
            axes[i+6].text(0.5, 0.5, f'No integrated ROMY.{comp} data available', 
                          ha='center', va='center', transform=axes[i+6].transAxes, fontsize=fs)
    for ax in axes:
        # Hide the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(True)

        ax.set_xlim(left=0)

    axes[8].spines['bottom'].set_visible(True)

    # Add frequency band information if filtering was applied
    if fmin is not None and fmax is not None:
        title = f'Bandpass filtered {fmin}-{fmax} Hz'
        title += f' | Model = a*P + b*H(P)'
        fig.suptitle(title, y=1.01, fontsize=fs)
    
    # Set common x-label
    axes[-1].set_xlabel(f'Time (hours) from {str(config["tbeg"]).split(".")[0]} UTC', fontsize=fs)
    
    plt.tight_layout()
    return fig


# In[112]:


fig = plot_waveforms(stt, fmin=config['fmin'], fmax=config['fmax'])


# In[ ]:


fig.savefig(f"{config['path_to_figs']}html_romy_baro.png")

try:
    fig.savefig(f"/freenas-ffb-01/baro_monitor_plots/{config['t1']}.png")
except Exception as e:
    print(e)
    pass

# In[ ]:




