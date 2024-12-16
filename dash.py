from bokeh.plotting import figure, curdoc
from bokeh.layouts import row, column
from bokeh.models import CheckboxGroup, ColumnDataSource, Slider, LinearColorMapper, Slider
from bokeh.transform import linear_cmap, log_cmap, factor_cmap
from bokeh.plotting import curdoc, figure
from bokeh.palettes import Viridis256

from mne import set_log_level

from mne_lsl.stream import StreamLSL as Stream

from dash_utils import bandpower, define_channels_type, EyeTrackingStream, Dashboard

import random
import numpy as np
import time
from matplotlib import pyplot as plt

set_log_level("WARNING")
curdoc().theme = "dark_minimal"
dash = Dashboard() 
# Create a slider to change the y-axis range
slider = Slider(start=0, end=10, value=1, step=0.1, title="Y-Range")

# EEG
try: 
    eye_stream = EyeTrackingStream(bufsize=4, source_id='eye_tracker').connect()
    out = eye_stream.average_eye_data().prepare_heatmap()
    color_mapper = LinearColorMapper(palette="Viridis256", low=0, high=5)
    figures, sources = dash.add_eye_tracking(out, color_mapper)
except:
    eye_stream = None

# Eye-Tracking
if True:
    eeg_stream = Stream(bufsize=4, source_id="acquisition").connect()
    eeg_stream.set_channel_types(define_channels_type(eeg_stream, "acquisition"))
    eeg_stream.pick("eeg") 
    eeg_stream.set_eeg_reference("average")
    print(eeg_stream.info['sfreq'])
    data, ts = eeg_stream.get_data(4, picks='eeg')

    # Parameters
    threshold = 50 # Threshold for color change
    mapper = LinearColorMapper(palette=['red','blue','red'], low=-150, high=150)
    lims = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50)]

    figures, sources = dash.add_offsets(eeg_stream.ch_names, mapper)
    figures, sources = dash.add_bands()
    psd, freqs = dash.compute_spectrum(np.squeeze(data[45,:]), eeg_stream.info['sfreq'], 1, 40, n_fft = np.squeeze(data[45,:]).shape[0])
    figures, sources = dash.add_spectrum(freqs, psd)
# except:
#     eeg_stream = None




def update_data():

    if eeg_stream is not None:
        out = eye_stream.average_eye_data().prepare_heatmap()
        sources['eye_tracking'].data = out


    if eeg_stream is not None:
        data, ts = eeg_stream.get_data(4, picks='eeg')
        avg_data = np.mean(data, axis=1)
        avg_data = avg_data/(0.01/np.mean(avg_data))
        data, ts = eeg_stream.get_data(4, picks='Oz')
        psd, freqs = dash.compute_spectrum(np.squeeze(data), eeg_stream.info['sfreq'], 1, 40, n_fft = np.squeeze(data).shape[0])
        bp = [bandpower(data, eeg_stream.info["sfreq"], "periodogram", band=bs) for bs in lims]
        # slider.on_change('value', update_slider)
        
        # Update the plot
        sources['offsets'].data = {'names': eeg_stream.ch_names, 'values': avg_data}
        sources['bands'].data = {'names': ['delta', 'theta', 'alpha', 'beta', 'gamma'], 'values': bp}
        sources['spectrum'].data = {'freqs': freqs, 'psd': psd}
        

# def update_slider(attr, old, new):
#     figure['offsets'].y_range.start = -slider.value
#     figure['offsets'].y_range.end = slider.value   


# Initially hide all figures
for fig in figures.values():
    fig.visible = False

# Create a CheckboxGroup widget
checkbox_group = CheckboxGroup(
    labels=list(figures.keys()), active=[]  # Initially, no figures are selected
)

# Define the callback function
def checkbox_handler(attr, old, new):
    active_figures = [checkbox_group.labels[i] for i in checkbox_group.active]
    for key, fig in figures.items():
        fig.visible = key in active_figures

# Attach the callback to the checkbox group
checkbox_group.on_change('active', checkbox_handler)

# Create the layout
layout = column(checkbox_group, *figures.values())  # Arrange widgets and figures

curdoc().add_periodic_callback(update_data, 500)  # Update every 1 second
curdoc().add_root(layout)

