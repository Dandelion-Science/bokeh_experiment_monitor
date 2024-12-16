from numpy.typing import NDArray
from scipy.signal import periodogram, welch
from scipy.integrate import simpson
from mne.time_frequency import psd_array_multitaper
import numpy as np
from bokeh.transform import linear_cmap, log_cmap, factor_cmap
from bokeh.palettes import Viridis256
from bokeh.models import CheckboxGroup, ColumnDataSource, Slider, LinearColorMapper
from bokeh.plotting import figure, curdoc

import pandas as pd
from mne_lsl.stream import StreamLSL
import mne


class Dashboard():
    """

    """
    def __init__(self):
        self.figures = dict()
        self.sources = dict()
        self.tools = tools = "lasso_select,tap,pan,box_zoom,reset,save,hover, wheel_zoom"

    def add_spectrum(self, freqs, psd):
        self.sources['spectrum'] = ColumnDataSource(data=dict(x=freqs, y=psd))
        self.figures['spectrum'] = figure(width=1100, 
                                    height=800, 
                                    title="Spectrum", 
                                    toolbar_location="above", 
                                    tools=self.tools,
                                    x_axis_label='Frequwncy (Hz)')
        self.figures['spectrum'].vbar(x='freqs', 
                                    top='psd', 
                                    width=0.7, 
                                    source=self.sources['spectrum'],) 
        return (self.figures, self.sources)

    def add_offsets(self, names, mapper):
        self.sources['offsets'] = ColumnDataSource(data={'names': names, 
                                                    'values': [0]*len(names)})
        self.figures['offsets'] = figure(x_range=names, 
                                        height=500, 
                                        width=1500, 
                                        title="Offsets", 
                                        toolbar_location='right', 
                                        tools=self.tools)
        self.figures['offsets'].vbar(x='names', 
                                    top='values', 
                                    width=0.7, 
                                    source=self.sources['offsets'], 
                                    color={'field': 'values', 'transform': mapper})
        self.figures['offsets'].xgrid.grid_line_color = None
        self.figures['offsets'].y_range.start = -100
        self.figures['offsets'].y_range.end = 100
        self.figures['offsets'].xaxis.major_label_orientation = "vertical"
        return (self.figures, self.sources)

    def add_bands(self):
        names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        values = [0]*5
        self.sources['bands'] = ColumnDataSource(data={'names': names, 
                                                    'values': values})
        self.figures['bands'] = figure(x_range=names, 
                                        height=500, 
                                        width=1500, 
                                        title="Bands", 
                                        toolbar_location='right', 
                                        tools=self.tools)
        self.figures['bands'].vbar(x='names', 
                                    top='values', 
                                    width=0.9, 
                                    source=self.sources['bands'])
        self.figures['bands'].xgrid.grid_line_color = None
        self.figures['bands'].y_range.start = 0
        self.figures['bands'].y_range.end = 0.15
        self.figures['bands'].xaxis.major_label_orientation = "horizontal"
        return (self.figures, self.sources)

    def add_eye_tracking(self, out, mapper):
        self.sources['eye_tracking'] = ColumnDataSource(data=out)
        self.figures['eye_tracking'] = figure(width=1100, 
                                            height=800,
                                            toolbar_location="above", 
                                            tools=self.tools)
        self.figures['eye_tracking'].rect(x='x', 
                                        y='y', 
                                        width=0.03,
                                        height=0.03,
                                        source=self.sources['eye_tracking'], 
                                        fill_color={'field': 'image', 'transform': mapper})
        return (self.figures, self.sources)

    @staticmethod
    def compute_spectrum(signal, sfreq, fmin, fmax, n_fft=1024, compute_snr=False, n_neighbors=4):
        """
        Computes the power spectral density (PSD) of a signal and optionally calculates 
        the signal-to-noise ratio (SNR) for all frequency bins within a specified 
        frequency range.

        Args:
            signal: The input signal (1D numpy array).
            sfreq: The sampling frequency of the signal (in Hz).
            fmin: The lower frequency bound (in Hz).
            fmax: The upper frequency bound (in Hz).
            n_fft: The length of the FFT used.
            compute_snr: If True, calculates SNR for all frequency bins.
            n_neighbors: The number of neighboring frequencies to use for SNR calculation.

        Returns:
            psd: The power spectral density (1D numpy array).
            freqs: The corresponding frequencies (1D numpy array).
            snr: The calculated SNR for each frequency bin (1D numpy array), 
                only returned if compute_snr is True.
        """

        psd, freqs = mne.time_frequency.psd_array_welch(
            signal, sfreq, fmin=fmin, fmax=fmax, n_fft=n_fft
        )

        if compute_snr:
            snr = np.zeros_like(psd)
            for i in range(len(freqs)):
                # Exclude immediate neighbors and select n_neighbors on each side
                exclude_indices = np.arange(max(0, i - 2), min(len(freqs), i + 3))
                neighbor_indices = np.setdiff1d(np.arange(len(freqs)), exclude_indices)
                neighbor_indices = neighbor_indices[
                    np.argsort(np.abs(neighbor_indices - i))
                ][:n_neighbors]

                # Calculate SNR for the current frequency bin
                signal_power = psd[i]
                noise_power = np.mean(psd[neighbor_indices])
                snr[i] = signal_power / noise_power

            return psd, freqs, snr

        return psd, freqs
            







class EyeTrackingStream(StreamLSL):
    """
    A class for handling eye-tracking data from an LSL stream.

    Inherits from mne_lsl.stream.StreamLSL.

    Methods
    -------
    get_eye_data()
        Retrieves the latest eye-tracking data from the stream and returns it as a Pandas DataFrame.
    average_eye_data()
        Calculates the average x and y coordinates from both eyes and adds them to the DataFrame.
    """

    def __init__(self, bufsize=4, source_id='eye_tracker'):
        """
        Initializes the EyeTrackingStream.

        Parameters
        ----------
        name : str | None
            Name of the LSL stream. If None, the first stream of type 'Gaze' will be used.
        type : str
            Type of the LSL stream. Default is 'Gaze'.
        chunksize : int
            Number of samples to read at a time. Default is 4.
        source_id : str
            Unique identifier for the stream. Default is 'eye_tracker'.
        """
        super().__init__(bufsize=4, source_id='eye_tracker')
        self.last_sample = []

    def get_eye_data(self):
        """
        Retrieves the latest eye-tracking data from the stream.

        Assumes the LSL stream provides data in the format:
        ['left_gaze_x', 'left_gaze_y', 'right_gaze_x', 'right_gaze_y']

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the eye-tracking data with columns: 'left_x', 'left_y', 'right_x', 'right_y'.
        """
        eye_chls = ['left_gaze_x', 'left_gaze_y', 'right_gaze_x', 'right_gaze_y']
        data = self.get_data(picks = eye_chls)[0]
        self.last_sample = pd.DataFrame(data.T, columns = eye_chls)
        return self.last_sample

    def average_eye_data(self):
        """
        Calculates the average x and y coordinates from both eyes.

        Retrieves eye-tracking data using `get_eye_data()` and adds two new columns:
        'average_x': Average of left_gaze_x and right_gaze_x
        'average_y': Average of left_gaze_y and right_gaze_y

        Returns
        -------
        pandas.DataFrame
            DataFrame with the added average x and y coordinates.
        """
        self.last_sample = self.get_eye_data()
        self.last_sample['average_x'] = (self.last_sample['left_gaze_x'] + self.last_sample['right_gaze_x']) / 2
        self.last_sample['average_y'] = (self.last_sample['left_gaze_y'] + self.last_sample['right_gaze_y']) / 2
        return self

    def prepare_heatmap(self):
        hist, xedges, yedges = np.histogram2d(self.last_sample['average_x'], 
                                                self.last_sample['average_y'], 
                                                bins=(150,100), 
                                                density=True,
                                                range=[[-1.78, 1.78], [-1, 1]])
        xcenters = np.tile(np.linspace(-1.78,1.78,150),100)
        ycenters = np.repeat(np.linspace(-1,1,100),150)
        return pd.DataFrame({'x': xcenters, 'y': ycenters, 'image': hist.T.flatten()})

def bandpower(
    data: NDArray[np.float64],
    fs: float,
    method: str,
    band: tuple[float, float],
    relative: bool = True,
    **kwargs,
) -> NDArray[np.float64]:
    """Compute the bandpower of the individual channels.

    Parameters
    ----------
    data : array of shape (n_channels, n_samples)
        Data on which the the bandpower is estimated.
    fs : float
        Sampling frequency in Hz.
    method : 'periodogram' | 'welch' | 'multitaper'
        Method used to estimate the power spectral density.
    band : tuple of shape (2,)
        Frequency band of interest in Hz as 2 floats, e.g. ``(8, 13)``. The
        edges are included.
    relative : bool
        If True, the relative bandpower is returned instead of the absolute
        bandpower.
    **kwargs : dict
        Additional keyword arguments are provided to the power spectral density
        estimation function.
        * 'periodogram': scipy.signal.periodogram
        * 'welch'``: scipy.signal.welch
        * 'multitaper': mne.time_frequency.psd_array_multitaper

        The only provided arguments are the data array and the sampling
        frequency.

    Returns
    -------
    bandpower : array of shape (n_channels,)
        The bandpower of each channel.
    """
    # compute the power spectral density
    assert (
        data.ndim == 2
    ), "The provided data must be a 2D array of shape (n_channels, n_samples)."
    if method == "periodogram":
        freqs, psd = periodogram(data, fs, **kwargs)
    elif method == "welch":
        freqs, psd = welch(data, fs, **kwargs)
    elif method == "multitaper":
        psd, freqs = psd_array_multitaper(data, fs, verbose="ERROR", **kwargs)
    else:
        raise RuntimeError(f"The provided method '{method}' is not supported.")
    # compute the bandpower
    assert len(band) == 2, "The 'band' argument must be a 2-length tuple."
    assert (
        band[0] <= band[1]
    ), "The 'band' argument must be defined as (low, high) (in Hz)."
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    bandpower = simpson(psd[:, idx_band], dx=freq_res)
    bandpower = bandpower / simpson(psd, dx=freq_res) if relative else bandpower
    return bandpower

def define_channels_type(stream, source_id: str) -> dict:
    """Define the type of each channel based on the source_id.

    Parameters
    ----------
    source_id : str
        The source_id of the stream.

    Returns
    -------
    mapping : dict
        A dictionary mapping the channel names to their type.
    """
    if source_id == "eeg_mock":
        mapping = dict(zip(stream.ch_names, ['eeg']*64+['stim']))
    elif source_id == "acquisition":
        mapping = dict(zip(stream.ch_names, ['stim']+['eeg']*68+['eog']*4))
    else:
        raise ValueError(f"The provided source_id '{source_id}' is not supported.")
    return mapping