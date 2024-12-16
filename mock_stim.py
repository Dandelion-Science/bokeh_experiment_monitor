import time

from matplotlib import pyplot as plt
from mne import set_log_level

from mne_lsl.datasets import sample
from mne_lsl.player import PlayerLSL as Player
from mne_lsl.stream import StreamLSL as Stream
import numpy as np

set_log_level("WARNING")

source_id = "eeg_mock"
fname = "/Users/dannemrodov/Downloads/IR_13_S01.bdf"
player = Player(fname, chunk_size=200, source_id=source_id, n_repeat=np.inf, annotations=True).start()
#player.info