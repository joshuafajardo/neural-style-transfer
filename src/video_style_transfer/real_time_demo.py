from real_time_vst import *
from load_videos import *

import tensorflow as tf
import tensorflow_io as tfio

from pathlib import Path

style_image = load_image("data/style/starry-night.jpg")

project_root = Path(__file__).resolve().parents[2]
process_videos(project_root)
frames, flows = load_prepared_videos(project_root)

RealTimeVstFactory(style_image, frames, flows)
