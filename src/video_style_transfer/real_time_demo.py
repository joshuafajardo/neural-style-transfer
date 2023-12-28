from real_time_vst import *
from load_videos import *

import tensorflow as tf
import tensorflow_io as tfio

from pathlib import Path

project_root = Path(__file__).resolve().parents[2]

style_image = load_image(project_root / "data/style/starry-night.jpg")

# process_videos(project_root)

frames, flows = load_prepared_videos(project_root)

RealTimeVstFactory(style_image, frames, flows)