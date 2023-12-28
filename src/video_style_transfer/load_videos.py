# load_videos.py

"""Loads and prepares the videos for training."""

__all__ = ["load_image", "process_videos", "load_prepared_videos"]
__author__ = "joshuafajardo"
__version__ = "0.1.0"

import subprocess
import shutil

import numpy as np
import torch
import torchvision

import tensorflow as tf

from pathlib import Path
from tqdm import tqdm


FRAME_COUNT_FILLER = 8
FRAME_DIMENSIONS = (640, 360)
SUPPORTED_VIDEO_EXTENSIONS = ["mp4", "mov"]

# Relative directories
REL_FRAME_DIR = "data/videos/frames"
REL_FLOWS_DIR = "data/videos/flows"
REL_ORIGINAL_DIR = "data/videos/original"
REL_SCALED_DIR = "data/videos/scaled"

FRAMES_FILE = "frames.npy"
FLOWS_FILE = "flows.npy"


def load_image(image_path):
    """Loads an image from the given path."""
    image = tf.keras.utils.load_img(str(image_path))
    return tf.keras.utils.img_to_array(image)


def load_prepared_videos(project_root):
    """Loads the prepared videos from the project root."""
    frame_dir = project_root / REL_FRAME_DIR
    flow_dir = project_root / REL_FLOWS_DIR
    frames = {}
    flows = {}
    for dir in frame_dir.glob("*/"):
        video_name = dir.name

        curr_frames = np.load(frame_dir / video_name / FRAMES_FILE)
        curr_flows = np.load(flow_dir / video_name / FLOWS_FILE)

        frames[video_name] = tf.convert_to_tensor(curr_frames)
        flows[video_name] = tf.convert_to_tensor(curr_flows)
    return frames, flows


def process_videos(project_root):
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

    original_videos_dir = project_root / REL_ORIGINAL_DIR
    scaled_videos_dir = project_root / REL_SCALED_DIR
    clear_dir(scaled_videos_dir)
    scale_videos(original_videos_dir, scaled_videos_dir)

    frame_dir = project_root / REL_FRAME_DIR
    flow_dir = project_root / REL_FLOWS_DIR
    clear_dir(frame_dir)
    clear_dir(flow_dir)


    videos = get_video_paths_from_dir(scaled_videos_dir)
    print("Saving frames and flows.")
    for video in tqdm(videos):
        video_name = Path(video).stem
        output_frames_dir = frame_dir / video_name
        output_flows_dir = flow_dir / video_name
        output_frames_dir.mkdir()
        output_flows_dir.mkdir()

        frames = generate_and_save_frames(video, output_frames_dir)
        _ = generate_and_save_flows(frames, output_flows_dir)


def scale_videos(input_dir, output_dir):
    """
    Scales the videos in the input_dir and saves them as mp4s in the
    output_dir.
    """
    for extension in SUPPORTED_VIDEO_EXTENSIONS:
        videos = get_video_paths_from_dir(input_dir, extension)
        for video in tqdm(videos):
            output_path = output_dir / f"{Path(video).stem}.mp4"
            subprocess.run(
                f'ffmpeg -i "{video}" -vf ' + 
                f'scale={FRAME_DIMENSIONS[0]}:{FRAME_DIMENSIONS[1]} ' +
                f'"{output_path}"',
                capture_output=True, shell=True, check=True)


def get_video_paths_from_dir(input_dir, extension="mp4"):
    """Returns a list of videos from the directory."""
    return input_dir.glob(f"*.{extension}")


def generate_and_save_frames(video_path, output_dir):
    """
    Loads and saves the frames of the video as pngs in the
    output_dir.
    Videos are loaded as tensors of shape (T, H, W, C), with values
    in the range [0, 255].
    """
    frames, _, _ = torchvision.io.read_video(str(video_path), pts_unit="sec",
                                             end_pts=1,
                                             output_format="THWC")
    for i in range(frames.shape[0]):
        frame_num = str(i).zfill(FRAME_COUNT_FILLER)
        output_path = output_dir / f"frame_{frame_num}.png"
        frame_as_image = torch.permute(frames[i], (2, 0, 1))
        torchvision.io.write_png(frame_as_image.cpu(), str(output_path))
    np.save(output_dir / FRAMES_FILE, frames.cpu().detach().numpy())
    return frames


def generate_and_save_flows(frames, output_dir):
    """Returns a list of optical flows between the frames."""
    transforms = torchvision.models.optical_flow.Raft_Large_Weights.DEFAULT.transforms()

    frames = frames.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)

    start_frames, end_frames = frames[:-1], frames[1:]
    start_frames, end_frames = transforms(start_frames, end_frames)

    raft_model = torchvision.models.optical_flow.raft_large(weights="DEFAULT")
    raft_model.eval()

    flows = raft_model(start_frames, end_frames)[-1]
    for i in range(flows.shape[0]):
        flow_image = torchvision.utils.flow_to_image(flows[i])
        frame_num = str(i).zfill(FRAME_COUNT_FILLER)
        output_path = output_dir / f"flow_{frame_num}.png"
        torchvision.io.write_png(flow_image, str(output_path))
    flows.permute(0, 2, 3, 1)  # (T, C, H, W) -> (T, H, W, C)
    np.save(output_dir / FLOWS_FILE, flows.cpu().detach().numpy())
    return flows


def clear_dir(dir):
    """Recursively clears the directory."""
    if dir.exists():
        shutil.rmtree(dir)
    dir.mkdir(parents=True)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    process_videos(project_root)
