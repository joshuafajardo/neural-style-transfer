import subprocess
import shutil

import numpy as np
import torch
import torchvision

from pathlib import Path
from tqdm import tqdm


FRAME_COUNT_FILLER = 8
FRAME_DIMENSIONS = (640, 360)
SUPPORTED_VIDEO_EXTENSIONS = ["mp4", "mov"]


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
                f'ffmpeg -i "{video}" -vf scale=128:128 "{output_path}"',
                stdout=subprocess.PIPE, shell=True, check=True)


def get_video_paths_from_dir(input_dir, extension="mp4"):
    """Returns a list of videos from the directory."""
    return input_dir.glob(f"*.{extension}")


def load_and_save_frames(video_path, output_dir):
    """
    Loads and saves the frames of the video as pngs in the
    output_dir.
    """
    frames, _, _ = torchvision.io.read_video(str(video_path), pts_unit="sec",
                                             end_pts=1,
                                             output_format="TCHW")
    frames_cpu = frames.cpu()  # In case the default device isn't CPU.
    for i in range(frames.shape[0]):
        frame = frames_cpu[i]
        frame_num = str(i).zfill(FRAME_COUNT_FILLER)
        output_path = output_dir / f"frame_{frame_num}.png"
        torchvision.io.write_png(frame, str(output_path))
    return frames


def load_and_save_flows(frames, output_dir):
    """Returns a list of optical flows between the frames."""
    print("function start")
    transforms = torchvision.models.optical_flow.Raft_Large_Weights.DEFAULT.transforms()

    start_frames, end_frames = frames[:-1], frames[1:]
    start_frames, end_frames = transforms(start_frames, end_frames)
    print(start_frames.shape)
    print(end_frames.shape)

    print("creating model")
    raft_model = torchvision.models.optical_flow.raft_large(weights="DEFAULT")
    raft_model.eval()
    print("model created")

    print("hello world")
    flows = raft_model(start_frames, end_frames)[-1]
    for i in range(flows.shape[0]):
        flow_image = torchvision.utils.flow_to_image(flows[i])
        print(flow_image.shape)
        frame_num = str(i).zfill(FRAME_COUNT_FILLER)
        output_path = output_dir / f"flow_{frame_num}.png"
        torchvision.io.write_png(flow_image, str(output_path))
    return flows


def clear_dir(dir):
    """Recursively clears the directory."""
    if dir.exists():
        shutil.rmtree(dir)
    dir.mkdir(parents=True)


def main():
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = Path(__file__).resolve().parents[2]

    original_videos_dir = project_root / "data/videos/original"
    scaled_videos_dir = project_root / "data/videos/scaled"
    clear_dir(scaled_videos_dir)
    scale_videos(original_videos_dir, scaled_videos_dir)

    frame_dir = project_root / "data/videos/frames"
    flow_dir = project_root / "data/videos/flows"
    clear_dir(frame_dir)
    clear_dir(flow_dir)


    videos = get_video_paths_from_dir(project_root / "data/videos/scaled")
    print("Saving frames and flows.")
    for video in tqdm(videos):
        video_name = Path(video).stem
        output_frames_dir = frame_dir / video_name
        output_flows_dir = flow_dir / video_name
        output_frames_dir.mkdir()
        output_flows_dir.mkdir()

        frames = load_and_save_frames(video, output_frames_dir)
        _ = load_and_save_flows(frames, output_flows_dir)


if __name__ == "__main__":
    main()
