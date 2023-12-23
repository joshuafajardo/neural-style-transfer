import glob
import os

import torch
import torchvision

from pathlib import Path
from tqdm import tqdm


FRAME_DIMENSIONS = (640, 360)


def get_mp4s_from_dir(input_dir):
    """Returns a list of mp4s from the directory."""
    return input_dir.glob("*.mp4")

def load_and_save_frames(video_path, output_dir):
    """
    Loads and saves the frames of the video as pngs in the
    output_dir.
    """
    frames, _, _ = torchvision.io.read_video(str(video_path))
    frames = torchvision.transforms.functional.resize(frames, FRAME_DIMENSIONS)
    for i in range(frames.shape[0]):
        output_path = output_dir + "/" + str(i) + ".png"
        torchvision.io.write_png(frames[i], output_path)
    return frames


def load_and_save_flows(frames, output_dir):
    """Returns a list of optical flows between the frames."""
    frames = frames
    start_frames = frames[:-1]
    end_frames = frames[1:]
    raft_model = torchvision.models.optical_flow.raft_large(pretrained=True,
                                                            progress=False)
    raft_model = raft_model
    raft_model = raft_model.eval()

    flows = raft_model(start_frames, end_frames)
    for i, flow in enumerate(flows):
        output_path = output_dir / i / ".png"
        torchvision.io.write_png(flow, output_path)
    return flows


def main():
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = Path(__file__).resolve().parents[2]

    mp4s = get_mp4s_from_dir(project_root / "data/videos/mp4s")
    for mp4 in tqdm(mp4s):
        video_name = Path(mp4).stem
        output_frames_dir = project_root / "data/videos/frames" / video_name
        output_flows_dir = project_root / "data/videos/flows" / video_name
        os.mkdir(output_frames_dir)
        os.mkdir(output_flows_dir)

        frames = load_and_save_frames(mp4, output_frames_dir)
        _ = load_and_save_flows(frames, output_flows_dir)


if __name__ == "__main__":
    main()
