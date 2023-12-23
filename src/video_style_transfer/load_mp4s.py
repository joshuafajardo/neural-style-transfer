import os
import torch
import torchvision
import glob

from tqdm import tqdm


FRAME_DIMENSIONS = (640, 360)


def get_mp4s_from_dir(input_dir):
    """Returns a list of mp4s from the directory."""
    return glob.glob(input_dir + "/*.mp4")

def load_and_save_frames(video_path, output_dir):
    """
    Loads and saves the frames of the video as pngs in the
    output_dir.
    """
    frames, _, _ = torchvision.io.read_video(video_path)
    frames = torchvision.transforms.functional.resize(frames, FRAME_DIMENSIONS)
    for i in range(frames.shape[0]):
        output_path = output_dir + "/" + str(i) + ".png"
        torchvision.io.write_png(frames[i], output_path)
    return frames


def get_frames_from_dir(frame_dir):
    """Returns a list of tensors from the frames in the directory."""
    paths = glob.glob(frame_dir + "/*.png")
    paths.sort()
    frames = []
    for path in paths:
        frames.append(torchvision.io.read_image(path))
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
        output_path = output_dir + "/" + str(i) + ".png"
        torchvision.io.write_png(flow, output_path)
    return flows


def main():
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

    mp4s = get_mp4s_from_dir("data/videos/mp4s")
    for mp4 in tqdm(mp4s):
        video_name = mp4.split("/")[-1].split(".")[0]
        output_frames_dir = "data/videos/frames" + video_name
        output_flows_dir = "data/videos/flows" + video_name
        os.mkdir(output_frames_dir)
        os.mkdir(output_flows_dir)

        frames = load_and_save_frames(mp4, output_frames_dir)
        _ = load_and_save_flows(frames, output_flows_dir)
    

if __name__ == "__main__":
    main()
