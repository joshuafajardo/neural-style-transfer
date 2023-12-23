import torch
import torchvision

import glob

FRAME_DIMENSIONS = (640, 360)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def save_mp4s_as_frames(input_path, output_dir):
    mp4s = glob.glob(input_path + "/*.mp4")
    for mp4 in mp4s:
        video, _, _ = torchvision.io.read_video(mp4)
        video = torchvision.transforms.functional.resize(video, FRAME_DIMENSIONS)
        # Save each frame as a png in the output_dir
        for i in range(video.shape[0]):
            output_path = output_dir + "/" + str(i) + ".png"
            torchvision.io.write_png(video[i], output_path)


def get_frames_from_dir(frame_dir):
    """Returns a list of tensors from the frames in the directory."""
    paths = glob.glob(frame_dir + "/*.png")
    paths.sort()
    frames = []
    for path in paths:
        frames.append(torchvision.io.read_image(path))
    return frames


def generate_video_flows(frames):
    frames = frames
    start_frames = frames[:-1]
    end_frames = frames[1:]
    raft_model = torchvision.models.optical_flow.raft_large(pretrained=True,
                                                            progress=False)
    raft_model = raft_model
    raft_model = raft_model.eval()
    return raft_model(start_frames, end_frames)


def save_flows_in_dir(flows, output_dir):
    for i, flow in enumerate(flows):
        output_path = output_dir + "/" + str(i) + ".png"
        torchvision.io.write_png(flow, output_path)


def main():
    input_path = "data/training/videos"
    output_dir = "data/frames"
    save_mp4s_as_frames(input_path, output_dir)
    frames = get_frames_from_dir(output_dir)

    flows = generate_video_flows(frames)

    output_dir = "data/flows"
    save_flows_in_dir(flows, output_dir)
    
if __name__ == "__main__":
    main()
