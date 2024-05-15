import argparse
import glob
import os
import subprocess

from tqdm import tqdm


def split(dataset_dir, output_dir):
    for vid, video in enumerate(tqdm(sorted(glob.glob(dataset_dir + "/*.yuv")))):
        basename = os.path.splitext(os.path.basename(video))[0]
        st = basename.split("_")

        key = st[0]
        video_size = st[1]
        frame_rate = st[2]
        dst_dir = f"{output_dir}/{key}"

        if "RaceHorsesC" in basename:
            basename = basename.replace("RaceHorsesC", "RaceHorses")
        elif "RaceHorsesD" in basename:
            basename = basename.replace("RaceHorsesD", "RaceHorses")

        output = f"{dst_dir}/{basename}_%03d.png"

        cmd = f"ffmpeg -f rawvideo -pixel_format yuv420p -video_size {video_size} -framerate {frame_rate} -i {video} -pixel_format rgb24 -start_number 0 {output}"
        print(cmd)

        os.makedirs(dst_dir, exist_ok=True)
        subprocess.run(cmd, shell=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_dir", help="dataset directory")
    parser.add_argument("output_dir", help="output directory")

    args = parser.parse_args()
    split(args.dataset_dir, args.output_dir)


if __name__ == "__main__":
    main()
