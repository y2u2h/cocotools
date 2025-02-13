import argparse
import collections as cl
import itertools
import json


def convert(input_file, width, height, frames, output_file, start_frame, vtmbms_file):
    annotation = json.load(open(input_file, "r"))
    annotation = annotation[0]["annotations"][0]
    result_count = annotation["result_count"]
    result = annotation["result"]

    vtmf = None
    if vtmbms_file:
        vtmf = open(vtmbms_file, "w")
        if vtmf:
            print("# VTMBMS Block Statistics", file=vtmf)
            print(f"# Sequence size: [{width}x{height}]", file=vtmf)
            print(f"# Block Statistic Type: TRACK_ID; Integer; [1, {result_count}]", file=vtmf)

    coco_annotations = [[] for _ in range(frames)]
    aid = 1
    track_id = 1
    for ann in result:
        ann = ann["value"]["sequence"]

        for seq in ann:
            image_id = seq["frame"] - start_frame
            x = seq["x"]
            y = seq["y"]
            w = seq["width"]
            h = seq["height"]

            bl = int(x / 100.0 * 3840.0 + 0.5)
            bt = int(y / 100.0 * 2160.0 + 0.5)
            bw = int(w / 100.0 * 3840.0 + 0.5)
            bh = int(h / 100.0 * 2160.0 + 0.5)

            bl = 0 if bl < 0 else width - 1 if bl >= width else bl
            bt = 0 if bt < 0 else height - 1 if bt >= height else bt
            area = bw * bh

            if area > 0:
                coco_annotations[image_id].append(
                    {
                        "id": aid,
                        "image_id": image_id,
                        "bbox": [bl, bt, bw, bh],
                        "area": area,
                        "iscrowd": 0,
                        "ignore": 0,
                        "track_id": track_id,
                    }
                )
                aid += 1
        track_id += 1

    if vtmf:
        for anns in coco_annotations:
            for ann in anns:
                poc = ann["image_id"]
                bbox = ann["bbox"]
                bl = bbox[0]
                bt = bbox[1]
                bw = bbox[2]
                bh = bbox[3]
                track_id = ann["track_id"]
                print(
                    f"BlockStat: POC {poc} @({bl:>4},{bt:>4}) [{bw:>4}x{bh:>4}] TRACK_ID={track_id}",
                    file=vtmf,
                )
        vtmf.close()

    coco_dict = cl.OrderedDict()
    coco_dict = list(itertools.chain.from_iterable(coco_annotations))

    with open(output_file, mode="w") as f:
        json.dump(coco_dict, f, indent=2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", type=str, help="input json file (label-studio format)")
    parser.add_argument("video_width", type=int, help="video width")
    parser.add_argument("video_height", type=int, help="video width")
    parser.add_argument("frames", type=int, help="number of frames")
    parser.add_argument("output_file", type=str, help="output annotation file (COCO format)")
    parser.add_argument("-start_frame", "--start_frame", type=int, default=1)
    parser.add_argument("-vtmbms_file", "--vtmbms_file", type=str, default="")

    args = parser.parse_args()
    convert(
        args.input_file,
        args.video_width,
        args.video_height,
        args.frames,
        args.output_file,
        args.start_frame,
        args.vtmbms_file,
    )


if __name__ == "__main__":
    main()
