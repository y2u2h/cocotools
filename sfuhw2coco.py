import argparse
import collections as cl
import csv
import glob
import json
import os

import imagesize
from tqdm import tqdm

SFU_HW_FORMAT = {
    "object_id": 0,
    "bbox_center_x": 1,
    "bbox_center_y": 2,
    "bbox_width": 3,
    "bbox_height": 4,
}

SFU_HW_SEQUENCES = {
    "Traffic": ("ClassA", "Traffic_2560x1600_30_crop", 2560, 1600),
    "ParkScene": ("ClassB", "ParkScene_1920x1080_24", 1920, 1080),
    "Cactus": ("ClassB", "Cactus_1920x1080_50", 1920, 1080),
    "BasketballDrive": ("ClassB", "BasketballDrive_1920x1080_50", 1920, 1080),
    "BQTerrace": ("ClassB", "BQTerrace_1920x1080_60", 1920, 1080),
    "Kimono": ("ClassB", "Kimono1_1920x1080_24", 1920, 1080),
    "BasketballDrill": ("ClassC", "BasketballDrill_832x480_50", 832, 480),
    "BQMall": ("ClassC", "BQMall_832x480_60", 832, 480),
    "PartyScene": ("ClassC", "PartyScene_832x480_50", 832, 480),
    "RaceHorsesC": ("ClassC", "RaceHorses_832x480_30", 832, 480),
    "BasketballPass": ("ClassD", "BasketballPass_416x240_50", 416, 240),
    "BQSquare": ("ClassD", "BQSquare_416x240_60", 416, 240),
    "BlowingBubbles": ("ClassD", "BlowingBubbles_416x240_50", 416, 240),
    "RaceHorsesD": ("ClassD", "RaceHorses_416x240_30", 416, 240),
}

SFU_HW_CATEGORIES = {
    0: "person",
    1: "bicycle",
    2: "car",
    5: "bus",
    7: "truck",
    8: "boat",
    13: "bench",
    17: "horse",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    32: "sports ball",
    41: "cup",
    56: "chair",
    58: "potted plant",
    60: "dining table",
    63: "laptop",
    67: "cell phone",
    74: "clock",
    77: "teddy bear",
}


def convert(sequence_dir, annotation_dir, output, scale, check_all_data, vtmbms_dir):
    # images
    coco_images = []
    coco_image_ids = {}
    iid = 0
    for seq_dir in tqdm(sorted(glob.glob(sequence_dir + "/*/"))):
        seq_dir = os.path.dirname(seq_dir)
        prefix = os.path.basename(seq_dir)

        for img in sorted(glob.glob(seq_dir + "/*.png")):
            width, height = imagesize.get(img)
            filename = os.path.basename(img)
            basename = os.path.splitext(filename)[0]

            coco_image_ids[basename] = iid
            coco_images.append(
                {
                    "id": iid,
                    "height": int((float(height) + (scale / 2.0)) / scale),
                    "width": int((float(width) + (scale / 2.0)) / scale),
                    "file_name": prefix + "/" + filename,
                }
            )
            iid += 1

    # annotations
    if vtmbms_dir:
        os.makedirs(vtmbms_dir, exist_ok=True)

    coco_annotations = []
    aid = 0
    for key, val in tqdm(SFU_HW_SEQUENCES.items()):
        cls = val[0]
        seq = val[1]
        width = float(val[2])
        height = float(val[3])

        vtmf = None
        if vtmbms_dir:
            vtmf = open(vtmbms_dir + "/" + key + ".vtmbmsstats", "w")
            iwidth = int(width)
            iheight = int(height)
            print("# VTMBMS Block Statistics", file=vtmf)
            print(f"# Sequence size: [{iwidth}x{iheight}]", file=vtmf)
            print("# Block Statistic Type: CATEGORY; Integer; [0, 77]", file=vtmf)

        txtlist = sorted(glob.glob(annotation_dir + f"/{cls}/{key}/{seq}_seq_*.txt"))
        if len(txtlist) == 0 and (key == "RaceHorsesC" or key == "RaceHorsesD"):
            print(f"Replace {key} -> RaceHorses")
            key = "RaceHorses"
            txtlist = sorted(glob.glob(annotation_dir + f"/{cls}/{key}/{seq}_seq_*.txt"))

        for txt in txtlist:
            basename = os.path.splitext(os.path.basename(txt))[0]
            parts = basename.partition("_seq_")
            image_key = parts[0] + "_" + parts[2]
            frame = parts[2]

            if image_key in coco_image_ids.keys():
                with open(txt, mode="r") as f:
                    for row in csv.reader(f, delimiter=" "):
                        cid = int(row[SFU_HW_FORMAT["object_id"]])
                        normalized_bbox_center_x = float(row[SFU_HW_FORMAT["bbox_center_x"]])
                        normalized_bbox_center_y = float(row[SFU_HW_FORMAT["bbox_center_y"]])
                        normalized_bbox_w = float(row[SFU_HW_FORMAT["bbox_width"]])
                        normalized_bbox_h = float(row[SFU_HW_FORMAT["bbox_height"]])

                        bbox_center_x = width * normalized_bbox_center_x
                        bbox_center_y = height * normalized_bbox_center_y
                        bbox_w = width * normalized_bbox_w
                        bbox_h = height * normalized_bbox_h

                        bl = (bbox_center_x - (bbox_w / 2.0)) / scale
                        bt = (bbox_center_y - (bbox_h / 2.0)) / scale
                        bw = bbox_w / scale
                        bh = bbox_h / scale

                        # clip values
                        bl = 0.0 if bl < 0.0 else width if bl > width else bl
                        bt = 0.0 if bt < 0.0 else height if bt > height else bt
                        bw = width - bl if (bl + bw) > width else bw
                        bh = height - bt if (bt + bh) > height else bh
                        area = bw * bh

                        if area > 0:
                            coco_annotations.append(
                                {
                                    "id": aid,
                                    "image_id": coco_image_ids[image_key],
                                    "category_id": cid,
                                    "bbox": [bl, bt, bw, bh],
                                    "area": area,
                                    "iscrowd": 0,
                                    "ignore": 0,
                                }
                            )
                            aid += 1

                            if vtmf:
                                ibl = int(bl)
                                ibt = int(bt)
                                ibw = int(bw)
                                ibh = int(bh)
                                print(
                                    f"BlockStat: POC {frame} @({ibl:>4},{ibt:>4}) [{ibw:>4}x{ibh:>4}] CATEGORY={cid}",
                                    file=vtmf,
                                )
            else: # image_key not in coco_image_ids.keys()
                if check_all_data:
                    print(f"Error: {image_key} is not in coco_image_ids")
                    exit(1)
                else:
                    print(f"{image_key} is not in coco_image_ids")

        if vtmf is not None:
            vtmf.close()

    # categories
    coco_categories = []
    for cid, cat in SFU_HW_CATEGORIES.items():
        coco_categories.append({"id": cid, "name": cat, "supercategory": "none"})

    # create json
    coco_dict = cl.OrderedDict()
    coco_dict["images"] = coco_images
    coco_dict["annotations"] = coco_annotations
    coco_dict["categories"] = coco_categories

    with open(output, mode="w") as f:
        json.dump(coco_dict, f, indent=2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_dir", help="dataset directory")
    parser.add_argument("annotation_dir", help="annotation directory")
    parser.add_argument("output", help="output annotation file")
    parser.add_argument("-scale", "--scale", type=float, default=1.0)
    parser.add_argument("-check_all_data", "--check_all_data", action="store_true")
    parser.add_argument("-vtmbms_dir", "--vtmbms_dir", default="")

    args = parser.parse_args()
    convert(args.dataset_dir, args.annotation_dir, args.output, args.scale, args.check_all_data, args.vtmbms_dir)


if __name__ == "__main__":
    main()
