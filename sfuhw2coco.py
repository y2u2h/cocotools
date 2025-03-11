import argparse
import collections as cl
import csv
import glob
import json
import os
from pathlib import Path

import imagesize
from tqdm import tqdm

COCO_CLASSES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
    "N/A",
]

COCO_SUPERCATEGORIES = [
    "__background__",
    "person",  # person
    "vehicle",  # bicycle
    "vehicle",  # car
    "vehicle",  # motorcycle
    "vehicle",  # airplane
    "vehicle",  # bus
    "vehicle",  # train
    "vehicle",  # truck
    "vehicle",  # boat
    "outdoor",  # traffic light
    "outdoor",  # fire hydrant
    "N/A",  # N/A
    "outdoor",  # stop sign
    "outdoor",  # parking meter
    "outdoor",  # bench
    "animal",  # bird
    "animal",  # cat
    "animal",  # dog
    "animal",  # horse
    "animal",  # sheep
    "animal",  # cow
    "animal",  # elephant
    "animal",  # bear
    "animal",  # zebra
    "animal",  # giraffe
    "N/A",  # N/A
    "accessory",  # backpack
    "accessory",  # umbrella
    "N/A",  # N/A
    "N/A",  # N/A
    "accessory",  # handbag
    "accessory",  # tie
    "accessory",  # suitcase
    "sports",  # frisbee
    "sports",  # skis
    "sports",  # snowboard
    "sports",  # sports ball
    "sports",  # kite
    "sports",  # baseball bat
    "sports",  # baseball glove
    "sports",  # skateboard
    "sports",  # surfboard
    "sports",  # tennis racket
    "kitchen",  # bottle
    "kitchen",  # N/A
    "kitchen",  # wine glass
    "kitchen",  # cup
    "kitchen",  # fork
    "kitchen",  # knife
    "kitchen",  # spoon
    "kitchen",  # bowl
    "food",  # banana
    "food",  # apple
    "food",  # sandwich
    "food",  # orange
    "food",  # broccoli
    "food",  # carrot
    "food",  # hot dog
    "food",  # pizza
    "food",  # donut
    "food",  # cake
    "furniture",  # chair
    "furniture",  # couch
    "furniture",  # potted plant
    "furniture",  # bed
    "N/A",  # N/A
    "furniture",  # dining table
    "N/A",  # N/A
    "N/A",  # N/A
    "furniture",  # toilet
    "N/A",  # N/A
    "electronic",  # tv
    "electronic",  # laptop
    "electronic",  # mouse
    "electronic",  # remote
    "electronic",  # keyboard
    "electronic",  # cell phone
    "appliance",  # microwave
    "appliance",  # oven
    "appliance",  # toaster
    "appliance",  # sink
    "appliance",  # refrigerator
    "N/A",  # N/A
    "indoor",  # book
    "indoor",  # clock
    "indoor",  # vase
    "indoor",  # scissors
    "indoor",  # teddy bear
    "indoor",  # hair drier
    "indoor",  # toothbrush
    "N/A",  # N/A
]

COCO_CATEGORY_IDS = {
    COCO_CLASSES[i]: i for i, cls in enumerate(COCO_CLASSES) if cls != "N/A" and cls != "__background__"
}

MIN_ID = COCO_CATEGORY_IDS["person"]
MAX_ID = COCO_CATEGORY_IDS["toothbrush"]

SFUHW_FORMAT = {
    "object_id": 0,
    "bbox_center_x": 1,
    "bbox_center_y": 2,
    "bbox_width": 3,
    "bbox_height": 4,
}

SFUHW_SEQUENCES = {
    "Traffic": ("ClassA", "Traffic_2560x1600_30_crop", 2560, 1600),
    "PeopleOnStreet": ("ClassA", "PeopleOnStreet_2560x1600_30_crop", 2560, 1600),
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

SFUHW_CATEGORIES = {
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
    73: "book",
    74: "clock",
    77: "teddy bear",
}

SFUHW_TO_COCO_ID = {
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "bus": 6,
    "truck": 8,
    "boat": 9,
    "bench": 15,
    "horse": 19,
    "backpack": 27,
    "umbrella": 28,
    "handbag": 31,
    "tie": 32,
    "sports ball": 37,
    "cup": 47,
    "chair": 62,
    "potted plant": 64,
    "dining table": 67,
    "laptop": 73,
    "cell phone": 77,
    "book": 84,
    "clock": 85,
    "teddy bear": 88,
}

SFUHW_TO_COCO_SUPERCATEGORY = {
    "person": "person",
    "bicycle": "vehicle",
    "car": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "boat": "vehicle",
    "bench": "outdoor",
    "horse": "animal",
    "backpack": "accessory",
    "umbrella": "accessory",
    "handbag": "accessory",
    "tie": "accessory",
    "sports ball": "sports",
    "cup": "kitchen",
    "chair": "furniture",
    "potted plant": "furniture",
    "dining table": "furniture",
    "laptop": "electronic",
    "cell phone": "electronic",
    "book": "indoor",
    "clock": "indoor",
    "teddy bear": "indoor",
}


def convert(sequence_dir, annotation_dir, output, scale, vtmbms_dir, check_all_data, separate):
    # images
    coco_images_per_seq = cl.OrderedDict()
    coco_images = []
    coco_image_ids = {}
    iid = 0
    for key in tqdm(SFUHW_SEQUENCES.keys()):
        seq_path = Path(sequence_dir + "/" + key)

        if not seq_path.exists() and key == "Kimono":
            print(f"Replace {key} -> Kimono1")
            seq_path = Path(sequence_dir + "/" + "Kimono1")

        seq_dir = str(seq_path)
        prefix = os.path.basename(seq_dir)
        coco_images_per_seq[key] = []
        if separate:
            iid = 0

        for img in sorted(glob.glob(seq_dir + "/*.png")):
            width, height = imagesize.get(img)
            filename = os.path.basename(img)
            basename = os.path.splitext(filename)[0]

            coco_image_ids[basename] = iid
            if separate:
                coco_images_per_seq[key].append(
                    {
                        "id": iid,
                        "height": int((float(height) + (scale / 2.0)) / scale),
                        "width": int((float(width) + (scale / 2.0)) / scale),
                        "file_name": prefix + "/" + filename,
                    }
                )
            else:
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

    coco_annotations_per_seq = cl.OrderedDict()
    coco_annotations = []
    aid = 1
    for key, val in tqdm(SFUHW_SEQUENCES.items()):
        cls = val[0]
        seq = val[1]
        width = float(val[2])
        height = float(val[3])

        coco_annotations_per_seq[key] = []
        if separate:
            aid = 1

        vtmf = None
        if vtmbms_dir:
            vtmf = open(vtmbms_dir + "/" + key + ".vtmbmsstats", "w")
            iwidth = int(width)
            iheight = int(height)
            print("# VTMBMS Block Statistics", file=vtmf)
            print(f"# Sequence size: [{iwidth}x{iheight}]", file=vtmf)
            print("# Block Statistic Type: CATEGORY_ID; Integer; [{MIN_ID}, {MAX_ID}]", file=vtmf)

        txtlist = sorted(glob.glob(annotation_dir + f"/{cls}/{key}/{seq}_seq_*.txt"))
        if len(txtlist) == 0 and (key == "RaceHorsesC" or key == "RaceHorsesD"):
            print(f"Replace {key} -> RaceHorses")
            parent_dir = "RaceHorses"
            txtlist = sorted(glob.glob(annotation_dir + f"/{cls}/{parent_dir}/{seq}_seq_*.txt"))

        for txt in txtlist:
            basename = os.path.splitext(os.path.basename(txt))[0]
            parts = basename.partition("_seq_")
            image_key = parts[0] + "_" + parts[2]
            frame = parts[2]

            if image_key in coco_image_ids.keys():
                with open(txt, mode="r") as f:
                    for row in csv.reader(f, delimiter=" "):
                        cid = int(row[SFUHW_FORMAT["object_id"]])
                        cid = SFUHW_TO_COCO_ID[SFUHW_CATEGORIES[cid]]

                        normalized_bbox_center_x = float(row[SFUHW_FORMAT["bbox_center_x"]])
                        normalized_bbox_center_y = float(row[SFUHW_FORMAT["bbox_center_y"]])
                        normalized_bbox_w = float(row[SFUHW_FORMAT["bbox_width"]])
                        normalized_bbox_h = float(row[SFUHW_FORMAT["bbox_height"]])

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
                            if separate:
                                coco_annotations_per_seq[key].append(
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
                            else:
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
            else:  # image_key not in coco_image_ids.keys()
                if check_all_data:
                    print(f"Error: {image_key} is not in coco_image_ids")
                    exit(1)
                else:
                    print(f"{image_key} is not in coco_image_ids")

        if vtmf is not None:
            vtmf.close()

    # categories
    coco_categories = []
    for category_name, category_id in COCO_CATEGORY_IDS.items():
        coco_categories.append(
            {
                "id": category_id,
                "name": category_name,
                "supercategory": COCO_SUPERCATEGORIES[category_id],
            }
        )

    if separate:
        for key, imgs, annos in zip(
            coco_images_per_seq.keys(), coco_images_per_seq.values(), coco_annotations_per_seq.values()
        ):
            # create json
            coco_dict = cl.OrderedDict()
            coco_dict["images"] = imgs
            coco_dict["annotations"] = annos
            coco_dict["categories"] = coco_categories

            outdir = Path(output)
            outdir.mkdir(parents=True, exist_ok=True)
            with open(outdir.joinpath(key + ".json"), mode="w") as f:
                json.dump(coco_dict, f, indent=2)
    else:
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
    parser.add_argument("-separate_by_video", "--separate_by_video", action="store_true")

    args = parser.parse_args()
    convert(
        args.dataset_dir,
        args.annotation_dir,
        args.output,
        args.scale,
        args.vtmbms_dir,
        args.check_all_data,
        args.separate_by_video,
    )


if __name__ == "__main__":
    main()
