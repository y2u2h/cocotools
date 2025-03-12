import argparse
import collections as cl
import csv
import glob
import json
import pathlib

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

SFUHW_OBJECT_FORMAT = {
    "class_id": 0,
    "bbox_center_x": 1,
    "bbox_center_y": 2,
    "bbox_width": 3,
    "bbox_height": 4,
}

SFUHW_TRACK_FORMAT = {
    "class_id": 0,
    "object_id": 1,
    "bbox_center_x": 2,
    "bbox_center_y": 3,
    "bbox_width": 4,
    "bbox_height": 5,
}

SFUHW_OBJECT_SEQUENCES = {
    "Traffic": ("ClassA", "Traffic_2560x1600_30_crop", 2560, 1600, 150),
    "PeopleOnStreet": ("ClassA", "PeopleOnStreet_2560x1600_30_crop", 2560, 1600, 150),
    "BQTerrace": ("ClassB", "BQTerrace_1920x1080_60", 1920, 1080, 600),
    "BasketballDrive": ("ClassB", "BasketballDrive_1920x1080_50", 1920, 1080, 500),
    "Cactus": ("ClassB", "Cactus_1920x1080_50", 1920, 1080, 500),
    "Kimono": ("ClassB", "Kimono1_1920x1080_24", 1920, 1080, 240),
    "ParkScene": ("ClassB", "ParkScene_1920x1080_24", 1920, 1080, 240),
    "BQMall": ("ClassC", "BQMall_832x480_60", 832, 480, 600),
    "BasketballDrill": ("ClassC", "BasketballDrill_832x480_50", 832, 480, 500),
    "PartyScene": ("ClassC", "PartyScene_832x480_50", 832, 480, 500),
    "RaceHorsesC": ("ClassC", "RaceHorses_832x480_30", 832, 480, 300),
    "BQSquare": ("ClassD", "BQSquare_416x240_60", 416, 240, 600),
    "BasketballPass": ("ClassD", "BasketballPass_416x240_50", 416, 240, 500),
    "BlowingBubbles": ("ClassD", "BlowingBubbles_416x240_50", 416, 240, 500),
    "RaceHorsesD": ("ClassD", "RaceHorses_416x240_30", 416, 240, 300),
    "KristenAndSara": ("ClassE", "KristenAndSara_1280x720_60", 1280, 720, 600),
    "Johnny": ("ClassE", "Johnny_1280x720_60", 1280, 720, 600),
    "FourPeople": ("ClassE", "FourPeople_1280x720_60", 1280, 720, 600),
}

SFUHW_TRACK_SEQUENCES = {
    "BasketballDrive": ("ClassB", "BasketballDrive_1920x1080_50", 1920, 1080, 500),
    "Cactus": ("ClassB", "Cactus_1920x1080_50", 1920, 1080, 500),
    "Kimono": ("ClassB", "Kimono1_1920x1080_24", 1920, 1080, 240),
    "ParkScene": ("ClassB", "ParkScene_1920x1080_24", 1920, 1080, 240),
    "BasketballDrill": ("ClassC", "BasketballDrill_832x480_50", 832, 480, 500),
    "PartyScene": ("ClassC", "PartyScene_832x480_50", 832, 480, 500),
    "RaceHorsesC": ("ClassC", "RaceHorses_832x480_30", 832, 480, 300),
    "BasketballPass": ("ClassD", "BasketballPass_416x240_50", 416, 240, 500),
    "BlowingBubbles": ("ClassD", "BlowingBubbles_416x240_50", 416, 240, 500),
    "RaceHorsesD": ("ClassD", "RaceHorses_416x240_30", 416, 240, 300),
    "KristenAndSara": ("ClassE", "KristenAndSara_1280x720_60", 1280, 720, 600),
    "Johnny": ("ClassE", "Johnny_1280x720_60", 1280, 720, 600),
    "FourPeople": ("ClassE", "FourPeople_1280x720_60", 1280, 720, 600),
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


def convert_all(annotation_dir, output, sequence_dir, track, scale):
    sequences = SFUHW_TRACK_SEQUENCES if track else SFUHW_OBJECT_SEQUENCES
    fmt = SFUHW_TRACK_FORMAT if track else SFUHW_OBJECT_FORMAT

    # images
    coco_images = []
    coco_image_ids = {}

    image_id = 0
    for key, (cls, prefix, width, height, frames) in sequences.keys():
        seq_path = pathlib.Path(sequence_dir + "/" + key)

        if not seq_path.exists() and key == "Kimono":
            print(f"Replace {key} -> Kimono1")
            seq_path = pathlib.Path(sequence_dir + "/" + "Kimono1")

        for img in sorted(glob.glob(str(seq_path) + "/*.png")):
            img = pathlib.Path(img)
            coco_image_ids[img.stem] = image_id
            coco_images.append(
                {
                    "id": image_id,
                    "height": int((float(height) + (scale / 2.0)) / scale),
                    "width": int((float(width) + (scale / 2.0)) / scale),
                    "file_name": seq_path.name + "/" + img.name,
                }
            )
            image_id += 1

    # annotations
    coco_annotations = []
    annotation_id = 1
    for key, (cls, prefix, width, height, frames) in tqdm(sequences.items()):
        txtlist_name = annotation_dir + f"/{cls}/{key}/{prefix}_seq_*.txt"
        txtlist = sorted(glob.glob(txtlist_name))
        if len(txtlist) == 0 and (key == "RaceHorsesC" or key == "RaceHorsesD"):
            print(f"Replace {key} -> RaceHorses")
            txtlist = sorted(glob.glob(annotation_dir + f"/{cls}/RaceHorces/{prefix}_seq_*.txt"))

        for txt in txtlist:
            txt = pathlib.Path(txt)
            parts = txt.stem.partition("_seq_")
            image_key = parts[0] + "_" + parts[2]

            if image_key in coco_image_ids.keys():
                with open(txt, mode="r") as f:
                    for row in csv.reader(f, delimiter=" ", skipinitialspace=True):
                        if len(row) != len(fmt):
                            print(f"data format error: txt={txt} row={row} fmt={fmt}")
                            exit(1)

                        category_id = SFUHW_TO_COCO_ID[SFUHW_CATEGORIES[int(row[fmt["class_id"]])]]

                        normalized_bbox_center_x = float(row[fmt["bbox_center_x"]])
                        normalized_bbox_center_y = float(row[fmt["bbox_center_y"]])
                        normalized_bbox_w = float(row[fmt["bbox_width"]])
                        normalized_bbox_h = float(row[fmt["bbox_height"]])

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
                                    "id": annotation_id,
                                    "image_id": coco_image_ids[image_key],
                                    "category_id": category_id,
                                    "bbox": [bl, bt, bw, bh],
                                    "area": area,
                                    "iscrowd": 0,
                                    "ignore": 0,
                                }
                            )
                            annotation_id += 1

                            if track:
                                coco_annotations[-1]["track_id"] = int(row[fmt["object_id"]])
            else:
                print(f"image_key={image_key} is not in coco_image_ids")
                exit(1)

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

    # output file
    coco_dict = cl.OrderedDict()
    coco_dict["images"] = coco_images
    coco_dict["annotations"] = coco_annotations
    coco_dict["categories"] = coco_categories
    with open(output, mode="w") as f:
        json.dump(coco_dict, f, indent=2)


def convert_separate(annotation_dir, output, track, scale, vtmbms):
    sequences = SFUHW_TRACK_SEQUENCES if track else SFUHW_OBJECT_SEQUENCES
    fmt = SFUHW_TRACK_FORMAT if track else SFUHW_OBJECT_FORMAT

    # images
    coco_images = cl.OrderedDict()
    coco_image_ids = {}

    for key, (cls, prefix, width, height, frames) in sequences.items():
        coco_images[key] = []
        for image_id in range(frames):
            img = pathlib.Path(prefix + f"_{image_id:03d}.png")
            coco_image_ids[img.stem] = image_id
            coco_images[key].append(
                {
                    "id": image_id,
                    "height": int((float(height) + (scale / 2.0)) / scale),
                    "width": int((float(width) + (scale / 2.0)) / scale),
                    "file_name": img.name,  # dummy file
                }
            )

    # annotations
    coco_annotations = cl.OrderedDict()
    for key, (cls, prefix, width, height, frames) in tqdm(sequences.items()):
        coco_annotations[key] = []

        vtmbmsstats = None
        if vtmbms:
            outdir = pathlib.Path(output)
            outdir.mkdir(parents=True, exist_ok=True)
            vtmbmsstats = open(str(outdir) + "/" + key + ".vtmbmsstats", "w")
            print("# VTMBMS Block Statistics", file=vtmbmsstats)
            print(f"# Sequence size: [{width}x{height}]", file=vtmbmsstats)
            print(f"# Block Statistic Type: CATEGORY_ID; Integer; [{MIN_ID}, {MAX_ID}]", file=vtmbmsstats)

        txtlist_name = annotation_dir + f"/{cls}/{key}/{prefix}_seq_*.txt"
        txtlist = sorted(glob.glob(txtlist_name))
        if len(txtlist) == 0 and (key == "RaceHorsesC" or key == "RaceHorsesD"):
            print(f"Replace {key} -> RaceHorses")
            txtlist = sorted(glob.glob(annotation_dir + f"/{cls}/RaceHorces/{prefix}_seq_*.txt"))

        annotation_id = 1
        for txt in txtlist:
            txt = pathlib.Path(txt)
            parts = txt.stem.partition("_seq_")
            image_key = parts[0] + "_" + parts[2]
            frame = parts[2]

            if image_key in coco_image_ids.keys():
                with open(txt, mode="r") as f:
                    for row in csv.reader(f, delimiter=" ", skipinitialspace=True):
                        if len(row) != len(fmt):
                            print(f"data format error: txt={txt} row={row} fmt={fmt}")
                            exit(1)

                        category_id = SFUHW_TO_COCO_ID[SFUHW_CATEGORIES[int(row[fmt["class_id"]])]]

                        normalized_bbox_center_x = float(row[fmt["bbox_center_x"]])
                        normalized_bbox_center_y = float(row[fmt["bbox_center_y"]])
                        normalized_bbox_w = float(row[fmt["bbox_width"]])
                        normalized_bbox_h = float(row[fmt["bbox_height"]])

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
                            coco_annotations[key].append(
                                {
                                    "id": annotation_id,
                                    "image_id": coco_image_ids[image_key],
                                    "category_id": category_id,
                                    "bbox": [bl, bt, bw, bh],
                                    "area": area,
                                    "iscrowd": 0,
                                    "ignore": 0,
                                }
                            )
                            annotation_id += 1

                            if track:
                                track_id = int(row[fmt["object_id"]])
                                coco_annotations[key][-1]["track_id"] = track_id

                            if vtmbmsstats:
                                bl = int(bl)
                                bt = int(bt)
                                bw = int(bw)
                                bh = int(bh)
                                print(
                                    f"BlockStat: POC {frame} @({bl:>4},{bt:>4}) [{bw:>4}x{bh:>4}] CATEGORY_ID={category_id}",
                                    file=vtmbmsstats,
                                )
            else:
                print(f"image_key={image_key} is not found in coco_image_ids")
                exit(1)

        if vtmbmsstats is not None:
            vtmbmsstats.close()

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

    # output file
    for (img_key, imgs), (anno_key, annos) in zip(coco_images.items(), coco_annotations.items()):
        if img_key == anno_key:
            coco_dict = cl.OrderedDict()
            coco_dict["images"] = imgs
            coco_dict["annotations"] = annos
            coco_dict["categories"] = coco_categories

            outdir = pathlib.Path(output)
            outdir.mkdir(parents=True, exist_ok=True)
            with open(outdir.joinpath(img_key + ".json"), mode="w") as f:
                json.dump(coco_dict, f, indent=2)
        else:
            print(f"img_key={img_key} is not equal to anno_key={anno_key}")
            exit(1)


def convert(annotation_dir, output, sequence_dir, track, separate, scale, vtmbms):
    if not separate and not sequence_dir:
        print("sequence_dir must be set when separate == False")
        exit(1)
    elif not separate:
        convert_all(annotation_dir, output, sequence_dir, track, scale)
    elif separate:
        convert_separate(annotation_dir, output, track, scale, vtmbms)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("annotation_dir", help="annotation directory")
    parser.add_argument("output", help="output annotation file/directory")
    parser.add_argument("-sequence_dir", type=str, default="", help="sequence directory")
    parser.add_argument("-track", "--track", action="store_true")
    parser.add_argument("-separate", "--separate", action="store_true")
    parser.add_argument("-scale", "--scale", type=float, default=1.0)
    parser.add_argument("-vtmbms", "--vtmbms", action="store_true")

    args = parser.parse_args()
    convert(
        args.annotation_dir,
        args.output,
        args.sequence_dir,
        args.track,
        args.separate,
        args.scale,
        args.vtmbms,
    )


if __name__ == "__main__":
    main()
