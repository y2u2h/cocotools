import argparse
import collections as cl
import itertools
import json
import pathlib

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
            print(f"# Block Statistic Type: CATEGORY_ID; Integer; [{MIN_ID}, {MAX_ID}]", file=vtmf)
            print(f"# Block Statistic Type: TRACK_ID; Integer; [1, {result_count}]", file=vtmf)

    coco_categories = []
    for category_name, category_id in COCO_CATEGORY_IDS.items():
        coco_categories.append(
            {
                "id": category_id,
                "name": category_name,
                "supercategory": COCO_SUPERCATEGORIES[category_id],
            }
        )

    coco_images = []
    for image_id in range(frames):
        coco_images.append(
            {
                "id": image_id,
                "height": height,
                "width": width,
                "file_name": f"dummy_{image_id:07d}.jpg",
            }
        )

    coco_annotations = [[] for _ in range(frames)]
    aid = 1
    track_id = 1
    for ann in result:
        value = ann["value"]
        sequence = value["sequence"]
        category_name = value["labels"][0]
        category_id = COCO_CATEGORY_IDS[category_name]

        for seq in sequence:
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
                        "category_id": category_id,
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

    output_path = pathlib.Path(output_file)

    coco_dict = cl.OrderedDict()
    coco_dict = list(itertools.chain.from_iterable(coco_annotations))
    with open(str(output_path), mode="w") as f:
        json.dump(coco_dict, f, indent=2)

    coco_dict = cl.OrderedDict()
    coco_dict["images"] = coco_images
    coco_dict["annotations"] = list(itertools.chain.from_iterable(coco_annotations))
    coco_dict["categories"] = coco_categories
    with open(str(f"{output_path.parent}/{output_path.stem}_gt{output_path.suffix}"), mode="w") as f:
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
