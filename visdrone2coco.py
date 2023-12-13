import argparse
import collections as cl
import csv
import glob
import json
import os

import imagesize
from tqdm import tqdm

VISDRONE_DET_FORMAT = {
    "bbox_left": 0,
    "bbox_top": 1,
    "bbox_width": 2,
    "bbox_height": 3,
    "score": 4,
    "object_category": 5,
    "truncation": 6,
    "occlusion": 7,
}

VISDRONE_VID_FORMAT = {
    "frame_index": 0,
    "target_id": 1,
    "bbox_left": 2,
    "bbox_top": 3,
    "bbox_width": 4,
    "bbox_height": 5,
    "score": 6,
    "object_category": 7,
    "truncation": 8,
    "occlusion": 9,
}

VISDRONE_CATEGORY = {
    0: "ignore",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}

VISDRONE_TO_COCO = {
    "ignore": {},
    "pedestrian": {"id": 0, "name": "pedestrian", "supercategory": "person"},
    "people": {"id": 1, "name": "people", "supercategory": "person"},
    "bicycle": {"id": 2, "name": "bicycle", "supercategory": "bicycle"},
    "car": {"id": 3, "name": "car", "supercategory": "car"},
    "van": {"id": 4, "name": "van", "supercategory": "truck"},
    "truck": {"id": 5, "name": "truck", "supercategory": "truck"},
    "tricycle": {"id": 6, "name": "tricycle", "supercategory": "motor"},
    "awning-tricycle": {"id": 7, "name": "awning-tricycle", "supercategory": "motor"},
    "bus": {"id": 8, "name": "bus", "supercategory": "bus"},
    "motor": {"id": 9, "name": "motor", "supercategory": "motor"},
    "others": {},
}


def convert_det(dataset_dir, annotation_dir, output, remap, scale):
    # images
    coco_images = []
    coco_image_ids = {}
    for iid, img in enumerate(tqdm(sorted(glob.glob(dataset_dir + "/*.jpg")))):
        width, height = imagesize.get(img)
        filename = os.path.basename(img)
        basename = os.path.splitext(filename)[0]

        coco_image_ids[basename] = iid
        coco_images.append(
            {
                "id": iid,
                "height": int((float(height) + (scale / 2.0)) / scale),
                "width": int((float(width) + (scale / 2.0)) / scale),
                "file_name": filename,
            }
        )

    # annotations
    coco_annotations = []
    aid = 0
    for txt in tqdm(sorted(glob.glob(annotation_dir + "/*.txt"))):
        basename = os.path.splitext(os.path.basename(txt))[0]

        with open(txt, mode="r") as f:
            for row in csv.reader(f):
                cid = int(row[VISDRONE_DET_FORMAT["object_category"]])
                is_append = False if remap else True

                if remap:
                    cat = VISDRONE_TO_COCO[VISDRONE_CATEGORY[cid]]
                    if cat:
                        cid = cat["id"]
                        is_append = True

                if is_append:
                    bl = float(row[VISDRONE_DET_FORMAT["bbox_left"]]) / scale
                    bt = float(row[VISDRONE_DET_FORMAT["bbox_top"]]) / scale
                    bw = float(row[VISDRONE_DET_FORMAT["bbox_width"]]) / scale
                    bh = float(row[VISDRONE_DET_FORMAT["bbox_height"]]) / scale
                    area = bw * bh

                    if area > 0:
                        coco_annotations.append(
                            {
                                "id": aid,
                                "image_id": coco_image_ids[basename],
                                "category_id": cid,
                                "bbox": [bl, bt, bw, bh],
                                "area": bw * bh,
                                "iscrowd": 0,
                                "ignore": 0,
                            }
                        )
                        aid += 1

    # categories
    coco_categories = []
    if remap:
        for vcat in VISDRONE_CATEGORY.values():
            cat = VISDRONE_TO_COCO[vcat]
            if cat:
                coco_categories.append(
                    {
                        "id": cat["id"],
                        "name": cat["name"],
                        "supercategory": cat["supercategory"],
                    }
                )
    else:
        for cid, cat in VISDRONE_CATEGORY.items():
            coco_categories.append({"id": cid, "name": cat, "supercategory": "none"})

    # create json
    coco_dict = cl.OrderedDict()
    coco_dict["images"] = coco_images
    coco_dict["annotations"] = coco_annotations
    coco_dict["categories"] = coco_categories

    with open(output, mode="w") as f:
        json.dump(coco_dict, f, indent=2)


def convert_vid(sequence_dir, annotation_dir, output, remap, scale):
    # images
    coco_images = []
    coco_image_ids = {}
    iid = 0
    for seq_dir in tqdm(sorted(glob.glob(sequence_dir + "/uav*/"))):
        seq_dir = os.path.dirname(seq_dir)
        prefix = os.path.basename(seq_dir)

        for img in sorted(glob.glob(seq_dir + "/*.jpg")):
            width, height = imagesize.get(img)
            filename = os.path.basename(img)
            basename = os.path.splitext(filename)[0]

            coco_image_ids[prefix + "_" + basename] = iid
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
    coco_annotations = []
    aid = 0
    for txt in tqdm(sorted(glob.glob(annotation_dir + "/*.txt"))):
        basename = os.path.splitext(os.path.basename(txt))[0]

        with open(txt, mode="r") as f:
            for row in csv.reader(f):
                cid = int(row[VISDRONE_VID_FORMAT["object_category"]])
                is_append = False if remap else True

                if remap:
                    cat = VISDRONE_TO_COCO[VISDRONE_CATEGORY[cid]]
                    if cat:
                        cid = cat["id"]
                        is_append = True

                if is_append:
                    bl = float(row[VISDRONE_VID_FORMAT["bbox_left"]]) / scale
                    bt = float(row[VISDRONE_VID_FORMAT["bbox_top"]]) / scale
                    bw = float(row[VISDRONE_VID_FORMAT["bbox_width"]]) / scale
                    bh = float(row[VISDRONE_VID_FORMAT["bbox_height"]]) / scale
                    fr = int(row[VISDRONE_VID_FORMAT["frame_index"]])
                    area = bw * bh

                    if area > 0:
                        key = basename + "_" + f"{fr:07}"
                        coco_annotations.append(
                            {
                                "id": aid,
                                "image_id": coco_image_ids[key],
                                "category_id": cid,
                                "bbox": [bl, bt, bw, bh],
                                "area": bw * bh,
                                "iscrowd": 0,
                                "ignore": 0,
                            }
                        )
                        aid += 1

    # categories
    coco_categories = []
    if remap:
        for vcat in VISDRONE_CATEGORY.values():
            cat = VISDRONE_TO_COCO[vcat]
            if cat:
                coco_categories.append(
                    {
                        "id": cat["id"],
                        "name": cat["name"],
                        "supercategory": cat["supercategory"],
                    }
                )
    else:
        for cid, cat in VISDRONE_CATEGORY.items():
            coco_categories.append({"id": cid, "name": cat, "supercategory": "none"})

    # create json
    coco_dict = cl.OrderedDict()
    coco_dict["images"] = coco_images
    coco_dict["annotations"] = coco_annotations
    coco_dict["categories"] = coco_categories

    with open(output, mode="w") as f:
        json.dump(coco_dict, f, indent=2)


def convert(dataset_type, dataset_dir, annotation_dir, output, remap, scale):
    if dataset_type == "DET":
        convert_det(dataset_dir, annotation_dir, output, remap, scale)
    elif dataset_type == "VID":
        convert_vid(dataset_dir, annotation_dir, output, remap, scale)
    else:
        print(f"Unsupported dataset_type [{dataset_type}].")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_type", help="dataset type [DET, VID]")
    parser.add_argument("dataset_dir", help="dataset directory")
    parser.add_argument("annotation_dir", help="annotation directory")
    parser.add_argument("output", help="output annotation file")
    parser.add_argument("-remap", "--remap", action="store_false")
    parser.add_argument("-scale", "--scale", type=float, default=1.0)

    args = parser.parse_args()
    convert(args.dataset_type, args.dataset_dir, args.annotation_dir, args.output, args.remap, args.scale)


if __name__ == "__main__":
    main()
