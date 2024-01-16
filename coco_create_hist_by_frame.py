import argparse
import collections as cl
import pathlib
import sys

from pycocotools.coco import COCO


def create_histogram(annotation, image, category):
    cocoGt = COCO(annotation)
    cid = cocoGt.getCatIds(catNms=category)[0]

    tmp = {}
    images = {}
    for k, v in sorted(cocoGt.imgs.items()):
        if image in v["file_name"]:
            tmp[k] = pathlib.Path(v["file_name"]).stem
            images[k] = v["file_name"]

    if len(tmp.values()) != len(set(tmp.values())):
        print("duplicate key exists !")
        print(sorted([k for k, v in cl.Counter(tmp.values()).items() if v > 1]))
        sys.exit()

    hist = {}
    for v in sorted(images.values()):
        hist[v] = [0] * 3

    for ann in cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=images.keys())):
        if ann["category_id"] is cid:
            key = images[ann["image_id"]]
            area = ann["area"]

            if area > (96**2):
                hist[key][2] += 1
            elif area <= (96**2) and area > (32**2):
                hist[key][1] += 1
            else:
                hist[key][0] += 1

    summary = [0, 0, 0]
    for v in hist.values():
        summary[0] += v[0]
        summary[1] += v[1]
        summary[2] += v[2]

    print(f"small  : {summary[0]}")
    print(f"medium : {summary[1]}")
    print(f"large  : {summary[2]}")
    print("image,small,medium,large")
    for k, v in hist.items():
        print(f"{k},{v[0]},{v[1]},{v[2]}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("annotation", help="input annotation file")
    parser.add_argument("image", help="search pattern of a image or sequence")
    parser.add_argument("category", help="search pattern of a category")

    args = parser.parse_args()
    create_histogram(args.annotation, args.image, args.category)


if __name__ == "__main__":
    main()
