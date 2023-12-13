import argparse
import collections as cl

from pycocotools.coco import COCO


def check(input_file, cid_range):
    cocoGt = COCO(input_file)

    images = {an["id"]: an["file_name"] for an in cocoGt.dataset["images"]}
    ids = []
    for i, an in enumerate(cocoGt.dataset["annotations"]):
        if an is None:
            print(f"annotation {i} is None.")
            if an["image_id"] in images.keys():
                print(f"{images[an['image_id']]}")

        aid = an["id"]
        ids.append(aid)
        if aid is None:
            print(f"annotation {i}: annotation_id is None.")
            print(f"{an}")
            if an["image_id"] in images.keys():
                print(f"{images[an['image_id']]}")

        if aid < 0:
            print(f"annotation {i}: annotation_id[{aid}] out of range !")
            print(f"{an}")
            if an["image_id"] in images.keys():
                print(f"{images[an['image_id']]}")

        iid = an["image_id"]
        if iid is None:
            print(f"annotation {i}: image_id is None.")
            print(f"{an}")
            if an["image_id"] in images.keys():
                print(f"{images[an['image_id']]}")

        if iid < 0:
            print(f"annotation {i}: image_id[{iid}] out of range !")
            print(f"{an}")
            if an["image_id"] in images.keys():
                print(f"{images[an['image_id']]}")

        cid = an["category_id"]
        if cid is None:
            print(f"annotation {i}: category_id is None.")
            print(f"{an}")
            if an["image_id"] in images.keys():
                print(f"{images[an['image_id']]}")

        if cid < cid_range[0] and cid > cid_range[1]:
            print(f"annotation {i}: category_id[{cid}] out of range !")
            print(f"{an}")
            if an["image_id"] in images.keys():
                print(f"{images[an['image_id']]}")

        bbox = an["bbox"]
        if bbox is None:
            print(f"annotation {i}: bbox is None.")
            print(f"{an}")
            if an["image_id"] in images.keys():
                print(f"{images[an['image_id']]}")

        if len(bbox) != 4:
            print(f"annotation {i}: bbox size[{len(bbox)}] mismatch !")
            print(f"{an}")
            if an["image_id"] in images.keys():
                print(f"{images[an['image_id']]}")

        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= 0 or bbox[3] <= 0:
            print(f"annotation {i}: bbox value[{bbox}] out of range !")
            print(f"{an}")
            if an["image_id"] in images.keys():
                print(f"{images[an['image_id']]}")

        area = an["area"]
        if area is None:
            print(f"annotation {i}: area is None.")
            print(f"{an}")
            if an["image_id"] in images.keys():
                print(f"{images[an['image_id']]}")

        if area <= 0:
            print(f"annotation {i}: area[{area}] out of range !")
            print(f"{an}")
            if an["image_id"] in images.keys():
                print(f"{images[an['image_id']]}")

    if len(ids) != len(set(ids)):
        print("duplicated annotation_id !")
        print(sorted([k for k, v in cl.Counter(ids).items() if v > 1], key=ids.index))

    print(f"image count: {len(cocoGt.dataset['images'])}")
    print(f"category count: {len(cocoGt.dataset['categories'])}")
    print(f"annotation count: {len(cocoGt.dataset['annotations'])}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="input annotation file")
    parser.add_argument("-category_id_range", "--category_id_range", nargs=2, default=[0, 9])
    args = parser.parse_args()

    check(args.input, args.category_id_range)


if __name__ == "__main__":
    main()
