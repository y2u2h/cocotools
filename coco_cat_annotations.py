import argparse
import collections as cl
import json
import sys
from pathlib import Path

from pycocotools.coco import COCO


def make(output_file, input_files, root_dirs):
    cat_id_to_name = dict()
    cat_name_to_id = dict()
    img_name_to_wh = dict()
    img_name_to_id = dict()
    coco_cats = []
    coco_imgs = []
    coco_anns = []
    img_id = 0
    ann_id = 0

    if input_files is None:
        print("please specify two or more inputs")
        sys.exit(1)
    if len(input_files) < 2:
        print(f"please specify two or more inputs[{len(input_files)}]")
        sys.exit(1)
    if output_file in input_files:
        print("duplicate file names")
        sys.exit(1)
    if root_dirs is not None:
        if len(input_files) != len(root_dirs):
            print(f"number of inputs[{len(input_files)}] and root_dirs[{len(root_dirs)}] are different")
            sys.exit(1)

    cocos = {f: COCO(f) for f in input_files}
    dirs = {f: d for f, d in zip(input_files, root_dirs)} if root_dirs is not None else None

    for f, coco in cocos.items():
        print(f"parse {f}...")
        remap_image_id = dict()
        cats = coco.loadCats(coco.getCatIds())
        imgs = coco.loadImgs(coco.getImgIds())
        anns = coco.loadAnns(coco.getAnnIds())

        for c in cats:
            if c["id"] in cat_id_to_name.keys():
                if (c["name"], c["supercategory"]) != cat_id_to_name[c["id"]]:
                    print(
                        f"mismatch category[{c['id']}] {(c['name'], c['supercategory'])} vs {cat_id_to_name[c['id']]}"
                    )
                    sys.exit(1)
            else:
                if c["name"] in cat_name_to_id.keys():
                    print(f"mismatch category id[{c['name']}] {c['id']} vs {cat_name_to_id[c['name']]}")
                    sys.exit(1)
                else:
                    cat_id_to_name[c["id"]] = (c["name"], c["supercategory"])
                    cat_name_to_id[c["name"]] = c["id"]
                    coco_cats.append(c)

        for i in imgs:
            file_name = Path(dirs[f]).joinpath(i["file_name"]) if dirs is not None else i["file_name"]
            if file_name in img_name_to_wh.keys():
                if (i["width"], i["height"]) != img_name_to_wh[file_name]:
                    print(
                        f"mismatch image size[{file_name}] {(i['width'], i['height'])} vs {img_name_to_wh[file_name]}"
                    )
                    sys.exit(1)
                else:
                    remap_image_id[i["id"]] = img_name_to_id[file_name]
            else:
                img_name_to_wh[file_name] = (i["width"], i["height"])
                img_name_to_id[file_name] = img_id
                remap_image_id[i["id"]] = img_id
                i["id"] = img_id
                i["file_name"] = str(file_name)
                coco_imgs.append(i)
                img_id += 1

        for a in anns:
            a["image_id"] = remap_image_id[a["image_id"]]
            a["id"] = ann_id
            coco_anns.append(a)
            ann_id += 1

    # create json
    coco_dict = cl.OrderedDict()
    if "info" in cocos[input_files[0]].dataset.keys():
        coco_dict["info"] = cocos[input_files[0]].dataset["info"]
    if "licenses" in cocos[input_files[0]].dataset.keys():
        coco_dict["licenses"] = cocos[input_files[0]].dataset["licenses"]
    coco_dict["images"] = coco_imgs
    coco_dict["annotations"] = coco_anns
    coco_dict["categories"] = coco_cats

    with open(output_file, mode="w") as f:
        json.dump(coco_dict, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="output annotation file name")
    parser.add_argument("-inputs", "--inputs", help="input annotation files", nargs="+")
    parser.add_argument(
        "-root_dirs", "--root_dirs", help="root dirs corresponding to input annotation files", nargs="+"
    )
    args = parser.parse_args()
    make(args.output, args.inputs, args.root_dirs)


if __name__ == "__main__":
    main()
