import argparse

from pycocotools.coco import COCO


def create_histogram(annotation, categories, images, model_input_width, model_input_height, csv):
    cocoGt = COCO(annotation)

    catids = cocoGt.getCatIds()
    if categories:
        catids = []
        cats = {v["name"]: v["id"] for k, v in cocoGt.loadCats(catids)}
        for cat in categories:
            catids.append(cats[cat])

    imgids = cocoGt.getImgIds()
    if images:
        imgids = []
        imgs = {v["file_name"]: v["id"] for k, v in cocoGt.loadImgs(imgids)}
        for img in images:
            imgids.extend([v for k, v in imgs.items() if img in k])

    first = True
    hist_0 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    hist_1 = [0, 0, 0]

    for k, v in cocoGt.anns.items():
        if csv is True and first is True:
            first = False
            print(
                "annotation_id,image_id,image_name,image_size,image_width,image_height,image_area,category_id,category_name,bbox_left,bbox_top,bbox_width,bbox_height,bbox_area,bbox_area_ratio",
                end="",
            )
            if model_input_width != 0 and model_input_height != 0:
                print(
                    ",model_width,model_height,mag_x,mag_y,mag,model_bbox_left,model_bbox_top,model_bbox_width,model_bbox_height,model_bbox_area,model_bbox_area_ratio"
                )
            else:
                print()

        aid = v["id"]

        img = cocoGt.imgs[v["image_id"]]
        iid = img["id"]
        iname = img["file_name"]
        iwidth = img["width"]
        iheight = img["height"]
        iarea = iwidth * iheight

        cat = cocoGt.cats[v["category_id"]]
        cid = cat["id"]
        cname = cat["name"]

        ann = cocoGt.anns[aid]
        bbox = ann["bbox"]
        bbox_left = bbox[0]
        bbox_top = bbox[1]
        bbox_width = bbox[2]
        bbox_height = bbox[3]
        bbox_area = ann["area"]
        bbox_area_ratio = float(bbox_area) / float(iarea) * 100.0

        if csv is True:
            print(
                f"{aid},{iid},{iname},{iwidth}x{iheight},{iwidth},{iheight},{iarea},{cid},{cname},{bbox_left},{bbox_top},{bbox_width},{bbox_height},{bbox_area},{bbox_area_ratio}",
                end="",
            )

        if model_input_width != 0 and model_input_height != 0:
            mag_x = float(model_input_width) / float(iwidth)
            mag_y = float(model_input_height) / float(iheight)
            mag = min(mag_x, mag_y)
            model_input_width = float(iwidth) * mag
            model_input_height = float(iheight) * mag
            model_bbox_left = float(bbox_left) * mag
            model_bbox_top = float(bbox_top) * mag
            model_bbox_width = float(bbox_width) * mag
            model_bbox_height = float(bbox_height) * mag
            model_bbox_area = model_bbox_width * model_bbox_height
            model_bbox_area_ratio = model_bbox_area / (model_input_width * model_input_height) * 100.0

            if csv is True:
                print(
                    f",{model_input_width:.02f},{model_input_height:.02f},{mag_x:.04f},{mag_y:.04f},{mag:.04f},{model_bbox_left:.02f},{model_bbox_top:.02f},{model_bbox_width:.02f},{model_bbox_height:.02f},{model_bbox_area:.02f},{model_bbox_area_ratio:.02f}"
                )
        else:
            if csv is True:
                print()

        if bbox_area > (512**2):
            hist_0[0] += 1
        elif bbox_area <= (512**2) and bbox_area > (256**2):
            hist_0[1] += 1
        elif bbox_area <= (256**2) and bbox_area > (128**2):
            hist_0[2] += 1
        elif bbox_area <= (128**2) and bbox_area > (96**2):
            hist_0[3] += 1
        elif bbox_area <= (96**2) and bbox_area > (64**2):
            hist_0[4] += 1
        elif bbox_area <= (64**2) and bbox_area > (32**2):
            hist_0[5] += 1
        elif bbox_area <= (32**2) and bbox_area > (16**2):
            hist_0[6] += 1
        elif bbox_area <= (16**2) and bbox_area > (8**2):
            hist_0[7] += 1
        else:
            hist_0[8] += 1

        if bbox_area > (96**2):
            hist_1[0] += 1
        elif bbox_area <= (96**2) and bbox_area > (32**2):
            hist_1[1] += 1
        else:
            hist_1[2] += 1

    if csv is False:
        print(f"8x8          : {hist_0[8]}")
        print(f"16x16        : {hist_0[7]}")
        print(f"32x32        : {hist_0[6]}")
        print(f"64x64        : {hist_0[5]}")
        print(f"96x96        : {hist_0[4]}")
        print(f"128x128      : {hist_0[3]}")
        print(f"256x256      : {hist_0[2]}")
        print(f"512x512      : {hist_0[1]}")
        print(f"over 512x512 : {hist_0[0]}")
        print(f"small        : {hist_1[2]}")
        print(f"medium       : {hist_1[1]}")
        print(f"large        : {hist_1[0]}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("annotation", help="input annotation file")
    parser.add_argument("-categories", "--categories", nargs="+")
    parser.add_argument("-images", "--images", nargs="+")
    parser.add_argument("-model_input_width", "--model_input_width", default=0)
    parser.add_argument("-model_input_height", "--model_input_height", default=0)
    parser.add_argument("-csv", "--csv", action="store_true")

    args = parser.parse_args()
    create_histogram(
        args.annotation,
        args.categories,
        args.images,
        args.model_input_width,
        args.model_input_height,
        args.csv,
    )


if __name__ == "__main__":
    main()
