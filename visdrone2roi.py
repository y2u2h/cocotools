import argparse
import collections as cl
import json
import pathlib

import numpy as np

from pycocotools.coco import COCO


def detect(annotation, output_dir, result, categories, images, block_size, margin, maxdet, score_thr, visdrone_video):
    cocoGt = COCO(annotation)

    iids = cocoGt.getImgIds()
    cids = cocoGt.getCatIds()
    if categories:
        cids = cocoGt.getCatIds(catNms=categories)

    if images:
        imgs = {v["file_name"]: v["id"] for k, v in cocoGt.imgs.items()}
        imgids = []
        for img in images:
            imgids.extend([v for k, v in imgs.items() if img in k])
        iids = imgids

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    datas = cl.OrderedDict()

    cocoDt = None
    if result:
        cocoDt = cocoGt.loadRes(result)
    else:
        maxdet = -1
        score_thr = -1

    for iid in iids:
        if cocoDt is None:
            im = cocoGt.loadImgs(ids=iid)[0]
            gt = cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=iid, catIds=cids))
        else:
            im = cocoDt.loadImgs(ids=iid)[0]
            dt = cocoDt.loadAnns(cocoDt.getAnnIds(imgIds=iid, catIds=cids))
            dtind = np.argsort([-d["score"] for d in dt], kind="mergesort")
            dtind = dtind[0:maxdet] if maxdet > 0 else dtind
            dt = [dt[i] for i in dtind if dt[i]["score"] >= score_thr]

        image_w = im["width"]
        image_h = im["height"]
        hblocks = int((image_w + (block_size - 1)) / block_size)
        vblocks = int((image_h + (block_size - 1)) / block_size)

        data = cl.OrderedDict()
        data["file_name"] = im["file_name"]
        data["image_width"] = image_w
        data["image_height"] = image_h
        data["block_size"] = block_size
        data["margin"] = margin
        data["maxdet"] = maxdet
        data["score_thr"] = score_thr
        data["num_of_hblocks"] = hblocks
        data["num_of_vblocks"] = vblocks

        tmp_data = cl.OrderedDict()
        annotations = gt if cocoDt is None else dt
        for target in annotations:
            x, y, w, h = target["bbox"]
            category_id = target["category_id"]
            score = -1
            area = target["area"]

            if cocoDt is not None:
                x = int(x + 0.5)
                y = int(y + 0.5)
                w = int(w + 0.5)
                h = int(h + 0.5)
                score = round(target["score"], 4)
                area = int(area + 0.5)

            bbox = [
                x,
                y,
                x + w - 1 if (x + w - 1) < image_w else image_w - 1,
                y + h - 1 if (y + h - 1) < image_h else image_h - 1,
            ]
            bbox_ext = [
                x - margin if (x - margin) >= 0 else 0,
                y - margin if (y - margin) >= 0 else 0,
                x + w + margin - 1 if (x + w + margin - 1) < image_w else image_w - 1,
                y + h + margin - 1 if (y + h + margin - 1) < image_h else image_h - 1,
            ]

            ovl_region = [int(b / block_size) for b in bbox]
            nbr_region = [int(b / block_size) for b in bbox_ext]
            nbr_hblocks = nbr_region[2] - nbr_region[0] + 1
            nbr_vblocks = nbr_region[3] - nbr_region[1] + 1

            for tmp_by in range(nbr_vblocks):
                for tmp_bx in range(nbr_hblocks):
                    bx = nbr_region[0] + tmp_bx
                    by = nbr_region[1] + tmp_by
                    bn = by * hblocks + bx
                    corner_xl = bx * block_size
                    corner_yt = by * block_size
                    hoverlap = (bx >= ovl_region[0]) and (bx <= ovl_region[2])
                    voverlap = (by >= ovl_region[1]) and (by <= ovl_region[3])

                    if tmp_data.get(bn) is None:
                        tmp_data[bn] = {
                            "block": bn,
                            "position": [bx, by],
                            "corner": [corner_xl, corner_yt],
                            "num_of_overlapped_datas": 0,
                            "overlapped_datas": [],
                            "num_of_near_datas": 0,
                            "near_datas": [],
                        }

                    if hoverlap and voverlap:
                        # overlapped block
                        if bx == ovl_region[0]:
                            roi_x = bbox[0] - corner_xl
                            roi_w = block_size - roi_x if roi_x + w > block_size else w
                        elif bx == ovl_region[2]:
                            roi_x = 0
                            roi_w = bbox[0] + w - corner_xl
                        else:
                            roi_x = 0
                            roi_w = block_size

                        if by == ovl_region[1]:
                            roi_y = bbox[1] - corner_yt
                            roi_h = block_size - roi_y if roi_y + h > block_size else h
                        elif by == ovl_region[3]:
                            roi_y = 0
                            roi_h = bbox[1] + h - corner_yt
                        else:
                            roi_y = 0
                            roi_h = block_size

                        tmp_data[bn]["num_of_overlapped_datas"] += 1
                        tmp_data[bn]["overlapped_datas"].append(
                            {
                                "category_id": category_id,
                                "bbox": [x, y, w, h],
                                "area": area,
                                "score": score,
                                "overlap_bbox": [roi_x, roi_y, roi_w, roi_h],
                                "overlap_area": roi_w * roi_h,
                                "overlap_ratio": round(float(roi_w) * float(roi_h) / float(area), 4),
                            }
                        )
                    else:
                        # near block
                        tmp_data[bn]["num_of_near_datas"] += 1
                        tmp_data[bn]["near_datas"].append(
                            {
                                "category_id": category_id,
                                "bbox": [x, y, w, h],
                                "area": area,
                                "score": score,
                            }
                        )

        data["num_of_regions"] = len(tmp_data.keys())
        data["regions"] = [v for k, v in sorted(tmp_data.items())]

        if visdrone_video:
            stem = pathlib.Path(im["file_name"].replace("/", "_")).stem.split("_")
            poc = int(stem[-1]) - 1
            del stem[-1]

            basename = "_".join(stem)
            if datas.get(basename) is None:
                datas[basename] = cl.OrderedDict()
                datas[basename]["video_width"] = image_w
                datas[basename]["video_height"] = image_h
                datas[basename]["block_size"] = block_size
                datas[basename]["margin"] = margin
                datas[basename]["maxdet"] = maxdet
                datas[basename]["score_thr"] = score_thr
                datas[basename]["num_of_hblocks"] = hblocks
                datas[basename]["num_of_vblocks"] = vblocks
                datas[basename]["num_of_frames"] = -1
                datas[basename]["frames"] = cl.OrderedDict()

            data["poc"] = poc
            data.move_to_end("poc", False)
            datas[basename]["frames"][poc] = data
        else:
            output_file = output_dir.joinpath(im["file_name"].replace("/", "_")).with_suffix(".json")
            with open(output_file, mode="w") as f:
                json.dump(data, f, indent=2)

    if visdrone_video:
        for basename, data in datas.items():
            data["num_of_frames"] = len(data["frames"].keys())
            data["frames"] = [v for k, v in sorted(data["frames"].items())]

            # dump json file
            if cocoDt is None:
                output_file = output_dir.joinpath(basename + "_gt").with_suffix(".json")
            else:
                output_file = output_dir.joinpath(basename + "_dt").with_suffix(".json")

            with open(output_file, mode="w") as f:
                json.dump(data, f, indent=2)

            # dump vtmbmsstas file for yuview
            if cocoDt is None:
                output_file = output_dir.joinpath(basename + "_gt").with_suffix(".vtmbmsstats")
            else:
                output_file = output_dir.joinpath(basename + "_dt").with_suffix(".vtmbmsstats")

            with open(output_file, mode="w") as f:
                print("# VTMBMS Block Statistics", file=f)
                print("# Sequence size: [{data['video_width']}x{data['video_height']}]", file=f)
                print("# Block Statistic Type: ROI; Integer; [0, 1]", file=f)

                for frame in data["frames"]:
                    for region in frame["regions"]:
                        xl = region["corner"][0]
                        yt = region["corner"][1]

                        if region["num_of_overlapped_datas"] > 0:
                            # overlapped block
                            print(
                                f"BlockStat: POC {frame['poc']} @({xl:>4},{yt:>4}) [{block_size:>4}x{block_size:>4}] ROI=0",
                                file=f,
                            )
                        else:
                            # near block
                            print(
                                f"BlockStat: POC {frame['poc']} @({xl:>4},{yt:>4}) [{block_size:>4}x{block_size:>4}] ROI=1",
                                file=f,
                            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("annotation", help="input annotation file (COCO format)")
    parser.add_argument("output_dir", help="output directory")
    parser.add_argument("-result", "--result", help="input result file (COCO format) [none]")
    parser.add_argument("-categories", "--categories", nargs="+", help="search categories [all]")
    parser.add_argument("-images", "--images", nargs="+", help="search images [all]")
    parser.add_argument("-block_size", "--block_size", type=int, default=64, help="roi block size [64]")
    parser.add_argument("-margin", "--margin", type=int, default=16, help="margin between bbox and block [16]")
    parser.add_argument("-maxdet", "--maxdet", type=int, default=100, help="max detections per image [100]")
    parser.add_argument("-score_thr", "--score_thr", type=float, default=0.3, help="bbox score threshold [0.3]")
    parser.add_argument(
        "-visdrone_video", "--visdrone_video", action="store_true", help="output stats for Visdrone VID dataset"
    )

    args = parser.parse_args()
    detect(
        args.annotation,
        args.output_dir,
        args.result,
        args.categories,
        args.images,
        args.block_size,
        args.margin,
        args.maxdet,
        args.score_thr,
        args.visdrone_video,
    )


if __name__ == "__main__":
    main()
