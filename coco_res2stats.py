import argparse
import json
import time

import numpy as np

from pycocotools.coco import COCO


class MyCOCO(COCO):
    def __init__(self, annotation_file=None):
        super().__init__(annotation_file)

    def loadRes(self, resFile):
        res = COCO()

        print("Loading and preparing results...")
        tic = time.time()
        if type(resFile) == str or (PYTHON_VERSION == 2 and type(resFile) == unicode):
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, "results in not an array of objects"
        annsImgIds = [ann["image_id"] for ann in anns]

        if "bbox" in anns[0] and not anns[0]["bbox"] == []:
            for id, ann in enumerate(anns):
                bb = ann["bbox"]
                x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                if not "segmentation" in ann:
                    ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann["area"] = bb[2] * bb[3]
                ann["id"] = id + 1
                ann["iscrowd"] = 0
        elif "segmentation" in anns[0]:
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann["area"] = maskUtils.area(ann["segmentation"])
                if not "bbox" in ann:
                    ann["bbox"] = maskUtils.toBbox(ann["segmentation"])
                ann["id"] = id + 1
                ann["iscrowd"] = 0
        elif "keypoints" in anns[0]:
            for id, ann in enumerate(anns):
                s = ann["keypoints"]
                x = s[0::3]
                y = s[1::3]
                x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann["area"] = (x1 - x0) * (y1 - y0)
                ann["id"] = id + 1
                ann["bbox"] = [x0, y0, x1 - x0, y1 - y0]
        print("DONE (t={:0.2f}s)".format(time.time() - tic))

        res.dataset["annotations"] = anns
        res.createIndex()
        return res


def convert(input_file, width, height, output_filename, category_id_range, image_id_offset):
    coco = MyCOCO()
    res = coco.loadRes(input_file)

    with open(output_filename, mode="w") as f:
        print("# VTMBMS Block Statistics", file=f)
        print(f"# Sequence size: [{width}x{height}]", file=f)

        if category_id_range:
            print(f"# Block Statistic Type: Category; Integer; {category_id_range}", file=f)
        else:
            print("# Block Statistic Type: Category; Integer; [0, 90]", file=f)

        print("# Block Statistic Type: Score; Integer; [0, 100]", file=f)
        for poc, anns in res.imgToAnns.items():
            poc += image_id_offset
            if poc < 0:
                print(f"POC must be greater or equal than 0. [poc={poc}]")
                exit(1)

            for ann in anns:
                bl = int(ann["bbox"][0])
                bt = int(ann["bbox"][1])
                bw = int(ann["bbox"][2])
                bh = int(ann["bbox"][3])
                category_id = ann["category_id"]
                score = int(ann["score"] * 100.0)
                print(
                    f"BlockStat: POC {poc} @({bl:>4},{bt:>4}) [{bw:>4}x{bh:>4}] Category={category_id}",
                    file=f,
                )
                print(
                    f"BlockStat: POC {poc} @({bl:>4},{bt:>4}) [{bw:>4}x{bh:>4}] Score={score}",
                    file=f,
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="COCO result file")
    parser.add_argument("sequence_width", help="sequence width")
    parser.add_argument("sequence_height", help="sequence height")
    parser.add_argument("output_filename", help="output VTMBMS file name")
    parser.add_argument("--category_id_range", nargs=2, type=int, help="Range of COCO categories")
    parser.add_argument("--image_id_offset", type=int, help="This offset is added to image_id in COCO result file")

    args = parser.parse_args()
    convert(args.input_file, args.sequence_width, args.sequence_height, args.output_filename, args.category_id_range, args.image_id_offset)


if __name__ == "__main__":
    main()
