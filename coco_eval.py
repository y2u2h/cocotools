import argparse

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate(annotation, result, categories, images):
    cocoGt = COCO(annotation)
    cocoDt = cocoGt.loadRes(result)
    E = COCOeval(cocoGt, cocoDt, "bbox")

    if categories:
        cats = {v["name"]: v["id"] for k, v in cocoGt.cats.items()}
        catids = []
        for cat in categories:
            catids.append(cats[cat])
        E.params.catIds = catids

    if images:
        imgs = {v["file_name"]: v["id"] for k, v in cocoGt.imgs.items()}
        imgids = []
        for img in images:
            imgids.extend([v for k, v in imgs.items() if img in k])
        E.params.imgIds = imgids

    E.evaluate()
    E.accumulate()
    E.summarize()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("annotation", help="input annotation file")
    parser.add_argument("result", help="result json file")
    parser.add_argument("-categories", "--categories", nargs="+")
    parser.add_argument("-images", "--images", nargs="+")

    args = parser.parse_args()
    evaluate(args.annotation, args.result, args.categories, args.images)


if __name__ == "__main__":
    main()
