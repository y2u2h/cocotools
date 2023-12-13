import argparse

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class MyCOCOeval(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType="bbox"):
        super().__init__(cocoGt, cocoDt, iouType)

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        """

        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100):
            p = self.params
            iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"
            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1]) if iouThr is None else "{:0.2f}".format(iouThr)
            )

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["precision"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval["recall"]
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((24,))
            stats[0] = _summarize(1, maxDets=self.params.maxDets[2])
            stats[1] = _summarize(1, iouThr=0.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=0.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng="small", maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng="large", maxDets=self.params.maxDets[2])
            stats[6] = _summarize(1, iouThr=0.5, areaRng="small", maxDets=self.params.maxDets[2])
            stats[7] = _summarize(1, iouThr=0.5, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[8] = _summarize(1, iouThr=0.5, areaRng="large", maxDets=self.params.maxDets[2])
            stats[9] = _summarize(1, iouThr=0.75, areaRng="small", maxDets=self.params.maxDets[2])
            stats[10] = _summarize(1, iouThr=0.75, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[11] = _summarize(1, iouThr=0.75, areaRng="large", maxDets=self.params.maxDets[2])
            stats[12] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[13] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[14] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[15] = _summarize(0, areaRng="small", maxDets=self.params.maxDets[2])
            stats[16] = _summarize(0, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[17] = _summarize(0, areaRng="large", maxDets=self.params.maxDets[2])
            stats[18] = _summarize(0, iouThr=0.5, areaRng="small", maxDets=self.params.maxDets[2])
            stats[19] = _summarize(0, iouThr=0.5, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[20] = _summarize(0, iouThr=0.5, areaRng="large", maxDets=self.params.maxDets[2])
            stats[21] = _summarize(0, iouThr=0.75, areaRng="small", maxDets=self.params.maxDets[2])
            stats[22] = _summarize(0, iouThr=0.75, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[23] = _summarize(0, iouThr=0.75, areaRng="large", maxDets=self.params.maxDets[2])
            return stats

        if not self.eval:
            raise Exception("Please run accumulate() first")
        summarize = _summarizeDets
        self.stats = summarize()


def evaluate(annotation, result, categories, images, maxdet):
    cocoGt = COCO(annotation)
    cocoDt = cocoGt.loadRes(result)
    E = MyCOCOeval(cocoGt, cocoDt, "bbox")

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

    if maxdet > 100:
        E.params.maxDets[2] = maxdet

    E.evaluate()
    E.accumulate()
    E.summarize()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("annotation", help="input annotation file")
    parser.add_argument("result", help="result json file")
    parser.add_argument("-categories", "--categories", nargs="+")
    parser.add_argument("-images", "--images", nargs="+")
    parser.add_argument("-maxdet", "--maxdet", type=int, default=100)

    args = parser.parse_args()
    evaluate(args.annotation, args.result, args.categories, args.images, args.maxdet)


if __name__ == "__main__":
    main()
