import argparse
import datetime
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class MyCOCOeval(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType="segm"):
        super().__init__(cocoGt, cocoDt, iouType)

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        """
        perform evaluation for single category and image
        :return: dict (single image results)
        """
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g["ignore"] or (g["area"] <= aRng[0] or g["area"] > aRng[1]):
                g["_ignore"] = 1
            else:
                g["_ignore"] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o["iscrowd"]) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g["_ignore"] for g in gt])
        dtIg = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]["id"]
                    gtm[tind, m] = d["id"]
        # set unmatched detections outside of area range to ignore
        a = np.array([d["area"] <= aRng[0] or d["area"] > aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
        # store results for given image and category
        return {
            "image_id": imgId,
            "category_id": catId,
            "aRng": aRng,
            "maxDet": maxDet,
            "dtIds": [d["id"] for d in dt],
            "gtIds": [g["id"] for g in gt],
            "dtMatches": dtm,
            "gtMatches": gtm,
            "dtScores": [d["score"] for d in dt],
            "gtIgnore": gtIg,
            "dtIgnore": dtIg,
        }

    def accumulate(self, p=None):
        """
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        """
        print("Accumulating evaluation results...")
        tic = time.time()
        if not self.evalImgs:
            print("Please run evaluate() first")
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision = -np.ones((T, R, K, A, M))  # -1 for the precision of absent categories
        precision_unsorted = -np.ones((T, R, K, A, M))  # -1 for the precision of absent categories
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)

        # get max length of dtScores for each condition
        len_raw = -1
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e["dtScores"][0:maxDet] for e in E])
                    if len(dtScores) > len_raw:
                        len_raw = len(dtScores)

        gtcnts = -np.ones((K, A, M))
        dtcnts = -np.ones((K, A, M))
        tp_raw = -np.ones((T, len_raw, K, A, M))
        fp_raw = -np.ones((T, len_raw, K, A, M))
        tp_sum_raw = -np.ones((T, len_raw, K, A, M))
        fp_sum_raw = -np.ones((T, len_raw, K, A, M))
        precision_raw = -np.ones((T, len_raw, K, A, M))
        precision_raw_unsorted = -np.ones((T, len_raw, K, A, M))
        recall_raw = -np.ones((T, len_raw, K, A, M))
        scores_raw = -np.ones((T, len_raw, K, A, M))

        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e["dtScores"][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind="mergesort")
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate([e["dtMatches"][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate([e["dtIgnore"][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e["gtIgnore"] for e in E])
                    npig = np.count_nonzero(gtIg == 0)

                    gtcnts[k, a, m] = npig
                    dtcnts[k, a, m] = len(dtScoresSorted)

                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float64)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float64)
                    for t, (tp, fp, tp_, fp_) in enumerate(zip(tp_sum, fp_sum, tps, fps)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R,))
                        q_unsorted = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist()
                        q = q.tolist()
                        pr_unsorted = pr.copy()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side="left")
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                q_unsorted[ri] = pr_unsorted[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass

                        precision[t, :, k, a, m] = np.array(q)
                        precision_unsorted[t, :, k, a, m] = np.array(q_unsorted)
                        scores[t, :, k, a, m] = np.array(ss)

                        padding = [-1.0] * (len_raw - len(dtScoresSorted))
                        tp_raw[t, :, k, a, m] = np.concatenate([tp_, np.array(padding)])
                        fp_raw[t, :, k, a, m] = np.concatenate([fp_, np.array(padding)])
                        tp_sum_raw[t, :, k, a, m] = np.concatenate([tp, np.array(padding)])
                        fp_sum_raw[t, :, k, a, m] = np.concatenate([fp, np.array(padding)])
                        precision_raw[t, :, k, a, m] = np.array(pr + padding)
                        precision_raw_unsorted[t, :, k, a, m] = np.array(pr_unsorted + padding)
                        recall_raw[t, :, k, a, m] = np.concatenate([rc, np.array(padding)])
                        scores_raw[t, :, k, a, m] = np.concatenate([dtScoresSorted, np.array(padding)])

        self.eval = {
            "params": p,
            "counts": [T, R, K, A, M],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "precision_unsorted": precision_unsorted,
            "recall": recall,
            "scores": scores,
            "gtcnt": gtcnts,
            "dtcnt": dtcnts,
            "tp": tp_raw,
            "fp": fp_raw,
            "tp_sum": tp_sum_raw,
            "fp_sum": fp_sum_raw,
            "precision_raw": precision_raw,
            "precision_raw_unsorted": precision_raw_unsorted,
            "recall_raw": recall_raw,
            "scores_raw": scores_raw,
        }
        toc = time.time()
        print("DONE (t={:0.2f}s).".format(toc - tic))

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        """

        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100):
            p = self.params
            iStr = " {:<18} {} @[ IoU={:<9} | area={:>12s} | maxDets={:>3d} ] = {:0.3f}"
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
            areas = len(self.params.areaRngLbl)
            stats = np.zeros((3 * areas * 2,))
            offset = 0

            for i, thr in enumerate([None, 0.5, 0.75]):
                for j, lbl in enumerate(self.params.areaRngLbl):
                    stats[areas * i + j + offset] = _summarize(
                        1, iouThr=thr, areaRng=lbl, maxDets=self.params.maxDets[2]
                    )

            offset = 3 * areas
            for i, thr in enumerate([None, 0.5, 0.75]):
                for j, lbl in enumerate(self.params.areaRngLbl):
                    stats[areas * i + j + offset] = _summarize(
                        0, iouThr=thr, areaRng=lbl, maxDets=self.params.maxDets[2]
                    )
            return stats

        if not self.eval:
            raise Exception("Please run accumulate() first")
        summarize = _summarizeDets
        self.stats = summarize()

    def prcurve(self, output_dir):
        p = self.params
        precision = self.eval["precision"]
        precision_unsorted = self.eval["precision_unsorted"]
        score = self.eval["scores"]

        tind = np.where(0.5 == p.iouThrs)[0]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == 100]

        # dimension of precision: [TxRxKxAxM]
        for kind, cid in enumerate(p.catIds):
            cat = self.cocoGt.loadCats(ids=[cid])[0]["name"]

            for aind, areaRngLbl in enumerate(p.areaRngLbl):
                pr = precision[tind, :, kind, aind, mind][0]
                pr_unsorted = precision_unsorted[tind, :, kind, aind, mind][0]
                sc = score[tind, :, kind, aind, mind][0]

                df = pd.DataFrame(
                    {
                        "recall": p.recThrs,
                        "precision": pr,
                        "precision_unsorted": pr_unsorted,
                        "score": sc,
                    }
                ).set_index("recall")

                outdir = Path(output_dir)
                outdir.mkdir(parents=True, exist_ok=True)
                df.to_csv(str(outdir.joinpath(f"prcurve_{cat}_{areaRngLbl}.csv")), float_format="%0.8f")

    def prcurve_raw(self, output_dir):
        p = self.params
        gtcnt = self.eval["gtcnt"]
        dtcnt = self.eval["dtcnt"]
        tp = self.eval["tp"]
        fp = self.eval["fp"]
        tp_sum = self.eval["tp_sum"]
        fp_sum = self.eval["fp_sum"]
        precision = self.eval["precision_raw"]
        precision_unsorted = self.eval["precision_raw_unsorted"]
        scores = self.eval["scores_raw"]
        recall = self.eval["recall_raw"]

        tind = np.where(0.5 == p.iouThrs)[0]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == 100]

        space = 0.10
        score_thrs = np.linspace(1.00 - space, 0.00, int(np.round((1.00 - space - 0.0) / space)) + 1, endpoint=True)

        # dimension of precision: [TxRxKxAxM]
        for kind, cid in enumerate(p.catIds):
            cat = self.cocoGt.loadCats(ids=[cid])[0]["name"]

            for aind, areaRngLbl in enumerate(p.areaRngLbl):
                gtc = gtcnt[kind, aind, mind]
                dtc = dtcnt[kind, aind, mind]
                if dtc <= 0:
                    continue

                padding = [-1.0] * (int(dtc[0]) - 1)
                gtc = np.concatenate([gtc, np.array(padding)])
                dtc = np.concatenate([dtc, np.array(padding)])

                t = tp[tind, :, kind, aind, mind][0][0 : int(dtc[0])]
                f = fp[tind, :, kind, aind, mind][0][0 : int(dtc[0])]
                ts = tp_sum[tind, :, kind, aind, mind][0][0 : int(dtc[0])]
                fs = fp_sum[tind, :, kind, aind, mind][0][0 : int(dtc[0])]
                pr = precision[tind, :, kind, aind, mind][0][0 : int(dtc[0])]
                pr_unsorted = precision_unsorted[tind, :, kind, aind, mind][0][0 : int(dtc[0])]
                sc = scores[tind, :, kind, aind, mind][0][0 : int(dtc[0])]
                rc = recall[tind, :, kind, aind, mind][0][0 : int(dtc[0])]

                df = pd.DataFrame(
                    {
                        "gtc": gtc,
                        "dtc": dtc,
                        "tp": t,
                        "fp": f,
                        "tp_sum": ts,
                        "fp_sum": fs,
                        "recall": rc,
                        "precision": pr,
                        "precision_unsorted": pr_unsorted,
                        "score": sc,
                    }
                )

                df[["gtc", "dtc", "tp", "fp", "tp_sum", "fp_sum"]] = df[
                    ["gtc", "dtc", "tp", "fp", "tp_sum", "fp_sum"]
                ].astype(int)

                idxthr = []
                tpnums = []
                fpnums = []
                for thr in score_thrs:
                    idxs = df[df["score"] > thr].index
                    if len(idxs) > 0:
                        idxthr.append(max(idxs))
                    else:
                        idxthr.append(-1)

                    score_bin = df[(df["score"] <= (thr + space)) & (df["score"] > thr)]
                    tpnums.append(len(score_bin[score_bin["tp"] == 1]))
                    fpnums.append(len(score_bin[score_bin["fp"] == 1]))

                df_ = df[["tp_sum", "fp_sum", "recall", "precision", "precision_unsorted", "score"]]
                hist = pd.DataFrame(columns=["tp_sum", "fp_sum", "recall", "precision", "precision_unsorted", "score"])

                for i, idx in enumerate(idxthr):
                    if idx > 0:
                        hist.loc[i] = df_.iloc[idx].values
                    else:
                        hist.loc[i] = [-1, -1, -1.0, -1.0, -1.0, -1.0]

                hist[["tp_sum", "fp_sum"]] = hist[["tp_sum", "fp_sum"]].astype(int)
                hist[["recall", "precision", "precision_unsorted", "score"]] = hist[
                    ["recall", "precision", "precision_unsorted", "score"]
                ].astype(float)

                if not hist.empty:
                    hist = hist[["tp_sum", "fp_sum", "recall", "precision", "precision_unsorted", "score"]]
                    hist["tpnum"] = tpnums
                    hist["fpnum"] = fpnums

                    outdir = Path(output_dir)
                    outdir.mkdir(parents=True, exist_ok=True)
                    hist.to_csv(str(outdir.joinpath(f"hist_raw_{cat}_{areaRngLbl}.csv")), float_format="%0.8f")

                outdir = Path(output_dir)
                outdir.mkdir(parents=True, exist_ok=True)
                df.to_csv(str(outdir.joinpath(f"prcurve_raw_{cat}_{areaRngLbl}.csv")), float_format="%0.8f")


def evaluate(annotation, result, categories, images, areas, maxdet, draw_prcurve, recthr_fine, output_dir):
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

    if areas:
        pre_area = 0
        area_ranges = [[0.0**2, 1e5**2]]
        area_labels = ["all"]
        for area in areas:
            area_ranges.append([float(pre_area) ** 2, float(area) ** 2])
            area_labels.append(f"{area}x{area}")
            pre_area = area
        area_ranges.append([float(pre_area) ** 2, 1e5**2])
        area_labels.append(f"over{pre_area}x{pre_area}")
        E.params.areaRng = area_ranges
        E.params.areaRngLbl = area_labels

    if maxdet > 100:
        E.params.maxDets[2] = maxdet

    if recthr_fine:
        E.params.recThrs = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.001)) + 1, endpoint=True)

    E.evaluate()
    E.accumulate()
    E.summarize()

    if draw_prcurve:
        E.prcurve(output_dir)
        E.prcurve_raw(output_dir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("annotation", help="input annotation file")
    parser.add_argument("result", help="result json file")
    parser.add_argument("-categories", "--categories", nargs="+")
    parser.add_argument("-images", "--images", nargs="+")
    parser.add_argument("-areas", "--areas", nargs="+")
    parser.add_argument("-maxdet", "--maxdet", type=int, default=100)
    parser.add_argument("-draw_prcurve", "--draw_prcurve", action="store_true")
    parser.add_argument("-recthr_fine", "--recthr_fine", action="store_true")
    parser.add_argument("-output_dir", "--output_dir", type=str, default="")

    args = parser.parse_args()
    evaluate(
        args.annotation,
        args.result,
        args.categories,
        args.images,
        args.areas,
        args.maxdet,
        args.draw_prcurve,
        args.recthr_fine,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
