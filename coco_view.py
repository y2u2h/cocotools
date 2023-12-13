import argparse
import collections as cl
import glob
import pathlib
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageColor, ImageDraw

from pycocotools.coco import COCO


class Grid:
    def __init__(self, gw, gh, sgw, sgh):
        self.__large_grid_width = gw
        self.__large_grid_height = gh
        self.__large_grid_visible = False
        self.__small_grid_width = sgw
        self.__small_grid_height = sgh
        self.__small_grid_visible = False

    @property
    def large_grid_width(self):
        pass

    @large_grid_width.setter
    def large_grid_width(self, gw):
        self.__large_grid_width = gw

    @property
    def large_grid_height(self):
        pass

    @large_grid_height.setter
    def large_grid_height(self, gh):
        self.__large_grid_height = gh

    @property
    def large_grid_visible(self):
        pass

    @large_grid_visible.setter
    def large_grid_visible(self, gv):
        self.__large_grid_visible = gv

    @property
    def small_grid_width(self):
        pass

    @small_grid_width.setter
    def small_grid_width(self, sgw):
        self.__small_grid_width = sgw

    @property
    def small_grid_height(self):
        pass

    @small_grid_height.setter
    def small_grid_height(self, sgh):
        self.__small_grid_height = sgh

    @property
    def small_grid_visible(self):
        pass

    @small_grid_visible.setter
    def small_grid_visible(self, sgv):
        self.__small_grid_visible = sgv

    def set_ticks(self, axes, width, height):
        # large grid
        axes.set_xticks(np.arange(0, width + 1, self.__large_grid_width), minor=False)
        axes.set_yticks(np.arange(0, height + 1, self.__large_grid_height), minor=False)
        axes.grid(color="cyan", which="major", linestyle="-")
        for tick in axes.xaxis.get_major_ticks():
            tick.gridline.set_visible(False)
        for tick in axes.yaxis.get_major_ticks():
            tick.gridline.set_visible(False)

        # small grid
        small_grid_width = self.__large_grid_width / 4 if self.__small_grid_width is None else self.__small_grid_width
        small_grid_height = (
            self.__large_grid_height / 4 if self.__small_grid_height is None else self.__small_grid_height
        )
        axes.set_xticks(np.arange(0, width + 1, small_grid_width), minor=True)
        axes.set_yticks(np.arange(0, height + 1, small_grid_height), minor=True)
        axes.grid(color="cyan", which="minor", linestyle="-", alpha=0.3)

        for tick in axes.xaxis.get_minor_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
            tick.gridline.set_visible(False)
        for tick in axes.yaxis.get_minor_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
            tick.gridline.set_visible(False)

    def show_large(self, axes):
        for tick in axes.xaxis.get_major_ticks():
            tick.gridline.set_visible(True)
        for tick in axes.yaxis.get_major_ticks():
            tick.gridline.set_visible(True)

    def show_small(self, axes):
        for tick in axes.xaxis.get_minor_ticks():
            tick.gridline.set_visible(True)
        for tick in axes.yaxis.get_minor_ticks():
            tick.gridline.set_visible(True)

    def hide(self, axes):
        for tick in axes.xaxis.get_major_ticks():
            tick.gridline.set_visible(False)
        for tick in axes.yaxis.get_major_ticks():
            tick.gridline.set_visible(False)
        for tick in axes.xaxis.get_minor_ticks():
            tick.gridline.set_visible(False)
        for tick in axes.yaxis.get_minor_ticks():
            tick.gridline.set_visible(False)

    def show_toggle(self, axes):
        if self.__large_grid_visible is False and self.__small_grid_visible is False:
            self.__large_grid_visible = True
        elif self.__large_grid_visible is True and self.__small_grid_visible is False:
            self.__small_grid_visible = True
        else:
            self.__large_grid_visible = False
            self.__small_grid_visible = False

        if self.__large_grid_visible is False and self.__small_grid_visible is False:
            self.hide(axes)
        elif self.__large_grid_visible is True and self.__small_grid_visible is False:
            self.show_large(axes)
        else:
            self.show_large(axes)
            self.show_small(axes)

        plt.draw()


class Zoom:
    scale = 1.0

    def __init__(self, axes):
        xlim = axes.get_xlim()
        ylim = axes.get_ylim()
        self.__xlim_0 = xlim[0]
        self.__xlim_1 = xlim[1]
        self.__ylim_0 = ylim[0]
        self.__ylim_1 = ylim[1]

    @property
    def xlim_0(self):
        return self.__xlim_0

    @property
    def xlim_1(self):
        return self.__xlim_1

    @property
    def ylim_0(self):
        return self.__ylim_0

    @property
    def ylim_1(self):
        return self.__ylim_1

    def zoomin(self, axes, x, y):
        Zoom.scale *= 0.5
        self.update(axes, x, y, 0.5)

    def zoomout(self, axes, x, y):
        next_scale = Zoom.scale * 2.0
        if next_scale >= 1.0:
            self.reset(axes)
        elif next_scale < 1.0:
            Zoom.scale *= 2.0
            self.update(axes, x, y, 2.0)

    def reset(self, axes):
        Zoom.scale = 1.0
        axes.set_xlim(self.__xlim_0, self.__xlim_1)
        axes.set_ylim(self.__ylim_0, self.__ylim_1)
        plt.draw()

    def update(self, axes, x, y, scale):
        xlim = axes.get_xlim()
        ylim = axes.get_ylim()
        xsize = (xlim[1] - xlim[0]) * scale
        ysize = (ylim[1] - ylim[0]) * scale
        relx = (xlim[1] - x) / (xlim[1] - xlim[0])
        rely = (ylim[1] - y) / (ylim[1] - ylim[0])
        axes.set_xlim([x - xsize * (1 - relx), x + xsize * relx])
        axes.set_ylim([y - ysize * (1 - rely), y + ysize * rely])
        plt.draw()


def key_press(event, fig, axes, zoom, grid, images):
    if event.key == "g":
        grid.show_toggle(axes)
    elif event.key == "j" or event.key == "k":
        if event.key == "j":
            if key_press.image_index < (len(images) - 1):
                key_press.image_index += 1
            else:
                return
        else:
            if key_press.image_index > 0:
                key_press.image_index -= 1
            else:
                return

        pre_xlim = axes.get_xlim()
        pre_ylim = axes.get_ylim()

        plt.cla()

        img = images[key_press.image_index]
        iw, ih = img.size
        print(f"\033[A{img.filename}: {iw}x{ih}")

        # display size
        fig_dpi = fig.get_dpi()
        figsize = iw / float(fig_dpi), ih / float(fig_dpi)
        fig.set_size_inches(figsize)
        fig.canvas.manager.set_window_title(img.filename)

        axes.imshow(img)
        axes.set_title(f"{iw}x{ih}")

        # grid settings
        grid.set_ticks(axes, iw, ih)

        # zoom settings
        xsize = (pre_xlim[1] - pre_xlim[0]) * 1.0
        ysize = (pre_ylim[1] - pre_ylim[0]) * 1.0
        relx = (pre_xlim[1] - event.xdata) / (pre_xlim[1] - pre_xlim[0])
        rely = (pre_ylim[1] - event.ydata) / (pre_ylim[1] - pre_ylim[0])
        axes.set_xlim([event.xdata - xsize * (1 - relx), event.xdata + xsize * relx])
        axes.set_ylim([event.ydata - ysize * (1 - rely), event.ydata + ysize * rely])

        plt.draw()

    elif event.key == "i":
        if event.inaxes:
            zoom.zoomin(axes, event.xdata, event.ydata)
    elif event.key == "o":
        if event.inaxes:
            zoom.zoomout(axes, event.xdata, event.ydata)
    elif event.key == "q":
        sys.exit()
    elif event.key == "escape":
        if event.inaxes:
            zoom.reset(axes)


key_press.image_index = 0


def view(
    dataset_dir,
    annotation,
    result,
    categories,
    images,
    areas,
    grid_sizes,
    maxdet,
    score_thr,
    search_depth,
    shuffle,
    roi_mode,
    roi_block_size,
    roi_margin,
    roi_categories,
):
    cocoGt = COCO(annotation)
    iids = []
    cats = dict([(v["id"], v["name"]) for k, v in cocoGt.cats.items()])

    search_list = []
    wildcards = ["/*"] * search_depth
    wildcards = "".join(wildcards)

    # create search name list
    files = glob.glob(dataset_dir + wildcards)
    for f in files:
        name = str(pathlib.Path(".").joinpath(*pathlib.Path(f).parts[-search_depth:]))
        search_list.append(name)

    # create image ids
    for k, v in cocoGt.imgs.items():
        if v["file_name"] in search_list:
            if images:
                for pat in images:
                    if pat in v["file_name"]:
                        iids.append(k)
                        print(f"{k}: {v['file_name']}")
                        break
            else:
                iids.append(k)
                print(f"{k}: {v['file_name']}")

    # select images with categories
    cids = cocoGt.getCatIds()
    if categories:
        cids = cocoGt.getCatIds(catNms=categories)
        iids = cocoGt.getImgIds(imgIds=iids, catIds=cids)
        print(f"{categories}: {len(iids)}")

    # exit when image ids are empty
    if len(iids) == 0:
        print("exit with error: empty image ids.")
        print("-- search_list")
        for s in search_list:
            print(s)
        sys.exit()

    # load detection result
    cocoDt = None
    if result:
        cocoDt = cocoGt.loadRes(result)

    # make color palette (categories + ground truth)
    cm = plt.get_cmap("gist_rainbow_r", len(cids) + 1)
    colors = [(int(255.0 * cm(i)[0]), int(255.0 * cm(i)[1]), int(255.0 * cm(i)[2])) for i in range(len(cids) + 1)]
    palette = {i: c for i, c in zip([-1] + cids, colors)}

    # area settings
    area_ranges = []
    area_labels = ""
    skips = [False] * len(iids)

    if areas and len(areas) == 2:
        area_ranges = [float(area) * float(area) for area in areas]
        area_labels = f"area [{areas[0]}x{areas[0]}-{areas[1]}x{areas[1]}]"
        skips = [True] * len(iids)
        for i, iid in enumerate(iids):
            for target in cocoGt.loadAnns(cocoGt.getAnnIds(iid)):
                if (target["area"] > area_ranges[0]) and (target["area"] <= area_ranges[1]):
                    skips[i] = False
                    break
        print(f"{area_labels}: {skips.count(False)}")

    # change default keymaps
    plt.rcParams["keymap.xscale"].remove("k")
    plt.rcParams["keymap.quit"] = "n"
    plt.rcParams["keymap.zoom"].remove("o")

    # randomize skips and iids
    if shuffle:
        merge = list(zip(skips, iids))
        random.shuffle(merge)
        skips, iids = zip(*merge)

    # check roi parameters
    if roi_mode >= 0 and roi_mode <= 1:
        if roi_block_size <= 0:
            print(f"ROI is not displayed because of invalid roi_block_size={roi_block_size} (> 0)")
            roi_mode = -1
        if roi_margin < 0:
            print(f"ROI is not displayed because of invalid roi_margin={roi_margin} (>= 0)")
            roi_mode = -1
    elif roi_mode >= 2:
        print(f"ROI is not displayed because of invalid roi_mode={roi_mode} (0-1)")
        roi_mode = -1

    for skip, iid in zip(skips, iids):
        if skip is True:
            continue

        obj = cocoGt.loadImgs(iid)[0]
        fn = pathlib.Path(dataset_dir + "/" + obj["file_name"])
        gt = cocoGt.loadAnns(cocoGt.getAnnIds(iid))
        iw = obj["width"]
        ih = obj["height"]
        print(f"{fn}: {iw}x{ih}")

        img_gt = Image.open(fn).convert("RGBA")
        img_gt.filename = fn
        drw_gt = ImageDraw.Draw(img_gt)

        for target in gt:
            x, y, w, h = target["bbox"]
            cat = cats[target["category_id"]]
            rect_color = "white"
            text_color = "black"

            if len(area_ranges) == 2:
                if (target["area"] > area_ranges[0]) and (target["area"] <= area_ranges[1]):
                    rect_color = palette[-1]
                    text_color = "white"

            if categories:
                if cat in categories:
                    rect_color = palette[-1]
                    text_color = "white"

            rect_text = f"{cat} {w}x{h}"
            _, _, tw, th = drw_gt.textbbox((0, 0), rect_text)
            tx = x
            ty = y - th - 2
            drw_gt.rectangle((x, y, x + w - 1, y + h - 1), outline=rect_color)
            drw_gt.rectangle((tx, ty, tx + tw + 2, ty + th + 2), outline=rect_color, fill=rect_color)
            drw_gt.text((tx + 1, ty + 0), rect_text, fill=ImageColor.getrgb(text_color))

        if roi_mode == 0:
            img_roi = Image.new("RGBA", img_gt.size)
            drw_roi = ImageDraw.Draw(img_roi)

            if roi_categories:
                roi_cids = cocoGt.getCatIds(catNms=roi_categories)
            else:
                roi_cids = cids

            roi_info = cl.defaultdict()
            for target in cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=iid, catIds=roi_cids)):
                x, y, w, h = target["bbox"]
                hblocks = int((iw + (roi_block_size - 1)) / roi_block_size)
                bbox = [x, y, x + w - 1, y + h - 1]
                bbox_ext = [
                    x - roi_margin if (x - roi_margin) >= 0 else 0,
                    y - roi_margin if (y - roi_margin) >= 0 else 0,
                    x + w + roi_margin - 1 if (x + w + roi_margin - 1) < iw else iw - 1,
                    y + h + roi_margin - 1 if (y + h + roi_margin - 1) < ih else ih - 1,
                ]

                ovl_region = [int(b / roi_block_size) for b in bbox]
                nbr_region = [int(b / roi_block_size) for b in bbox_ext]
                nbr_hblocks = nbr_region[2] - nbr_region[0] + 1
                nbr_vblocks = nbr_region[3] - nbr_region[1] + 1

                for tmp_by in range(nbr_vblocks):
                    for tmp_bx in range(nbr_hblocks):
                        bx = nbr_region[0] + tmp_bx
                        by = nbr_region[1] + tmp_by
                        bn = by * hblocks + bx
                        corner_xl = bx * roi_block_size
                        corner_yt = by * roi_block_size
                        hoverlap = (bx >= ovl_region[0]) and (bx <= ovl_region[2])
                        voverlap = (by >= ovl_region[1]) and (by <= ovl_region[3])

                        if roi_info.get(bn) is None:
                            roi_info[bn] = {
                                "position": [bx, by],
                                "corner": [corner_xl, corner_yt],
                                "num_of_overlapped_datas": 0,
                                "num_of_neighbor_datas": 0,
                            }

                        if hoverlap and voverlap:
                            # overlapped block
                            roi_info[bn]["num_of_overlapped_datas"] += 1
                        else:
                            # neighbor block
                            roi_info[bn]["num_of_neighbor_datas"] += 1

            for bn, roi in roi_info.items():
                bx = roi["position"][0]
                by = roi["position"][1]
                xl = roi["corner"][0]
                yt = roi["corner"][1]
                xr = xl + roi_block_size
                yb = yt + roi_block_size

                rect_text = f"{bn},{bx},{by}"
                _, _, tw, th = drw_gt.textbbox((0, 0), rect_text)
                if tw > roi_block_size:
                    rect_text = f"{bn}"
                    _, _, tw, th = drw_gt.textbbox((0, 0), rect_text)
                    if tw > roi_block_size:
                        rect_text = ""

                if roi["num_of_overlapped_datas"] > 0:
                    # overlapped block
                    drw_roi.rectangle((xl, yt, xr, yb), outline=(0, 255, 255, 255), fill=(0, 255, 255, 32))
                    drw_roi.text((xl + 1, yt + 1), rect_text, fill=ImageColor.getrgb("white"))
                else:
                    # neighbor block
                    drw_roi.rectangle((xl, yt, xr, yb), outline=(173, 255, 47, 255), fill=(173, 255, 47, 32))
                    drw_roi.text((xl + 1, yt + 1), rect_text, fill=ImageColor.getrgb("white"))

            img_gt = Image.alpha_composite(img_gt, img_roi)
            img_gt.filename = fn

        imgs = [img_gt]

        if cocoDt is not None:
            dt = cocoDt.loadAnns(cocoDt.getAnnIds(imgIds=iid, catIds=cids))
            dtind = np.argsort([-d["score"] for d in dt], kind="mergesort")
            dtind = dtind[0:maxdet] if maxdet > 0 else dtind
            dt = [dt[i] for i in dtind if dt[i]["score"] >= score_thr]

            img_dt = img_gt.copy()
            img_dt.filename = img_gt.filename
            drw_dt = ImageDraw.Draw(img_dt)

            for target in dt:
                x, y, w, h = target["bbox"]
                cat = cats[target["category_id"]]
                col = palette[target["category_id"]]
                score = round(target["score"], 2)

                _, _, tw, th = drw_dt.textbbox((0, 0), f"{cat} {score}")
                tx = x
                ty = y - th - 2
                drw_dt.rectangle((x, y, x + w, y + h), outline=col)
                drw_dt.rectangle((tx, ty, tx + tw + 2, ty + th + 2), outline=col, fill=col)
                drw_dt.text((tx + 1, ty + 0), f"{cat} {score}", fill=ImageColor.getrgb("white"))

            if roi_mode == 1:
                img_roi = Image.new("RGBA", img_gt.size)
                drw_roi = ImageDraw.Draw(img_roi)

                roi_info = cl.defaultdict()
                for target in dt:
                    x, y, w, h = target["bbox"]
                    x = int(x + 0.5)
                    y = int(y + 0.5)
                    w = int(w + 0.5)
                    h = int(h + 0.5)
                    hblocks = int((iw + (roi_block_size - 1)) / roi_block_size)

                    bbox = [
                        x,
                        y,
                        x + w - 1 if (x + w - 1) < iw else iw - 1,
                        y + h - 1 if (y + h - 1) < ih else ih - 1,
                    ]
                    bbox_ext = [
                        x - roi_margin if (x - roi_margin) >= 0 else 0,
                        y - roi_margin if (y - roi_margin) >= 0 else 0,
                        x + w + roi_margin - 1 if (x + w + roi_margin - 1) < iw else iw - 1,
                        y + h + roi_margin - 1 if (y + h + roi_margin - 1) < ih else ih - 1,
                    ]

                    ovl_region = [int(b / roi_block_size) for b in bbox]
                    nbr_region = [int(b / roi_block_size) for b in bbox_ext]
                    nbr_hblocks = nbr_region[2] - nbr_region[0] + 1
                    nbr_vblocks = nbr_region[3] - nbr_region[1] + 1

                    for tmp_by in range(nbr_vblocks):
                        for tmp_bx in range(nbr_hblocks):
                            bx = nbr_region[0] + tmp_bx
                            by = nbr_region[1] + tmp_by
                            bn = by * hblocks + bx
                            corner_xl = bx * roi_block_size
                            corner_yt = by * roi_block_size
                            hoverlap = (bx >= ovl_region[0]) and (bx <= ovl_region[2])
                            voverlap = (by >= ovl_region[1]) and (by <= ovl_region[3])

                            if roi_info.get(bn) is None:
                                roi_info[bn] = {
                                    "position": [bx, by],
                                    "corner": [corner_xl, corner_yt],
                                    "num_of_overlapped_datas": 0,
                                    "num_of_neighbor_datas": 0,
                                }

                            if hoverlap and voverlap:
                                # overlapped block
                                roi_info[bn]["num_of_overlapped_datas"] += 1
                            else:
                                # neighbor block
                                roi_info[bn]["num_of_neighbor_datas"] += 1

                for bn, roi in roi_info.items():
                    bx = roi["position"][0]
                    by = roi["position"][1]
                    xl = roi["corner"][0]
                    yt = roi["corner"][1]
                    xr = xl + roi_block_size
                    yb = yt + roi_block_size

                    rect_text = f"{bn},{bx},{by}"
                    _, _, tw, th = drw_gt.textbbox((0, 0), rect_text)
                    if tw > roi_block_size:
                        rect_text = f"{bn}"
                        _, _, tw, th = drw_gt.textbbox((0, 0), rect_text)
                        if tw > roi_block_size:
                            rect_text = ""

                    if roi["num_of_overlapped_datas"] > 0:
                        # overlapped block
                        drw_roi.rectangle((xl, yt, xr, yb), outline=(0, 255, 255, 255), fill=(0, 255, 255, 32))
                        drw_roi.text((xl + 1, yt + 1), rect_text, fill=ImageColor.getrgb("white"))
                    else:
                        # neighbor block
                        drw_roi.rectangle((xl, yt, xr, yb), outline=(173, 255, 47, 255), fill=(173, 255, 47, 32))
                        drw_roi.text((xl + 1, yt + 1), rect_text, fill=ImageColor.getrgb("white"))

                img_dt = Image.alpha_composite(img_dt, img_roi)
                img_dt.filename = img_gt.filename

            imgs.append(img_dt)

        fig, axes = plt.subplots()

        # display size
        fig_dpi = fig.get_dpi()
        figsize = iw / float(fig_dpi), ih / float(fig_dpi)
        fig.set_size_inches(figsize)
        axes.imshow(img_gt)
        axes.set_title(f"{iw}x{ih}")

        # zoom settings
        zoom = Zoom(axes)

        # grid settings
        grid = Grid(*grid_sizes)
        grid.set_ticks(axes, iw, ih)

        # key event
        fig.canvas.manager.set_window_title(fn)
        fig.canvas.mpl_connect(
            "key_press_event",
            lambda event: key_press(event, fig, axes, zoom, grid, imgs),
        )
        key_press.image_index = 0

        # show image
        plt.show()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_dir", help="dataset directory")
    parser.add_argument("annotation", help="input annotation file")
    parser.add_argument("-result", "--result", type=str, default="", help="input result file (COCO format)")
    parser.add_argument("-categories", "--categories", nargs="+")
    parser.add_argument("-images", "--images", nargs="+")
    parser.add_argument("-areas", "--areas", type=int, nargs="+")
    parser.add_argument("-gw", "--grid_width", type=int, default=608)
    parser.add_argument("-gh", "--grid_height", type=int, default=608)
    parser.add_argument("-sgw", "--small_grid_width", type=int, default=32)
    parser.add_argument("-sgh", "--small_grid_height", type=int, default=32)
    parser.add_argument("-maxdet", "--maxdet", type=int, default=100)
    parser.add_argument("-score_thr", "--score_thr", type=float, default=0.3)
    parser.add_argument("-search_depth", "--search_depth", type=int, default=1, help="search depth from dataset_dir")
    parser.add_argument("-shuffle", "--shuffle", action="store_true")
    parser.add_argument("-roi_mode", "--roi_mode", type=int, default=-1, help="0: gound truth, 1: detection results")
    parser.add_argument("-roi_block_size", "--roi_block_size", type=int, default=64)
    parser.add_argument("-roi_margin", "--roi_margin", type=int, default=32)
    parser.add_argument("-roi_categories", "--roi_categories", nargs="+")

    args = parser.parse_args()
    view(
        args.dataset_dir,
        args.annotation,
        args.result,
        args.categories,
        args.images,
        args.areas,
        (args.grid_width, args.grid_height, args.small_grid_width, args.small_grid_height),
        args.maxdet,
        args.score_thr,
        args.search_depth,
        args.shuffle,
        args.roi_mode,
        args.roi_block_size,
        args.roi_margin,
        args.roi_categories,
    )


if __name__ == "__main__":
    main()
