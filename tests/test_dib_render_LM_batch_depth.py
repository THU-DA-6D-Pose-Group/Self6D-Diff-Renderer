# test 19 NIPS DIB-Renderer
# render multi objects in batch, one in one image
import os
import os.path as osp
import sys
import time
import cv2
import numpy as np
import torch
from tqdm import tqdm
import matplotlib

import matplotlib.pyplot as plt
from transforms3d.axangles import axangle2mat
from transforms3d.quaternions import mat2quat

# from kaolin.graphics import DIBRenderer

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../"))
from core.dr_utils.dib_renderer_x import DIBRenderer
from core.dr_utils.dr_utils import load_objs, render_dib_vc_batch

matplotlib.use("TkAgg")

output_directory = osp.join(cur_dir, "../output/results")

output_directory_dib = osp.join(output_directory, "dib")
os.makedirs(output_directory_dib, exist_ok=True)

model_root = osp.join(cur_dir, "../data/lm_models/")
HEIGHT = 480
WIDTH = 640
ZNEAR = 0.01
ZFAR = 10.0
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])


def heatmap(input, min=None, max=None, to_255=False, to_rgb=False):
    """ Returns a BGR heatmap representation """
    if min is None:
        min = np.amin(input)
    if max is None:
        max = np.amax(input)
    rescaled = 255 * ((input - min) / (max - min + 0.001))

    final = cv2.applyColorMap(rescaled.astype(np.uint8), cv2.COLORMAP_JET)
    if to_rgb:
        final = final[:, :, [2, 1, 0]]
    if to_255:
        return final.astype(np.uint8)
    else:
        return final.astype(np.float32) / 255.0


def grid_show(ims, titles=None, row=1, col=3, dpi=200, save_path=None, title_fontsize=5, show=True):
    if row * col < len(ims):
        print("_____________row*col < len(ims)___________")
        col = int(np.ceil(len(ims) / row))
    if titles is not None:
        assert len(ims) == len(titles), "{} != {}".format(len(ims), len(titles))
    fig = plt.figure(dpi=dpi, figsize=plt.figaspect(row / float(col)))
    k = 0
    for i in range(row):
        for j in range(col):
            if k >= len(ims):
                break
            plt.subplot(row, col, k + 1)
            plt.axis("off")
            plt.imshow(ims[k])
            if titles is not None:
                # plt.title(titles[k], size=title_fontsize)
                plt.text(
                    0.5,
                    1.08,
                    titles[k],
                    horizontalalignment="center",
                    fontsize=title_fontsize,
                    transform=plt.gca().transAxes,
                )
            k += 1

    # plt.tight_layout()
    if show:
        plt.show()
    else:
        if save_path is not None:
            os.makedirs(osp.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
    return fig


def main():
    objs = [
        "ape",
        "benchvise",
        # "bowl",
        "camera",
        "can",
        "cat",
        # "cup",
        "driller",
        "duck",
        "eggbox",
        "glue",
        "holepuncher",
        "iron",
        "lamp",
        "phone",
    ]
    obj_paths = [osp.join(model_root, "{}/textured.obj".format(cls_name)) for cls_name in objs]
    texture_paths = [osp.join(model_root, "{}/texture_map.png".format(cls_name)) for cls_name in objs]

    models = load_objs(obj_paths, texture_paths, height=HEIGHT, width=WIDTH)
    ren = DIBRenderer(HEIGHT, WIDTH, mode="VertexColorBatch")

    # pose =============================================
    R1 = axangle2mat((1, 0, 0), angle=0.5 * np.pi)
    R2 = axangle2mat((0, 0, 1), angle=-0.7 * np.pi)
    R = np.dot(R1, R2)
    quat = mat2quat(R)
    t = np.array([-0.1, 0.1, 1.3], dtype=np.float32)
    t2 = np.array([0.1, 0.1, 1.3], dtype=np.float32)
    t3 = np.array([-0.1, -0.1, 1.3], dtype=np.float32)
    t4 = np.array([0.1, -0.1, 1.3], dtype=np.float32)
    t5 = np.array([0, 0.1, 1.3], dtype=np.float32)
    """
    (2) render multiple objs in a batch, one obj one image
    """
    tensor_args = {"device": "cuda", "dtype": torch.float32}
    Rs = [R, R.copy(), R.copy(), R.copy(), R.copy()]
    quats = [quat, quat.copy(), quat.copy(), quat.copy(), quat.copy()]
    ts = [t, t2, t3, t4, t5]

    Rs = torch.as_tensor(np.asarray(Rs)).to(**tensor_args)
    ts = torch.as_tensor(np.asarray(ts)).to(**tensor_args)
    # poses = [np.hstack((_R, _t.reshape(3, 1))) for _R, _t in zip(Rs, ts)]
    obj_ids = np.random.choice(list(range(0, len(objs))), len(Rs))
    Ks = [K for _ in Rs]
    # bxhxwx3 rgb, bhw1 prob, bhw1 mask, bhw depth
    ren_ims, ren_probs, ren_masks, ren_depths = render_dib_vc_batch(
        ren, Rs, ts, Ks, obj_ids, models, rot_type="mat", H=480, W=640, near=0.01, far=100.0, with_depth=True
    )
    for i in range(len(Rs)):
        cur_im = ren_ims[i].detach().cpu().numpy()
        cur_prob = ren_probs[i, :, :, 0].detach().cpu().numpy()
        cur_mask = ren_masks[i, :, :, 0].detach().cpu().numpy()
        cur_depth = ren_depths[i].detach().cpu().numpy()
        show_ims = [cur_im, cur_prob, cur_mask, heatmap(cur_depth, to_rgb=True)]
        show_titles = ["im", "prob", "mask", "depth"]
        grid_show(show_ims, show_titles, row=2, col=2)


if __name__ == "__main__":
    main()
