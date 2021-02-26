
# Self6D-Diff-Renderer
This is the code of differentiable rendering used in the work:

**Gu Wang\*, Fabian Manhardt\*, Jianzhun Shao, Xiangyang Ji, Nassir Navab, Federico Tombari. Self6D: Self-Supervised Monocular 6D Object Pose Estimation. In ECCV 2020 (oral).**
[[ArXiv]](https://arxiv.org/abs/2004.06468)
[[Video]](https://youtu.be/bEtzjb8f430)
[[Bilibili]](https://www.bilibili.com/video/BV1iV411U77h/)

We mainly extend the implementation of DIB-Renderer from [kaolin](https://github.com/NVIDIAGameWorks/kaolin) to support:
- perspective projection with real camera intrinsics
- rendering depth maps

## Requirements
1. Ubuntu >= 16.04, CUDA >= 10.0, Python >= 3.6, PyTorch >=1.3
2. kaolin (currently only support <= v0.1)
    ```
    git clone https://github.com/NVIDIAGameWorks/kaolin.git
    cd kaolin
    git checkout v0.1
    python setup.py develop
    ```

## Usage
We provide an example for rendering LINEMOD objects, just run
```
python tests/test_dib_render_LM_batch_depth.py
```

## Citing
If you find this useful in your research, please consider citing:
```
@InProceedings{wang2020self6d,
    title={Self6D: Self-Supervised Monocular 6D Object Pose Estimation},
    author={Wang, Gu and Manhardt, Fabian and Shao, Jianzhun and Ji, Xiangyang and Navab, Nassir and Tombari, Federico},
    booktitle={The European Conference on Computer Vision (ECCV)},
    month={August},
    year={2020}
}
```
and the original DIB-Renderer
```
@inproceedings{chen2019learning_dibrenderer,
  title={Learning to predict 3d objects with an interpolation-based differentiable renderer},
  author={Chen, Wenzheng and Ling, Huan and Gao, Jun and Smith, Edward and Lehtinen, Jaakko and Jacobson, Alec and Fidler, Sanja},
  booktitle={NeurIPS},
  pages={9605--9616},
  year={2019}
}
```
