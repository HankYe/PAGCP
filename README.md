### Introduction

This is the official implementation of PAGCP for YOLOv5 compression in the paper, [**Performance-aware Approximation of Global Channel Pruning for Multitask CNNs**](https://github.com/HankYe/yolov5prune). PAGCP is a novel pruning paradigm containing a sequentially greedy channel pruning algorithm and a performance-aware oracle criterion, to approximately solve the objective problem of GCP. The developed pruning strategy dynamically computes the filter saliency in a greedy fashion based on the pruned structure at the previous step, and control each layer’s pruning ratio by the constraint of the performance-aware oracle criterion.

### Main Results on COCO2017

[assets]: https://github.com/HankYe/yolov5prune/releases

|Model |size<br><sup>(pixels) |mAP<sup>val<br>0.5:0.95 |mAP<sup>val<br>0.5  | |params<br><sup>(M) |FLOPs<br><sup>640 (B)
|---                          |---  |---      |---      |---|---   |---
|[YOLOv5m][assets]            |640  |43.6     |62.7     |   |21.4  |51.3
|[YOLOv5m_pruned][assets]     |640  |41.5     |60.7     |   |7.7   |23.5
|[YOLOv5l][assets]            |640  |47.0     |66.0     |   |46.7  |115.4
|[YOLOv5l_pruned][assets]     |640  |45.5     |64.5     |   |16.1  |49.1
|[YOLOv5x][assets]            |640  |48.8     |67.7     |   |87.4  |218.8
|[YOLOv5x_pruned][assets]     |640  |47.2     |66.1     |   |29.3  |81.0

<details>
  <summary>Table Notes</summary>

* AP values are for single-model single-scale. **Reproduce mAP**
  by `python val.py --data coco.yaml --img 640 --weights /path/to/model/checpoints`
* All pre-trained and pruned models are trained with hyp.scratch.yaml to align the setting.

</details>
<details open>
<summary>Install</summary>

[**Python>=3.6.0**](https://www.python.org/) is required with all [requirements.txt](https://github.com/HankYe/yolov5prune/blob/master/requirements.txt) installed including [**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->

```bash
$ git clone https://github.com/HankYe/yolov5prune
$ cd yolov5prune
$ conda create -n yolov5prune python==3.8 # (>=3.6)
$ pip install -r requirements.txt
```

</details>

<details>
<summary>Compression</summary>

Repeatedly run the command below to prune models on [COCO](https://github.com/HankYe/yolov5prune/blob/master/data/scripts/get_coco.sh) dataset, in which hyper-parameters can be tuned to get better compression performance.

```bash
$ python compress.py --model $model name$ --dataset COCO --data coco.yaml --batch 64 --weights /path/to/to-prune/model --initial_rate 0.06 --initial_thres 6. --topk 0.8 --exp --sequential --device 0
```

</details>


### Citation
If you find this work helpful in your research, please cite.
````
@article{Ye22performance,
  title={Performance-aware Approximation of Global Channel Pruning for Multitask CNNs},
  author={Hancheng Ye and Bo Zhang and Tao Chen and Jianyuan Fan and Bin Wang},
  journal={},
  year={}
}
````

### Acknowledgement
We greatly acknowledge the authors of _YOLOv5_ and _Torch_pruning_ for their open-source codes. Visit the following links to access more contributions of them.
* [YOLOv5](https://github.com/ultralytics/yolov5)
* [Torch_pruning](https://github.com/VainF/Torch-Pruning)
