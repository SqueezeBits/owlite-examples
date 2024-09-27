# OwLite Pose Estimation Example 
- Model: YOLOv8-S
- Dataset: COCO'17 Pose Dataset

## Prerequisites

### Prepare dataset
If you already have [COCO 2017 dataset](http://cocodataset.org), kindly modify the data path in within the "ultralytics/cfg/datasets/coco-pose.yaml" file. Alternatively, executing the baseline will automatically download the dataset to the designated directory.

### Apply patch
```
cd YOLOv8
patch -p1 < ../apply_owlite.patch
```

### Setup environment
1. create conda env and activate it
    ```
    conda create -n <env_name> python=3.10 -y
    conda activate <env_name>
    ```
    Conda environment can be created with Python versions between 3.10 and 3.12 by replacing ```3.10``` with ```3.11``` or ```3.12```. Compatible Python versions for each PyTorch version can be found in [PyTorch compatibility matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix).

2. install required packages
    ```
    pip install .
    ```
3. install OwLite package following the [installation guide](https://squeezebits.gitbook.io/owlite/user-guide/getting-started/install)


## How To Run

### Run baseline model
```
CUDA_VISIBLE_DEVICES=0 python main.py --model-dir yolov8s-pose.pt --data-cfg ultralytics/cfg/datasets/coco-pose.yaml --task pose owlite --project <owlite_project_name> --baseline <owlite_baseline_name> 
```

### Run quantized model
1. Create an experiment and save the config on OwLite GUI
2. Run the code for OwLite PTQ 
    ```
    CUDA_VISIBLE_DEVICES=0 python main.py --model-dir yolov8s-pose.pt --data-cfg ultralytics/cfg/datasets/coco-pose.yaml --task pose owlite --project <owlite_project_name> --baseline <owlite_baseline_name> --experiment <owlite_experiment_name> --ptq
    ```

## Results

<details>
<summary>YOLOv8-S</summary>

### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: Percentile (99.9) for the activation of the last conv layers of the `cv4` modules in each head, MSE for the others.

    
### Accuracy and Lateny Results
TensorRT Evaluation GPU: A6000

| Quantization     | Input Size        | mAP 0.50~0.95 (%) | mAP 0.50 (%) | GPU Latency (ms) | 
| ---------------|:-----------------:|:-----------------:|:------------:|:----------------:|
| FP16 TensorRT    | (32, 3, 640, 640) | 59.8              | 85.8         | 18.81            |
| OwLite INT8 PTQ  | (32, 3, 640, 640) | 58.3              | 85.8         | 10.02            |
| OwLite INT8 PTQ* | (32, 3, 640, 640) | 59.3              | 85.7         | 10.12            |
| INT8 TensorRT    | (32, 3, 640, 640) | 56.1              | 84.7         | 10.43            |

- The INT8 TensorRT engine was built by applying FP16 and INT8 flags using [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy), as further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide).
- *Results by applying FP16 weight and activation of the last conv layers in the `cv4` modules in each head from OwLite Recommended Config
</details>

## Reference
https://github.com/ultralytics/ultralytics