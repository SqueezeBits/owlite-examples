# OwLite Object Detection Example 
- Model: YOLOv6
- Dataset: COCO'17 Dataset

## Prerequisites

### Prepare dataset
If you already have [COCO 2017 dataset](http://cocodataset.org), kindly modify the data path in within the "data/coco.yaml" file. Alternatively, executing the baseline will automatically download the dataset to the designated directory.

### Apply patch
```
cd YOLOv6
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
    pip install -r requirements.txt
    ```
3. install OwLite package following the [installation guide](https://squeezebits.gitbook.io/owlite/user-guide/getting-started/install)

## How To Run

### Run baseline model
```
CUDA_VISIBLE_DEVICES=0 python tools/eval.py --specific-shape owlite --project <owlite_project_name> --baseline <owlite_baseline_name> 
```

### Run quantized model
1. Create an experiment and save the config on OwLite GUI
2. Run the code for OwLite PTQ 
    ```
    CUDA_VISIBLE_DEVICES=0 python tools/eval.py --specific-shape owlite --project <owlite_project_name> --baseline <owlite_baseline_name> --experiment <owlite_experiment_name> --ptq
    ```

## Results

<details>
<summary>YOLOv6-N</summary>

### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: Percentile (99.99%)


### Accuracy and Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | mAP 0.50~0.95 (%) | mAP 0.50 (%) | GPU Latency (ms) |
| --------------- |:-----------------:|:-----------------:|:------------:|:----------------:|
| FP16 TensorRT   | (32, 3, 640, 640) | 37.4              | 53.0         | 7.2             |
| OwLite INT8 PTQ | (32, 3, 640, 640) | 35.4              | 50.8         | 4.3              |
| INT8 TensorRT   | (32, 3, 640, 640) | 34.9              | 50.7         | 4.5              |

- The INT8 TensorRT engine was built by applying FP16 and INT8 flags using [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy), as further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide).
</details>

<details>
<summary>YOLOv6-S</summary>

### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: Percentile (99.99%)

    
### Accuracy and Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | mAP 0.50~0.95 (%) | mAP 0.50 (%) | GPU Latency (ms) |  
| --------------- |:-----------------:|:-----------------:|:------------:|:----------------:|
| FP16 TensorRT   | (32, 3, 640, 640) | 44.7              | 61.6         | 17.3             |
| OwLite INT8 PTQ | (32, 3, 640, 640) | 41.8              | 58.0         | 8.6              |
| INT8 TensorRT   | (32, 3, 640, 640) | 41.0              | 57.5         | 8.7              |

- The INT8 TensorRT engine was built by applying FP16 and INT8 flags using [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy), as further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide).
</details>

<details>
<summary>YOLOv6-M</summary>

### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: Percentile (99.99%)


### Accuracy and Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | mAP 0.50~0.95 (%) | mAP 0.50 (%) | GPU Latency (ms) |
| --------------- |:-----------------:|:-----------------:|:------------:|:----------------:|
| FP16 TensorRT   | (32, 3, 640, 640) | 49.5              | 66.7         | 35.13            |
| OwLite INT8 PTQ | (32, 3, 640, 640) | 48.8              | 66.2         | 17.32            |
| INT8 TensorRT   | (32, 3, 640, 640) | 37.0              | 61.5         | 17.73            |

- The INT8 TensorRT engine was built by applying FP16 and INT8 flags using [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy), as further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide).
</details>

<details>
<summary>YOLOv6-L</summary>

### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: MSE


### Accuracy and Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | mAP 0.50~0.95 (%) | mAP 0.50 (%) | GPU Latency (ms) |
| --------------- |:-----------------:|:-----------------:|:------------:|:----------------:|
| FP16 TensorRT   | (32, 3, 640, 640) | 52.1              | 69.9         | 59.1             |
| OwLite INT8 PTQ | (32, 3, 640, 640) | 51.6              | 69.7         | 31.7             |
| INT8 TensorRT   | (32, 3, 640, 640) | 43.8              | 65.8         | 30.8              |

- The INT8 TensorRT engine was built by applying FP16 and INT8 flags using [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy), as further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide).
</details>

## Reference
https://github.com/meituan/YOLOv6
