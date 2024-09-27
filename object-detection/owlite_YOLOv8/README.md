# OwLite Object Detection Example 
- Model: YOLOv8
- Dataset: COCO'17 Dataset

## Prerequisites

### Prepare dataset
If you already have [COCO 2017 dataset](http://cocodataset.org), kindly modify the data path in within the "ultralytics/cfg/datasets/coco.yaml" file. Alternatively, executing the baseline will automatically download the dataset to the designated directory.

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
CUDA_VISIBLE_DEVICES=0 python main.py owlite --project <owlite_project_name> --baseline <owlite_baseline_name> 
```

### Run quantized model
1. Create an experiment and save the config on OwLite GUI
2. Run the code for OwLite PTQ 
    ```
    CUDA_VISIBLE_DEVICES=0 python main.py owlite --project <owlite_project_name> --baseline <owlite_baseline_name> --experiment <owlite_experiment_name> --ptq
    ```

## Results
<details>
<summary>YOLOv8-N</summary>

### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: MSE

    
### Accuracy and Lateny Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | mAP 0.50~0.95 (%) | mAP 0.50 (%) | GPU Latency (ms) | 
| --------------- |:-----------------:|:-----------------:|:------------:|:----------------:|
| FP16 TensorRT   | (32, 3, 640, 640) | 37.0              | 52.1         | 9.74             |
| OwLite INT8 PTQ | (32, 3, 640, 640) | 35.8              | 51.3         | 6.70             |
| INT8 TensorRT   | (32, 3, 640, 640) | 34.4              | 49.2         | 6.29             |

- The INT8 TensorRT engine was built by applying FP16 and INT8 flags using [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy), as further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide).
</details>

<details>
<summary>YOLOv8-S</summary>

### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: MSE

    
### Accuracy and Lateny Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | mAP 0.50~0.95 (%) | mAP 0.50 (%) | GPU Latency (ms) | 
| --------------- |:-----------------:|:-----------------:|:------------:|:----------------:|
| FP16 TensorRT   | (32, 3, 640, 640) | 44.7              | 61.2         | 17.5             |
| OwLite INT8 PTQ | (32, 3, 640, 640) | 43.9              | 60.7         | 9.17             |
| INT8 TensorRT   | (32, 3, 640, 640) | 41.6              | 57.6         | 9.48             |

- The INT8 TensorRT engine was built by applying FP16 and INT8 flags using [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy), as further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide).
</details>

<details>
<summary>YOLOv8-M</summary>

### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: MSE

    
### Accuracy and Lateny Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | mAP 0.50~0.95 (%) | mAP 0.50 (%) | GPU Latency (ms) | 
| --------------- |:-----------------:|:-----------------:|:------------:|:----------------:|
| FP16 TensorRT   | (32, 3, 640, 640) | 49.9              | 66.6         | 39.3             |
| OwLite INT8 PTQ | (32, 3, 640, 640) | 48.6              | 66.2         | 22.5             |
| INT8 TensorRT   | (32, 3, 640, 640) | 46.9              | 62.8         | 21.9             |

- The INT8 TensorRT engine was built by applying FP16 and INT8 flags using [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy), as further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide).
</details>

<details>
<summary>YOLOv8-L</summary>

### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: MSE

    
### Accuracy and Lateny Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | mAP 0.50~0.95 (%) | mAP 0.50 (%) | GPU Latency (ms) | 
| --------------- |:-----------------:|:-----------------:|:------------:|:----------------:|
| FP16 TensorRT   | (32, 3, 640, 640) | 52.6              | 69.2         | 63.0             |
| OwLite INT8 PTQ | (32, 3, 640, 640) | 51.1              | 68.7         | 34.0             |
| INT8 TensorRT   | (32, 3, 640, 640) | 49.5              | 65.3         | 33.6             |

- The INT8 TensorRT engine was built by applying FP16 and INT8 flags using [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy), as further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide).
</details>

<details>
<summary>YOLOv8-X</summary>

### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: MSE

    
### Accuracy and Lateny Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | mAP 0.50~0.95 (%) | mAP 0.50 (%) | GPU Latency (ms) | 
| --------------- |:-----------------:|:-----------------:|:------------:|:----------------:|
| FP16 TensorRT   | (32, 3, 640, 640) | 53.7              | 70.2         | 105.99            |
| OwLite INT8 PTQ | (32, 3, 640, 640) | 52.1              | 69.9         | 54.15             |
| INT8 TensorRT   | (32, 3, 640, 640) | 51.0              | 67.1         | 53.44             |

- The INT8 TensorRT engine was built by applying FP16 and INT8 flags using [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy), as further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide).
</details>


## Reference
https://github.com/ultralytics/ultralytics
