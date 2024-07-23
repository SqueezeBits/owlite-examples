# OwLite Object Detection Example 
- Model: YOLOX-S, YOLOX-M, YOLOX-L, YOLOX-X
- Dataset: COCO'17 Dataset

## Prerequisites

### Prepare dataset
Prepare [COCO 2017 dataset](http://cocodataset.org) referring to the [README](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/datasets/README.md) from the original repository.

### Apply patch
```
cd YOLOX
patch -p1 < ../apply_owlite.patch
```

### Setup environment
1. create conda env and activate it
    ```
    conda create -n <env_name> python=3.10 -y
    conda activate <env_name>
    ```
2. install required packages
    ```
    pip install -e .
    ```
3. install OwLite package following the [installation guide](https://squeezebits.gitbook.io/owlite/user-guide/getting-started/install)


## How To Run

### Run baseline model
```
CUDA_VISIBLE_DEVICES=0 python -m tools.eval -n yolox-s -c yolox_s.pth -b 64 -d 1 --conf 0.001 owlite --project <owlite_project_name> --baseline <owlite_baseline_name>
```
### Run quantized model
1. Create an experiment and save the config on OwLite GUI
2. Run the code for OwLite PTQ 
    ```
    CUDA_VISIBLE_DEVICES=0 python -m tools.eval -n yolox-s -c yolox_s.pth -b 64 -d 1 --conf 0.001 owlite --project <owlite_project_name> --baseline <owlite_baseline_name> --experiment <owlite_experiment_name> --ptq
    ```

- We tested on the pretrained YOLOX-{S, M, L, X} models, which can be downloaded at the original [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) repository.

3. Run the code for OwLite QAT
    ```
    CUDA_VISIBLE_DEVICES=0 python -m tools.train -f ../exps_qat/yolox_s_owlite_qat.py -c yolox_s.pth -d 1 -b 64 owlite --project <owlite_project_name> --baseline <owlite_baseline_name> --experiment <owlite_experiment_name> --qat
    ```

## Results

<details>
<summary>YOLOX-S</summary>

### Configuration
#### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: MSE
  - QAT backward: CLQ
  - Gradient scales for weight quantization in Conv were set to 0.01

### Accuracy and Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | mAP 0.50~0.95 (%) | GPU Latency (ms) |
| --------------- |:-----------------:|:-----------------:|:----------------:|
| FP16 TensorRT   | (64, 3, 640, 640) | 40.5              | 33.38            |
| OwLite INT8 PTQ | (64, 3, 640, 640) | 40.0              | 18.43            |
| OwLite INT8 QAT | (64, 3, 640, 640) | 40.1              | 18.43            |
| INT8 TensorRT   | (64, 3, 640, 640) | 37.5              | 19.44            |

- The INT8 TensorRT engine was built by applying FP16 and INT8 flags using [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy), as further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide).
</details>

<details>
<summary>YOLOX-M</summary>

### Configuration
#### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: MSE

### Accuracy and Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | mAP 0.50~0.95 (%) | GPU Latency (ms) | 
| --------------- |:-----------------:|:-----------------:|:----------------:|
| FP16 TensorRT   | (32, 3, 640, 640) | 46.9              | 37.37            |
| OwLite INT8 PTQ | (32, 3, 640, 640) | 46.6              | 19.52            |
| INT8 TensorRT   | (32, 3, 640, 640) | 43.9              | 20.47            |

- The INT8 TensorRT engine was built by applying FP16 and INT8 flags using [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy), as further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide).
</details>

<details>
<summary>YOLOX-L</summary>

### Configuration
#### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: MSE

### Accuracy and Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | mAP 0.50~0.95 (%) | GPU Latency (ms) |
| --------------- |:-----------------:|:-----------------:|:----------------:|
| FP16 TensorRT   | (16, 3, 640, 640) | 49.7              | 31.97            |
| OwLite INT8 PTQ | (16, 3, 640, 640) | 49.3              | 16.77            |
| INT8 TensorRT   | (16, 3, 640, 640) | 47.2              | 16.59            |

- The INT8 TensorRT engine was built by applying FP16 and INT8 flags using [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy), as further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide).
</details>

<details>
<summary>YOLOX-X</summary>

### Configuration
#### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: MSE

### Accuracy and Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | mAP 0.50~0.95 (%) | GPU Latency (ms) |
| --------------- |:-----------------:|:-----------------:|:----------------:|
| FP16 TensorRT   | (16, 3, 640, 640) | 51.1              | 58.79            |
| OwLite INT8 PTQ | (16, 3, 640, 640) | 50.5              | 28.18            |
| INT8 TensorRT   | (16, 3, 640, 640) | 48.2              | 29.12            |

- The INT8 TensorRT engine was built by applying FP16 and INT8 flags using [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy), as further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide).
</details>

## Reference
https://github.com/Megvii-BaseDetection/YOLOX
