# OwLite Object Detection Example 

## Prerequisites

### Prepare dataset
Prepare [COCO 2017 dataset](http://cocodataset.org) referring to the [README](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/README.md) from the original repository.

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
    pip install -r requirements.txt
    ```
3. install OwLite package
    ```
    pip install --extra-index-url https://pypi.ngc.nvidia.com git+https://github.com/SqueezeBits/owlite
    ```


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

- We tested on the pretrained YOLOX-{s, m, l, x} models, which can be downloaded at the original [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) repository

## Results

<details>
<summary>YOLOX-S</summary>

### Configuration
#### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: MSE

### Accuracy Results

| Quantization    | Input Size        | mAP 0.50~0.95 (%) |
| --------------- |:-----------------:|:-----------------:|
| FP32            | (64, 3, 640, 640) | 40.5 |
| OwLite INT8 PTQ | (64, 3, 640, 640) | 39.9 |

### Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | GPU Latency (ms) | 
| --------------- |:-----------------:|:----------------:|
| FP16 TensorRT   | (64, 3, 640, 640) | 33.38            |
| OwLite INT8 PTQ | (64, 3, 640, 640) | 19.45            |

</details>

<details>
<summary>YOLOX-M</summary>

### Configuration
#### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: MSE

### Accuracy Results

| Quantization    | Input Size        | mAP 0.50~0.95 (%) |
| --------------- |:-----------------:|:-----------------:|
| FP32            | (32, 3, 640, 640) | 46.9 |
| OwLite INT8 PTQ | (32, 3, 640, 640) | 46.4 |

### Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | GPU Latency (ms) | 
| --------------- |:-----------------:|:----------------:|
| FP16 TensorRT   | (32, 3, 640, 640) | 37.37            |
| OwLite INT8 PTQ | (32, 3, 640, 640) | 20.33            |

</details>

<details>
<summary>YOLOX-L</summary>

### Configuration
#### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: MSE

### Accuracy Results

| Quantization    | Input Size        | mAP 0.50~0.95 (%) | 
| --------------- |:-----------------:|:-----------------:|
| FP32            | (32, 3, 640, 640) | 49.7 |
| OwLite INT8 PTQ | (32, 3, 640, 640) | 48.9 |

### Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | GPU Latency (ms) | 
| --------------- |:-----------------:|:----------------:|
| FP16 TensorRT   | (32, 3, 640, 640) | 61.52            |
| OwLite INT8 PTQ | (32, 3, 640, 640) | 31.04            |

</details>

<details>
<summary>YOLOX-X</summary>

### Configuration
#### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: MSE

### Accuracy Results

| Quantization    | Input Size        | mAP 0.50~0.95 (%) | 
| --------------- |:-----------------:|:-----------------:|
| FP32            | (16, 3, 640, 640) | 51.1 |
| OwLite INT8 PTQ | (16, 3, 640, 640) | 50.4 |

### Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | GPU Latency (ms) | 
| --------------- |:-----------------:|:----------------:|
| FP16 TensorRT   | (16, 3, 640, 640) | 58.79            |
| OwLite INT8 PTQ | (16, 3, 640, 640) | 29.56            |

</details>

## Reference
https://github.com/Megvii-BaseDetection/YOLOX
