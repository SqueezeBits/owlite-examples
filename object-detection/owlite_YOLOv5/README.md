# OwLite Object Detection Example 
- Model: YOLOv5
- Dataset: COCO'17 Dataset

## Prerequisites

### Prepare dataset
If you already have [COCO 2017 dataset](http://cocodataset.org), kindly modify the data path in within the "data/coco.yaml" file. Alternatively, executing the baseline will automatically download the dataset to the designated directory.

### Apply patch
```
cd YOLOv5
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
python train.py owlite --project <owlite_project_name> --baseline <owlite_baseline_name>
```

### Run quantized model
1. Create an experiment and save the config on OwLite GUI
2. Run the code for OwLite PTQ 
    ```
    python train.py owlite --project <owlite_project_name> --baseline <owlite_baseline_name> --experiment <owlite_experiment_name> --ptq
    ```
3. Run the code for OwLite QAT
    ```
    python train.py owlite --project <owlite_project_name> --baseline <owlite_baseline_name> --experiment <owlite_experiment_name> --qat
    ```

## Results

<details>
<summary>YOLOv5-N</summary>

### Configuration

#### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: MSE
  - QAT backward: CLQ

#### Training Configuration

- Learning Rate: 1e-4
- Weight Decay: 1e-4
- Epochs: 4
    
### Accuracy and Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | mAP 0.50~0.95 (%) | mAP 0.50 (%) | GPU Latency (ms) | 
| --------------- |:-----------------:|:-----------------:|:------------:|:----------------:|
| FP16 TensorRT   | (32, 3, 640, 640) | 23.8              | 39.7         | 9.23             |
| OwLite INT8 PTQ | (32, 3, 640, 640) | 22.7              | 38.2         | 7.47             |
| Owlite INT8 QAT | (32, 3, 640, 640) | 23.9              | 41.1         | 7.47             |
| INT8 TensorRT   | (32, 3, 640, 640) | 16.4              | 29.4         | 7.78             |



- The INT8 TensorRT engine was built by applying FP16 and INT8 flags using [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy), as further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide).
</details>

<details>
<summary>YOLOv5-S</summary>

### Configuration

#### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: MSE
  - QAT backward: CLQ

#### Training Configuration

- Learning Rate: 1e-4
- Weight Decay: 1e-4
- Epochs: 2
    
### Accuracy and Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | mAP 0.50~0.95 (%) | mAP 0.50 (%) | GPU Latency (ms) | 
| --------------- |:-----------------:|:-----------------:|:------------:|:----------------:|
| FP16 TensorRT   | (32, 3, 640, 640) | 33.2              | 50.6         | 15.1             |
| OwLite INT8 PTQ | (32, 3, 640, 640) | 32.6              | 50.3         | 9.76             |
| OwLite INT8 QAT | (32, 3, 640, 640) | 33.2              | 51.6         | 9.76             |
| INT8 TensorRT   | (32, 3, 640, 640) | 28.7              | 45.2         | 9.75             |

- The INT8 TensorRT engine was built by applying FP16 and INT8 flags using [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy), as further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide).
</details>

<details>
<summary>YOLOv5-M</summary>

### Configuration

#### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: MSE
  - QAT backward: CLQ

#### Training Configuration

- Learning Rate: 1e-4
- Weight Decay: 1e-4
- Epochs: 4
    
### Accuracy and Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | mAP 0.50~0.95 (%) | mAP 0.50 (%) | GPU Latency (ms) | 
| --------------- |:-----------------:|:-----------------:|:------------:|:----------------:|
| FP16 TensorRT   | (32, 3, 640, 640) | 41.2              | 58.4         | 28.18            |
| OwLite INT8 PTQ | (32, 3, 640, 640) | 40.4              | 57.9         | 16.92            |
| OwLite INT8 QAT | (32, 3, 640, 640) | 41.2              | 59.7         | 16.92            |
| INT8 TensorRT   | (32, 3, 640, 640) | 34.5              | 49.6         | 17.45            |


- The INT8 TensorRT engine was built by applying FP16 and INT8 flags using [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy), as further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide).
</details>

<details>
<summary>YOLOv5-L</summary>

### Configuration

#### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: MSE
  - QAT backward: CLQ

#### Training Configuration

- Learning Rate: 1e-4
- Weight Decay: 1e-4
- Epochs: 4
    
### Accuracy and Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | mAP 0.50~0.95 (%) | mAP 0.50 (%) | GPU Latency (ms) | 
| --------------- |:-----------------:|:-----------------:|:------------:|:----------------:|
| FP16 TensorRT   | (16, 3, 640, 640) | 44.8              | 61.8         | 24.42            |
| OwLite INT8 PTQ | (16, 3, 640, 640) | 44.1              | 61.5         | 12.92            |
| OwLite INT8 QAT | (16, 3, 640, 640) | 44.7              | 62.8         | 12.92            |
| INT8 TensorRT   | (16, 3, 640, 640) | 40.5              | 56.3         | 13.33            |

- The INT8 TensorRT engine was built by applying FP16 and INT8 flags using [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy), as further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide).
</details>

<details>
<summary>YOLOv5-X</summary>

### Configuration

#### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: MSE
  - QAT backward: CLQ

#### Training Configuration

- Learning Rate: 1e-4
- Weight Decay: 1e-4
- Epochs: 4
    
### Accuracy and Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | mAP 0.50~0.95 (%) | mAP 0.50 (%) | GPU Latency (ms) | 
| --------------- |:-----------------:|:-----------------:|:------------:|:----------------:|
| FP16 TensorRT   | (16, 3, 640, 640) | 46.5              | 63.3         | 46.48            |
| OwLite INT8 PTQ | (16, 3, 640, 640) | 46.0              | 63.1         | 24.22            |
| OwLite INT8 QAT | (16, 3, 640, 640) | 46.3              | 64.2         | 24.22            |
| INT8 TensorRT   | (16, 3, 640, 640) | 41.4              | 56.9         | 24.64            |

- The INT8 TensorRT engine was built by applying FP16 and INT8 flags using [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy), as further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide).
</details>

## Reference
https://github.com/ultralytics/yolov5
