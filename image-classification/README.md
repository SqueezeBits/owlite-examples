# OwLite Image Classification Example 
- Model : ResNet18, ResNet50, MobileNet-V2, MobileNet-V3-Large, EfficientNet-B0, EfficientNet-V2-S, Swin-B
- Dataset : ImageNet Dataset



## Prerequisites

### Prepare dataset
Download ImageNet dataset referring to the [README](https://github.com/pytorch/examples/blob/main/imagenet/README.md) from the original repository.

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

### Quick Start

### Run baseline model
```
python main.py -a <model_name> --data <dataset_dir> --pretrained --evaluate owlite --project <owlite_project_name> --baseline <owlite_baseline_name>
```
#### Example:
```
python main.py -a resnet18 --data ~/datasets/imagenet --pretrained --evaluate owlite --project TestProject --baseline torchvision_ResNet18
```

### Run quantized model
1. Create an experiment and save the config on OwLite GUI
2. Run the code for OwLite PTQ 
    ```
    python main.py -a <model_name> --data <dataset_dir> --pretrained owlite --project <owlite_project_name> --baseline <owlite_baseline_name> --experiment <owlite_experiment_name> --ptq
    ```
    Example:
    ```
    python main.py -a resnet18 --data ~/datasets/imagenet --pretrained --evaluate owlite --project TestProject --baseline torchvision_ResNet18 --experiment ResNet18_PTQ --ptq
    ```

3. Run the code for OwLite QAT
    ```
    python main.py -a <model_name> --data <dataset_dir> --pretrained owlite --project <owlite_project_name> --baseline <owlite_baseline_name> --experiment <owlite_experiment_name> --qat
    ```
    Example:
    ```
    python main.py -a resnet18 --data ~/datasets/imagenet --pretrained --evaluate owlite --project TestProject --baseline torchvision_ResNet18 --experiment ResNet18_QAT --qat
    ```

## Results

<details>
<summary>ResNet-18</summary>

### Configuration

#### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: AbsMax
  - QAT backward: CLQ
  - Gradient scales for weight quantization in {Conv, Gemm} were set to 0.01

#### Training Configuration

- Learning Rate: 1e-5
- Weight Decay: 1e-5
- Epochs: 1
  
### Accuracy Results

| Quantization    | Input Size         | Top 1 Acc (%) | Top 5 Acc (%) |   
| --------------- |:------------------:|:-------------:|:-------------:|
| FP32            | (256, 3, 224, 224) | 69.8          | 89.1          |
| OwLite INT8 PTQ | (256, 3, 224, 224) | 69.5          | 88.9          |
| OwLite INT8 QAT | (256, 3, 224, 224) | 69.8          | 89.2          |
| INT8 TensorRT   | (256, 3, 224, 224) | 69.5          | 89.0          |

- INT8 TensorRT engine was build using applying FP16 and INT8 flags, further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide)

### Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size         | GPU Latency (ms) | 
| --------------- |:------------------:|:----------------:|
| FP16 TensorRT   | (256, 3, 224, 224) | 10.6             |
| OwLite INT8     | (256, 3, 224, 224) | 4.76             |
| INT8 TensorRT   | (256, 3, 224, 224) | 4.60             | 

</details>

<details>
<summary>ResNet-50</summary>

### Configuration

#### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: AbsMax
  - QAT backward: CLQ
  - Gradient scales for weight quantization in {Conv, Gemm} were set to 0.01

#### Training Configuration

- Learning Rate: 1e-5
- Weight Decay: 1e-5
- Epochs: 1
   
### Accuracy Results

| Quantization    | Input Size         | Top 1 Acc (%) | Top 5 Acc (%) |   
| --------------- |:------------------:|:-------------:|:-------------:|
| FP32            | (256, 3, 224, 224) | 76.2          | 92.9          |
| OwLite INT8 PTQ | (256, 3, 224, 224) | 75.9          | 92.8          |
| OwLite INT8 QAT | (256, 3, 224, 224) | 76.1          | 92.8          |
| INT8 TensorRT   | (256, 3, 224, 224) | 76.1          | 92.9          |

- INT8 TensorRT engine was build using applying FP16 and INT8 flags, further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide)

### Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size         | GPU Latency (ms) | 
| --------------- |:------------------:|:----------------:|
| FP16 TensorRT   | (256, 3, 224, 224) | 30.1             |
| OwLite INT8     | (256, 3, 224, 224) | 14.5             |
| INT8 TensorRT   | (256, 3, 224, 224) | 14.6             |
</details>

<details>
<summary>MobileNet-V2</summary>

### Configuration

#### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: Percentile (99.99)
  - QAT backward: CLQ
  - Gradient scales for weight quantization in {Conv, Gemm} were set to 0.01

#### Training Configuration

- Learning Rate: 2e-5
- Weight Decay: 1e-5
- Epochs: 1
  
### Accuracy Results

| Quantization    | Input Size         | Top 1 Acc (%) | Top 5 Acc (%) |   
| --------------- |:------------------:|:-------------:|:-------------:|
| FP32            | (256, 3, 224, 224) | 71.9          | 90.3          |
| OwLite INT8 PTQ | (256, 3, 224, 224) | 71.3          | 90.0          |
| OwLite INT8 QAT | (256, 3, 224, 224) | 71.7          | 90.2          |
| INT8 TensorRT   | (256, 3, 224, 224) | 70.6          | 89.6          |

- INT8 TensorRT engine was build using applying FP16 and INT8 flags, further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide)

### Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size         | GPU Latency (ms) | 
| --------------- |:------------------:|:----------------:|
| FP16 TensorRT   | (256, 3, 224, 224) | 11.2             |
| OwLite INT8     | (256, 3, 224, 224) | 6.17             |
| INT8 TensorRT   | (256, 3, 224, 224) | 6.29             |
</details>

<details>
<summary>MobileNet-V3-Large</summary>

### Configuration

#### Quantization Configuration

- Apply Owlite Recommended Config with the following calibration method
  - PTQ calibration: Percentile (99.99)
  - QAT backward: CLQ

#### Training Configuration

- Learning Rate: 2e-5
- Weight Decay: 1e-5
- Epochs: 5
  
### Accuracy Results

| Quantization    | Input Size         | Top 1 Acc (%) | Top 5 Acc (%) |   
| --------------- |:------------------:|:-------------:|:-------------:|
| FP32            | (256, 3, 224, 224) | 74.0          | 91.3          |
| OwLite INT8 PTQ | (256, 3, 224, 224) | 71.6          | 90.1          |
| OwLite INT8 QAT | (256, 3, 224, 224) | 72.9          | 90.7          |
| INT8 TensorRT   | (256, 3, 224, 224) | 71.4          | 90.0          |

- INT8 TensorRT engine was build using applying FP16 and INT8 flags, further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide)

### Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size         | GPU Latency (ms) | 
| --------------- |:------------------:|:----------------:|
| FP16 TensorRT   | (256, 3, 640, 640) | 11.84            |
| OwLite INT8     | (256, 3, 640, 640) | 6.77             |
| INT8 TensorRT   | (256, 3, 640, 640) | 6.82             |
</details>

<details>
<summary>EfficientNet-B0</summary>

### Configuration

#### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: MSE for weight quantization in {Conv, Gemm}, Percentile (99.9) for activation quantization  
  - QAT backward: CLQ
  - Gradient scales for weight quantization in {Conv, Gemm} were set to 0.01

#### Training Configuration

- Learning Rate: 5e-5
- Weight Decay: 1e-5
- Epochs: 5

### Accuracy Results

| Quantization    | Input Size        | Top 1 Acc (%) | Top 5 Acc (%) |   
| --------------- |:-----------------:|:-------------:|:-------------:|
| FP32            | (64, 3, 224, 224) | 77.7          | 93.6          |
| OwLite INT8 PTQ | (64, 3, 224, 224) | 73.3          | 91.4          |
| OwLite INT8 QAT | (64, 3, 224, 224) | 76.6          | 93.1          |
| INT8 TensorRT   | (64, 3, 224, 224) | 72.2          | 91.0          |

- INT8 TensorRT engine was build using applying FP16 and INT8 flags, further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide)

### Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | GPU Latency (ms) | 
| --------------- |:-----------------:|:----------------:|
| FP16 TensorRT   | (64, 3, 224, 224) | 6.45             |
| OwLite INT8     | (64, 3, 224, 224) | 3.09             |
| INT8 TensorRT   | (64, 3, 224, 224) | 3.27             |

</details>

<details>
<summary>EfficientNet-V2-S</summary>

### Configuration

#### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: Percentile (99.99)
  - QAT backward: CLQ
  - Gradient scales for weight quantization in {Conv, Gemm} were set to 0.01

#### Training Configuration

- Learning Rate: 2e-5
- Weight Decay: 1e-5
- Epochs: 2

### Accuracy Results

| Quantization    | Input Size        | Top 1 Acc (%) | Top 5 Acc (%) |   
| --------------- |:-----------------:|:-------------:|:-------------:|
| FP32            | (64, 3, 224, 224) | 81.3          | 95.3          |
| OwLite INT8 PTQ | (64, 3, 224, 224) | 80.3          | 94.9          |
| OwLite INT8 QAT | (64, 3, 224, 224) | 81.1          | 95.4          |
| INT8 TensorRT   | (64, 3, 224, 224) | 80.2          | 95.0          |

- INT8 TensorRT engine was build using applying FP16 and INT8 flags, further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide)

### Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | GPU Latency (ms) | 
| --------------- |:-----------------:|:----------------:|
| FP16 TensorRT   | (64, 3, 224, 224) | 12.3             |
| OwLite INT8     | (64, 3, 224, 224) | 6.54             |
| INT8 TensorRT   | (64, 3, 224, 224) | 6.83             |

</details>

<details>
<summary>Swin-B</summary>

### Configuration

#### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: MSE
  - QAT backward: CLQ
  - Gradient scales for weight quantization in {Conv, Gemm, Matmul} were set to 0.01

#### Training Configuration

- Learning Rate: 1e-5
- Weight Decay: 1e-5
- Epochs: 1

### Accuracy Results

| Quantization    | Input Size        | Top 1 Acc (%) | Top 5 Acc (%) |   
| --------------- |:-----------------:|:-------------:|:-------------:|
| FP32            | (64, 3, 224, 224) | 83.2          | 96.5          |
| OwLite INT8 PTQ | (64, 3, 224, 224) | 82.9          | 96.1          |
| OwLite INT8 QAT | (64, 3, 224, 224) | 83.0          | 96.3          |
| INT8 TensorRT   | (64, 3, 224, 224) | 83.2          | 96.5          |

- INT8 TensorRT engine was build using applying FP16 and INT8 flags, further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide)

### Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | GPU Latency (ms) | 
| --------------- |:-----------------:|:----------------:|
| FP16 TensorRT   | (64, 3, 224, 224) | 81.9             |
| OwLite INT8     | (64, 3, 224, 224) | 56.7             |
| INT8 TensorRT   | (64, 3, 224, 224) | 80.7             |

</details>

## Reference
https://github.com/pytorch/examples/blob/main/imagenet/main.py