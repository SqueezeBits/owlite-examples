# OwLite Face Landmark Example 
- Model: PIPNet-ResNet18
- Dataset: WFLW Dataset

## Prerequisites

### Prepare dataset
Download [WFLW](https://wywu.github.io/projects/LAB/WFLW.html) dataset to `data/WFLW`, and preprocess the dataset using `python preprocess.py WFLW`. Detailed explanation can be found in the original [repo](https://github.com/jhb86253817/PIPNet#supervised-learning)

### Apply patch
```
cd PIPNet
patch -p1 < ../apply_owlite.patch
```
### Download checkpoint
Create two folders `logs` and `snapshots`, and download checkpoint to `snapshots` following the original [repo](https://github.com/jhb86253817/PIPNet#demo).

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
3. install OwLite package following the [installation guide](https://squeezebits.gitbook.io/owlite/user-guide/getting-started/install)

## How To Run

### Run baseline model
```
python lib/test.py --cfg-file experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py owlite --project <owlite_project_name> --baseline <owlite_baseline_name> 
```
### Run quantized model
1. Create an experiment and save the config on OwLite GUI
2. Run the code for OwLite PTQ 
    ```
    python lib/test.py --cfg-file experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py owlite --project <owlite_project_name> --baseline <owlite_baseline_name> --experiment <owlite_experiment_name> --ptq
    ```
## Results

<details>
<summary>PIPNet-ResNet18</summary>

### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: MSE

    
### Accuracy and Lateny Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | NME | FR | AUC | GPU Latency (ms) | 
| --------------- |:-----------------:|:-----------------:|:------------:|:----------------:|:----------------:|
| FP16 TensorRT   | (16, 3, 256, 256) | 4.57 | 4.48 | 56.3 | 1.09 |
| OwLite INT8 PTQ | (16, 3, 256, 256) | 4.58 | 4.39 | 56.2 | 0.55 |
| INT8 TensorRT   | (16, 3, 256, 256) | 4.57 | 4.60 | 56.3 | 0.61 |

- The INT8 TensorRT engine was built by applying FP16 and INT8 flags using [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy), as further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide).
</details>

## Reference
https://github.com/jhb86253817/PIPNet