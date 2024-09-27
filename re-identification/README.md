# OwLite Re-identification Example
- Model: Swin
- Dataset: DG-Market

## Prerequisites

### Prepare dataset
Download and prepare [Market1501 Dataset](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view) referring to the the [Dataset & Preparation](https://github.com/layumi/Person_reID_baseline_pytorch?tab=readme-ov-file#dataset--preparation) from the original repository.

### Apply patch
```
cd Person_reID_baseline_pytorch
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

### Train baseline model
```
python train.py --use_swin --name swin
```

### Run baseline model
```
python test.py --name swin owlite --project <owlite_project_name> --baseline <owlite_baseline_name>
```

### Run quantized model
1. Create an experiment and save the config on OwLite GUI
2. Run the code for OwLite PTQ 
    ```
    python test.py --name swin owlite --project <owlite_project_name> --baseline <owlite_baseline_name> --experiment <owlite_experiment_name> --ptq
    ```

## Results

<details>
<summary>Swin</summary>

### Configuration

#### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: MSE

### Accuracy and Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | Rank@1 | mAP (%) | GPU Latency (ms) | 
| --------------- |:-----------------:|:-----------------:|:------------:|:----------------:|
| FP16 TensorRT   | (64, 3, 224, 224) | 92.79              | 79.15         | 36.59             |
| OwLite INT8 PTQ | (64, 3, 224, 224) | 92.61              | 78.67         | 27.12             |
| INT8 TensorRT   | (64, 3, 224, 224) | 92.79              | 79.15         | 36.59             |

- The INT8 TensorRT engine was built by applying FP16 and INT8 flags using [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy). However, the results were the same as those of the FP16 TensorRT engine, as the attempt to build with INT8 failed, leading to fallback to FP16 for all operations. Further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide).
</details>

## Reference
https://github.com/layumi/Person_reID_baseline_pytorch.git