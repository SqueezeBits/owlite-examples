# OwLite Object Detection Example 
- Model: YOLOv5-S
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

## Reference
https://github.com/ultralytics/yolov5