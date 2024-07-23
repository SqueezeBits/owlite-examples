# OwLite Object Detection Example 
- Model: YOLOv7
- Dataset: COCO'17 Dataset

## Prerequisites

### Prepare dataset
If you already have [COCO 2017 dataset](http://cocodataset.org), kindly modify the data path in within the "data/coco.yaml" file. Alternatively, executing the baseline will automatically download the dataset to the designated directory.

### Apply patch
```
cd yolov7
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
3. install OwLite package following the [installation guide](https://squeezebits.gitbook.io/owlite/user-guide/getting-started/install)

## How To Run

### Run baseline model
```
python test.py --data data/coco.yaml --img 640 --batch 16 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name exp --no-trace owlite --project <owlite_project_name> --baseline <owlite_baseline_name>
```

### Run quantized model
1. Create an experiment and save the config on OwLite GUI
2. Run the code for OwLite PTQ 
    ```
    python test.py --data data/coco.yaml --img 640 --batch 16 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name exp --no-trace owlite --project <owlite_project_name> --baseline <owlite_baseline_name> --experiment <owlite_experiment_name> --ptq
    ```

## Results

<details>
<summary>YOLOv7</summary>

### Configuration

#### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: MSE
    
### Accuracy and Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | mAP 0.50~0.95 (%) | mAP 0.50 (%) | GPU Latency (ms) | 
| --------------- |:-----------------:|:-----------------:|:------------:|:----------------:|
| FP16 TensorRT   | (16, 3, 640, 640) | 50.9              | 69.4         | 27.38            |
| OwLite INT8 PTQ | (16, 3, 640, 640) | 50.6              | 69.2         | 14.36            |
| INT8 TensorRT   | (16, 3, 640, 640) | 38.9              | 59.5         | 15.74            |

- The INT8 TensorRT engine was built by applying FP16 and INT8 flags using [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy), as further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide).
</details>

## Reference
https://github.com/WongKinYiu/yolov7
