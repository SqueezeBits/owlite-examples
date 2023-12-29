# OwLite Object Detection Example 

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
CUDA_VISIBLE_DEVICES=0 python tools/eval.py --specific-shape owlite --project <owlite_project_name> --baseline <owlite_baseline_name> 
```

### Run quantized model
1. Create an experiment and save the config on OwLite GUI
2. Run the code for OwLite PTQ 
    ```
    CUDA_VISIBLE_DEVICES=0 python tools/eval.py --specific-shape owlite --project <owlite_project_name> --baseline <owlite_baseline_name> --experiment <owlite_experiment_name> --ptq
    ```

## Results

### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: Percentile (99.99%)

    
### Accuracy Results

| Quantization    | Input Size        | mAP 0.50~0.95 (%) | mAP 0.50 (%) |   
| --------------- |:-----------------:|:-----------------:|:------------:|
| FP32            | (32, 3, 640, 640) | 44.7              | 61.6         |
| OwLite INT8 PTQ | (32, 3, 640, 640) | 41.9              | 58.1         |
| INT8 TensorRT   | (32, 3, 640, 640) | 7.3               | 11.3         |

- INT8 TensorRT engine was build using applying FP16 and INT8 flags, further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide)

### Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | GPU Latency (ms) | 
| --------------- |:-----------------:|:----------------:|
| FP16 TensorRT   | (32, 3, 640, 640) | 17.3             |
| OwLite INT8 PTQ | (32, 3, 640, 640) | 8.6              |
| INT8 TensorRT   | (32, 3, 640, 640) | 8.7              | 


## Reference
https://github.com/meituan/YOLOv6.git
