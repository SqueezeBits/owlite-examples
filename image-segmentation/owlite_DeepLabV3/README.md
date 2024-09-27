# OwLite Image Segmentation (DeepLabV3) Example
- Model : Deeplabv3_Resnet50  
- Dataset : VOC2012 Dataset (Augmented)

## Prerequisites

### Prepare dataset
Running the code with `--download` flag will automatically download the data ([Guide](https://github.com/VainF/DeepLabV3Plus-Pytorch#22--pascal-voc-trainaug-recommended))

### Apply patch
```
cd DeepLabV3Plus-Pytorch
patch -p1 < ../apply_owlite.patch
```

### Setup environment
1. create conda env and activate
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

### Download checkpoint
Download checkpoint from the original [repo](https://github.com/VainF/DeepLabV3Plus-Pytorch#1-performance-on-pascal-voc2012-aug-21-classes-513-x-513)
```
wget https://www.dropbox.com/s/3eag5ojccwiexkq/best_deeplabv3_resnet50_voc_os16.pth
```

## How To Run
### Run baseline model
```
python main.py --model deeplabv3_resnet50 --ckpt <checkpoint_path> --data_root <data_path> --gpu_id 0 owlite --project <owlite_project_name> --baseline <owlite_baseline_name>
```

### Run quantized model
1. Create an experiment and save the configuration on OwLite GUI
2. Run the code
    - For PTQ
        ```
        python main.py --model deeplabv3_resnet50 --ckpt <checkpoint_path> --data_root <data_path> --gpu_id 0 owlite --project <owlite_project_name> --baseline <owlite_baseline_name> --experiment <owlite_experiment_name> --ptq
        ```

## Results

<details>
<summary>Deeplabv3_Resnet50</summary>

### Quantization Configuration
- Apply OwLite Recommended Configuration with the following calibration method

- PTQ calibration : MSE

### Accuracy and Latency Results
Evaluation GPU: A6000

| Quantization    | Input Size         | mIoU  | GPU Latency (ms) |   
| --------------- |:-----------------:|:------:|:----------------:|
| FP16 TensorRT   | (16, 3, 513, 513) | 0.7674 | 19.657           |
| OwLite INT8 PTQ | (16, 3, 513, 513) | 0.7683 | 9.823            |
| TensorRT INT8   | (16, 3, 513, 513) | 0.7649 | 9.830            |

- The INT8 TensorRT engine was built by applying FP16 and INT8 flags using [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy), as further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide).
</details>

## Reference

DeepLabV3Plus-Pytorch (https://github.com/VainF/DeepLabV3Plus-Pytorch)
