# OwLite Text-Recognition Example
- Model: VitSTR-Small
- Dataset: ICDAR2015

## Prerequisites

### Prepare dataset
Download and prepare [ICDAR2015](https://rrc.cvc.uab.es/?ch=4&com=introduction) with one of the options below:
- referring to the [README.md](https://github.com/roatienza/deep-text-recognition-benchmark?tab=readme-ov-file) from the original repository
- download directly from the [ICDAR2015 website](https://rrc.cvc.uab.es/?ch=4&com=introduction) and run *create_lmdb_dataset.py* after applying patch with the following example commands.
    ```
     python create_lmdb_dataset.py --inputPath /data/ICDAR2015/train/ --gtFile /data/ICDAR2015/train/gt.txt --outputPath /data/ICDAR2015/lmbd_train
    ```

    ```
     python create_lmdb_dataset.py --inputPath /data/ICDAR2015/test/ --gtFile /data/ICDAR2015/test/gt.txt --outputPath /data/ICDAR2015/lmbd_test
    ```



### Prepare model
Download and prepare target pretrained model's weight listed in the [original repository](https://github.com/roatienza/deep-text-recognition-benchmark?tab=readme-ov-file)
- example command

    ```
    wget https://github.com/roatienza/deep-text-recognition-benchmark/releases/download/v0.1.0/vitstr_small_patch16_224_aug.pth
    ```
    
### Apply patch
```
cd ViTSTR
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
python test.py --eval_data /data/ICDAR2015/lmbd_test/ --Transformer --sensitive --data_filtering_off --TransformerModel vitstr_small_patch16_224 --saved_model vitstr_small_patch16_224_aug.pth owlite --project <owlite_project_name> --baseline <owlite_baseline_name>
```

### Run quantized model
1. Create an experiment and save the config on OwLite GUI
2. Run the code for OwLite PTQ
    ```
    python test.py --eval_data /data/ICDAR2015/lmbd_test/ --Transformer --sensitive --data_filtering_off --TransformerModel vitstr_small_patch16_224 --saved_model vitstr_small_patch16_224_aug.pth owlite --project <owlite_project_name> --baseline <owlite_baseline_name> --experiment <owlite_experiment_name> --ptq
    ```
3. Run the code for OwLite QAT
    ```
    python train.py --train_data /data/ICDAR2015/lmbd_train/ --valid_data /data/ICDAR2015/lmbd_test/ --Transformer --TransformerModel vitstr_small_patch16_224 --sensitive --scheduler --saved_model vitstr_small_patch16_224_aug.pth owlite --project <owlite_project_name> --baseline <owlite_baseline_name> --duplicate-from best_qat --experiment <owlite_experiment_name> --qat
    ```

## Results

<details>
<summary>VitSTR-Small</summary>

### Configuration

#### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: Percentile (99.9) for weight quantization in {Conv, Gemm, MatMul}, MinMax for activation quantization 

### Accuracy and Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size         | Acc (%) | GPU Latency (ms) | 
| --------------- |:------------------:|:-------:|:----------------:|
| FP16 TensorRT   | (128, 1, 224, 224) | 71.58   | 22.93            |
| OwLite INT8 PTQ | (128, 1, 224, 224) | 70.02   | 14.88            |
| OwLite INT8 QAT | (128, 1, 224, 224) | 71.00   | 14.88            |
| INT8 TensorRT*  | (128, 1, 224, 224) | 71.58   | 22.93            |

- The INT8 TensorRT engine was built by applying FP16 and INT8 flags using [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy). However, the results were the same as those of the FP16 TensorRT engine, as the attempt to build with INT8 failed, leading to fallback to FP16 for all operations. Further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide).
</details>

## Reference
https://github.com/roatienza/deep-text-recognition-benchmark