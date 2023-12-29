# OwLite Text Classification Example
- Model: BERT-base
- Dataset: GLUE Dataset
## Prerequisites

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
4. Apply patch & install transformer package
    ```
    cd transformers
    patch -p1 < ../apply_owlite.patch

    pip install -e .
    cd ..
    ```
## How to Run

### 1. Finetune pretrained languange model 

    CUDA_VISIBLE_DEVICES=0 python run_glue.py --model_name_or_path bert-base-cased --task_name <task_name> --do_eval --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 --output_dir <finetuned_model_dir> --do_finetuning

- For wnli, you use the --learning_rate=5e-6 instead of 2e-5.
    <task_name> is one of {mrpc, cola, stsb, wnli, qqp, sst2, mnli, qnli, rte}.

### 2. Run baseline model
    CUDA_VISIBLE_DEVICES=0 python run_glue.py --model_name_or_path <finetuned_model_dir> --task_name <task_name> --do_eval --max_seq_length 128 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --output_dir <finetuned_model_dir> --dataloader_drop_last true --project <owlite_project_name> --baseline <owlite_baseline_name>

### 3-1. Run post-training quantization (PTQ)
    
    CUDA_VISIBLE_DEVICES=0 python run_glue.py --model_name_or_path <finetuned_model_dir> --task_name <task_name> --do_eval --max_seq_length 128 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --output_dir <output_dir> --dataloader_drop_last true --project <owlite_project_name> --baseline <owlite_baseline_name> --experiment <owlite_experiment_name> --ptq
    
### 3-2. Run quantization-aware training (QAT)
    
    CUDA_VISIBLE_DEVICES=0 python run_glue.py --model_name_or_path <finetuned_model_dir> --task_name <task_name> --do_eval --max_seq_length 128 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --learning_rate 2e-5 --output_dir <output_dir> --dataloader_drop_last true --project <owlite_project_name> --baseline <owlite_baseline_name> --experiment <owlite_experiment_name> --qat
    
## Results

### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
    - PTQ calibration: 
        - MSE for MRPC task
        - Percentile (99.99%) for other tasks
    - QAT backward: CLQ
    - Hyperparameter setting for QAT
        - learning_rate: 
            - 1e-4 for CoLA task
            - 5e-4 for MRPC task
            - 5e-6 for other tasks
        - weight_decay: 0.0
        - num_train_epochs
            - 10 for MRPC task
            - 3 for other tasks
    
### GLUE Results

| Quantization    | Input Size        | CoLA (Matthews corr.) | SST-2 | MRPC (Acc./F1) | STS-B (Pearson/Spearman corr.)| QQP (F1/Acc.)| MNLI (Matched/Mismatched Acc.) | QNLI | RTE | WNLI |  
| --------------- |:-----------------:|:-----------------:|:------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| FP32            | (8x128) | 61.13 | 90.83 | 84.56/89.04 | 89.30/88.96 | 87.21/90.56 | 83.87/84.40 | 90.72 | 65.43 | 57.14 | 
| OwLite INT8 PTQ | (8x128) | 57.17 | 91.17 | 68.38/71.90 | 86.58/88.97 | 85.78/88.86 | 82.43/82.46 | 87.90 | 63.57 | 57.14 | 
| OwLite INT8 QAT | (8x128) | 57.24 | 91.17 | 78.43/83.94 | 87.49/86.92 | 85.88/88.83 | 83.04/82.76 | 88.03 | 63.94 | 57.14 | 


### Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | Latency (ms) |  
| --------------- |:-----------------:|:-----------------:|
| FP16 TensorRT   | (8x128) | 2.32 |
| OwLite INT8     | (8x128) | 1.66 |

### References
https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification
