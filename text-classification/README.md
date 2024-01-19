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
    ```
## How to Run

### 1. Finetune pretrained languange model 

    CUDA_VISIBLE_DEVICES=0 python owlite_run_glue.py --model_name_or_path bert-base-cased --task_name <task_name> --do_eval --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 --output_dir <finetuned_model_dir> --do_finetuning

- For wnli, you use the --learning_rate=5e-6 instead of 2e-5.
    <task_name> is one of {mrpc, cola, stsb, wnli, qqp, sst2, mnli, qnli, rte}.

### 2. Run baseline model
    CUDA_VISIBLE_DEVICES=0 python owlite_run_glue.py --model_name_or_path <finetuned_model_dir> --task_name <task_name> --do_eval --max_seq_length 128 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --output_dir <output_dir> --dataloader_drop_last true --project <owlite_project_name> --baseline <owlite_baseline_name>

### 3-1. Run post-training quantization (PTQ)
    
    CUDA_VISIBLE_DEVICES=0 python owlite_run_glue.py --model_name_or_path <finetuned_model_dir> --task_name <task_name> --do_eval --max_seq_length 128 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --output_dir <output_dir> --dataloader_drop_last true --project <owlite_project_name> --baseline <owlite_baseline_name> --experiment <owlite_experiment_name> --ptq
    
### 3-2. Run quantization-aware training (QAT)
    
    CUDA_VISIBLE_DEVICES=0 python owlite_run_glue.py --model_name_or_path <finetuned_model_dir> --task_name <task_name> --do_eval --max_seq_length 128 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --output_dir <output_dir> --dataloader_drop_last true --learning_rate 1e-6 --project <owlite_project_name> --baseline <owlite_baseline_name> --experiment <owlite_experiment_name> --qat
    
## Results

### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
    - PTQ calibration: Require to find the optimal calibration method for each node.
    - QAT backward: CLQ
    - Hyperparameter setting for QAT
        - learning_rate:
            - 1e-7 for RTE task
            - 5e-6 for SST-2, STS-B, WNLI tasks
            - 1e-6 for other tasks
        - num_train_epochs:
            - 10 for CoLA task
            - 5 for QQP, QNLI tasks
            - 3 for other tasks
    
### GLUE Results

| Quantization    | Input Size | CoLA (Matthews corr.) | SST-2 | MRPC (Acc./F1) | STS-B (Pearson/Spearman corr.)| QQP (Acc./F1)| MNLI (Matched/Mismatched Acc.) | QNLI  | RTE   | WNLI  |  
| --------------- |:----------:|:---------------------:|:-----:|:--------------:|:-----------------------------:|:------------:|:------------------------------:|:-----:|:-----:|:-----:|
| FP32            | (8x128)    | 61.13                 | 91.51 | 84.31/89.08    | 87.85/87.60                   | 90.59/87.29  | 83.97/84.45                    | 90.72 | 65.43 | 57.14 | 
| OwLite INT8 PTQ | (8x128)    | 59.25                 | 91.17 | 82.35/87.67    | 86.54/86.56                   | 89.06/85.72  | 83.00/82.57                    | 89.50 | 65.06 | 57.14 | 
| OwLite INT8 QAT | (8x128)    | 59.63                 | 91.17 | 84.07/88.85    | 86.95/87.05                   | 88.90/85.86  | 82.95/83.12                    | 89.59 | 65.06 | 57.14 | 


### Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size | Latency (ms) |  
| --------------- |:----------:|:------------:|
| FP16 TensorRT   | (8x128)    | 2.32         |
| OwLite INT8     | (8x128)    | 1.65         |

### References
https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification