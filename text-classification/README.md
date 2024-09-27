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
    Conda environment can be created with Python versions between 3.10 and 3.12 by replacing ```3.10``` with ```3.11``` or ```3.12```. Compatible Python versions for each PyTorch version can be found in [PyTorch compatibility matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix).

2. install required packages
    ```
    pip install -r requirements.txt
    ```
3. install OwLite package following the [installation guide](https://squeezebits.gitbook.io/owlite/user-guide/getting-started/install)

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
    
### GLUE Accuracy and Latency Results
TensorRT Evaluation GPU: A6000

| Quantization  | Input Size | CoLA (Matthews corr.) | SST-2 | MRPC (Acc./F1) | STS-B (Pearson/Spearman corr.)| QQP (Acc./F1)| MNLI (Matched/Mismatched Acc.) | QNLI  | RTE   | WNLI  | Latency (ms) |  
| ------------- |:----------:|:---------------------:|:-----:|:--------------:|:-----------------------------:|:------------:|:------------------------------:|:-----:|:-----:|:-----:|:------------:|
| FP16 TensorRT | (8x128)    | 61.13                 | 91.51 | 84.31/89.08    | 87.85/87.60                   | 90.59/87.29  | 83.97/84.45                    | 90.72 | 65.43 | 57.14 | 2.32         | 
| OwLite INT8   | (8x128)    | 57.88                 | 91.51 | 82.84/87.97    | 87.35/87.16                   | 89.33/86.03  | 83.43/83.23                    | 89.74 | 64.68 | 57.14 | 1.65         |
| INT8 TensorRT*| (8x128)    | 61.13                 | 91.51 | 84.31/89.08    | 87.85/87.60                   | 90.59/87.29  | 83.97/84.45                    | 90.72 | 65.43 | 57.14 | 2.32         |

- The INT8 TensorRT engine was built by applying FP16 and INT8 flags using [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy). However, the results were the same as those of the FP16 TensorRT engine, as the attempt to build with INT8 failed, leading to fallback to FP16 for all operations. Further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide).

### References
https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification