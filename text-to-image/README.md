# OwLite Stable Diffusion Example
- Model : SD-XL 1.0-base, SD v1.5
- task : text-to-image



## Prerequisites

### Setup environment
1. create conda env and activate it
    ```
    conda create -n <env_name> python=3.10 -y
    conda activate <env_name>
    ```
    Conda environment can be created with Python versions between 3.10 and 3.12 by replacing ```3.10``` with ```3.11``` or ```3.12```. Compatible Python versions for each PyTorch version can be found in [PyTorch compatibility matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix).

2. install OwLite package following the [installation guide](https://squeezebits.gitbook.io/owlite/user-guide/getting-started/install)
    

3. install required packages
    ```
    pip install -r requirements.txt
    ```

## How To Run

### Quick Start

### Run baseline model
```
python stable_diffusion_with_owlite.py [--model-version {1.5,xl}] [--cache-dir CACHE_DIR] [--n-steps N_STEPS] [--guidance GUIDANCE_SCALE] [--num-valid-images NUM_VALID_IMAGES] [--test-prompts TEST_PROMPTS] [--negative-prompt NEGATIVE_PROMPT] [--output-dir OUTPUT_DIR] [--device DEVICE] [--seed SEED] owlite --project OWLITE_PROJECT --baseline OWLITE_BASELINE
```
#### Example:
```
python stable_diffusion_with_owlite.py --test-prompts your/test/prompts.txt owlite --project StableDiffusion --baseline sdxl_base
```

### Run quantized model
1. Create an experiment and save the config on OwLite GUI
2. Run the code for OwLite PTQ
    ```
    python stable_diffusion_with_owlite.py [--model-version {1.5,xl}] [--cache-dir CACHE_DIR] [--n-steps N_STEPS] [--guidance GUIDANCE_SCALE] [--num-valid-images NUM_VALID_IMAGES] [--test-prompts TEST_PROMPTS] [--negative-prompt NEGATIVE_PROMPT] [--output-dir OUTPUT_DIR] [--device DEVICE] [--seed SEED] owlite --project OWLITE_PROJECT --baseline OWLITE_BASELINE --experiment OWLITE_EXPERIMENT_NAME --ptq [--duplicate-from OWLITE_DUPLICATE_FROM] [--calib-prompts OWLITE_CALIB_PROMPTS] [--ptq] [--download-engine]
    ```
    Example:
    ```
    python stable_diffusion_with_owlite.py --test-prompts your/test/prompts.txt owlite --project StableDiffusion --baseline sdxl_base --experiment exp --ptq --calib-prompts your/calib/prompts.txt
    ```

## Results

### Quantization Configuration

- Apply Conv OpType setting with the following calibration method
- PTQ calibration : MinMax for weight and MSE for input

### Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | GPU Latency (ms) |
| --------------- |:----------------:|
| FP16 TensorRT   | 151.2            |
| OwLite INT8 PTQ | 135.1            |

### Generated example images

![SDXL image](./asset/sdxl_base_quant.png)
The images above were generated with the SD XL baseline model, and the images below were generated with the quantised model.