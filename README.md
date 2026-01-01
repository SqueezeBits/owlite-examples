![OwLite logo](https://github.com/SqueezeBits/owlite/assets/64083281/abaa3ad9-0c86-4a9c-9b8d-f54ed6d9524b)

<div align="center">
<p align="center">
  <a href="https://www.squeezebits.com/">Website</a> •
  <a href="https://owlite.ai/">Web UI</a> •
  <a href="#environment-setup">Environment setup</a> •
  <a href="#available-tasks">Available tasks</a> •
  <a href="#contact">Contact</a>
</p>
<p align="center">
  <a href="https://github.com/SqueezeBits/owlite/releases"><img src="https://img.shields.io/github/v/release/SqueezeBits/owlite?color=EE781F" /></a>
  <a href="https://squeezebits.gitbook.io/owlite/quick/readme"><img src="https://img.shields.io/badge/Documentation-FFA32C" /></a>
  <a href="https://github.com/SqueezeBits/owlite#installation"><img src="https://img.shields.io/badge/python-≥3.10, <3.13-blue" /></a>
  <a><img src="https://img.shields.io/badge/pytorch-≥2.1.2, <2.5-blue" /></a>
</p>
</div>

# OwLite Examples

```OwLite Examples``` repository offers illustrative example codes to help users seamlessly compress PyTorch deep learning models and transform them into TensorRT engines. This is achieved through the OwLite package, which requires only slight modification to current codebases. Most of the examples are based on other public repositories to show how easy it is to apply OwLite compression on the existing codes. You can also check the powerful results of OwLite compression, as demonstrated by numerous examples. 

## Getting Started

First clone this repository and it's included submodules.

    git clone --recurse-submodules https://github.com/SqueezeBits/owlite-examples.git

Start up a docker instance using the provided `compose.yaml`.

	./start_owlite

Log into Owlite. An account can be created at [owlite.ai](https://owlite.ai/auth/register).

	owlite login

	Enter your email: user@owlite.ai
	Enter yout password:

After successful login, the following message will be displayed.
		
	OwLite [INFO] Logged in as <user>
	OwLite [INFO] Your price plan: <PLAN TYPE>
	OwLite [INFO] Your workgroup: <workgroup>
	OwLite [INFO] Your authentication token is saved at /home/owlet/.cache/owlite/tokens

Head over to the [introduction](introduction/README.md) to begin using OwLite with ResNet18.

## Available Tasks
- [Image classification](image-classification/README.md)
- [Object detection](object-detection/README.md)
- [Image segmentation](image-segmentation/README.md)
- [Text classification](text-classification/README.md)
- [Re-identification](re-identification/README.md)
- [Face landmark](face-landmark/README.md)
- [Pose estimation](pose-estimation/README.md)
- [Text-to-image](text-to-image/README.md)
- [Text recognition](text-recognition/README.md)

List of available tasks will be continuously updated.

## Contact
Please contact owlite-admin@squeezebits.com for any questions or suggestions.

<br>
<br>
<div align="center"><img src="https://github.com/SqueezeBits/owlite/assets/64083281/bdbf55a6-ead7-42d3-b0b7-f1e8eddfb558" width="300px"></div>