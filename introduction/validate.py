import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from torchvision.models import resnet18

import numpy as np
from tqdm import tqdm


def accuracy(output, target, topk=(1,)): # https://github.com/pytorch/examples/blob/main/imagenet/main.py
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def validate(model, dataloader):
    model.eval()
    avg_top1 = []
    with torch.no_grad():
        for images, target in tqdm(dataloader, desc="Validating model"):
            output = model(images.to(device))
            avg_top1.append(np.array(accuracy(output.cpu(), target)).mean())

    print("Accuracy:", np.array(avg_top1).mean())
    return float(np.array(avg_top1).mean())



# Image transformation setup
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize])

val_dataset = ImageNet(root='./data', 
                       split='val',
                       transform=transforms)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=256,
                                         shuffle=False,
                                         num_workers=4,
                                         pin_memory=True)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = resnet18(weights="DEFAULT")
model.to(device)
print(f"[INFO] ResNet18 Loaded to {device}")


import owlite

experiment = None
# experimnet = "first_quantization"

owl = owlite.init(project="Introduction", baseline="resnet18_val", experiment=experiment)

dummy_input = torch.randn(64, 3, 224, 224)
model = owl.convert(model, dummy_input)

if experiment: # Quantized models required some data to be quantized accurately
    with owlite.calibrate(model) as calibrate_model:
        for images, _ in val_loader:
            calibrate_model(images.to(device))
            break # A single batch of images is sufficient
    
owl.export(model) # Export model to ONNXß
owl.benchmark() # Benchmark model
owl.log(accuracy=validate(model, val_loader)) # Add accuracy to OwLite dashboard