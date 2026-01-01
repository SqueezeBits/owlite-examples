# Loading ResNet18 from torchvision models
import torch
from torchvision.models import resnet18

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = resnet18(weights="DEFAULT")
model.to(device)
print(f"[INFO] ResNet18 Loaded to {device}")


# Analyze ResNet18 with OwLite wrapper
import owlite

experiment = None
# experimnet = "first_quantization"
owl = owlite.init(project="Introduction", baseline="resnet18_val", experiment=experiment)

# Wrap model with OwLite convert and provide a dummy input
dummy_input = torch.randn(64, 3, 224, 224)
model = owl.convert(model, dummy_input)

owl.export(model) # Export model to ONNX
owl.benchmark() # Benchmark model
