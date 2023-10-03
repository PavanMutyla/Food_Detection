# selecting a pre-trained ResNet-18 model
weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
auto_transform = weights.transforms()
