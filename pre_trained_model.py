
link_pre_trained_model_pytorch = 'https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html#torchvision.models.resnet18'

model = torchvision.models.resnet18

link_weights =  'https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html#torchvision.models.ResNet18_Weights'

# getting summary of the  model using torchinfo.summary
summary(model = model, input_size=(32,3,224,224), col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])

# ouput of cells in notebook
