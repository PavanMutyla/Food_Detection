
mdoel = torchvision.models.resnet18.to(device)

# checking summary of the pre-trained model
pre_summ = summary(model=model, input_size = (32,3,224,224),col_names=["input_size", "output_size", "num_params", "trainable"],col_width=20,row_settings=["var_names"])

# freezing layer params / weights
for parameters in model.parameters():
    parameters.requires_grad = False


# fine-tuning the classifier/linear/output block of resnet18

model.fc #-> the linear block

torch.cuda.manual_seed(44)

model.fc = nn.Linear(in_features=512, out_features = out_shape, bias = True)


# checking the summary of the fine-tuned model
summary(model = model, input_size=(32,3,224,224), col_names=["input_size","output_size","num_params", "trainable"], col_width=20, row_settings=['var_names'])
