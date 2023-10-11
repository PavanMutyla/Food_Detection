vit_weights = torchvision.models.ViT_B_32_Weights.DEFAULT

vit_model = torchvision.models.vit_b_32(weights = vit_weights)

vit_model = vit_model.to(device)


# summary of the ViT
summary(model = vit_model, input_size = (32,3,224,224), col_names=['input_size','output_size', 'num_params', 'trainable'], row_settings=['var_names'])

for para in vit_model.parameters():
    para.requires_grad = False


# fine tuning the vit model 
vit_model.heads = nn.Sequential(nn.Linear(in_features = 768, out_features = len(classes), bias = True))


# checking summary of the fine tuned model

summary(model = vit_model, inpout_size = (32,3,224,224), col_names['input_size', 'output_size', 'num_params', 'trainable'], row_settings=['var_names'])


# preparing data for the vit model
vit_train = ImageFolder(root = train_path, transform = vit_transform)
vit_test = ImageFolder(root = test_path, transform = vit_transform)


# preparing batches

vit_train_batch= DataLoader(vit_train, batch_size = 32, shuffle = True)

vit_test_batch = DataLoader(vit_test, batch_size =32, shuffle =False)


# training and testing model

train_model(model = vit_model, data = vit_train_batch, optimizer =vit_optim, loss_func= loss_func, accuracy = accuracy )

test_model(model = vit_model, data = vit_test_batch, loss_func = loss_func, accuracy = accuracy)


