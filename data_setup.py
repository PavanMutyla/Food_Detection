trian_path = "path"
test_path = "path"

train_data = ImageFolder(root = train_path,transform = auto_transform )

test_data = ImageFolder(root=test_path, transform = auto_transform)

#creating databatches
batch_size = 32
train_batch = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
test_batch = DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False)
class_names = train_data.classes
