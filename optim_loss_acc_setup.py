def accuracy(y_pred,y_true):
    corr = torch.eq(y_true,y_pred).sum().item()
    acc = corr/len(y_pred)*100
    return acc



optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.01, momentum = 0.9)

loss_function = nn.CrossEntropyLoss()
