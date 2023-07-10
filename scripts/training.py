def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()

    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Forward
        y_pred = model(X) # returns shape [32,3]

        # Loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        
        # optimizer
        optimizer.zero_grad()

        # backward
        loss.backward()

        # step
        optimizer.step()

        # accuracy across batch
        y_pred_class = torch.argmax(y_pred,dim=1)
        train_acc += accuracy_fn(y_pred_class, y)

    # get average loss and acc per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model, dataloader, loss_fn, device):
    model.eval()

    test_loss, test_acc = 0, 0 
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # forward
            y_pred = model(X)

            # loss
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            # accuracy across batch
            test_pred_label = torch.argmax(y_pred,dim=1)
            test_acc += accuracy_fn(test_pred_label, y)


    # get average loss and acc per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model, optimizer, loss_fn, epochs, device):
    # Create empty results dictionary
    results = {"train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
    }

    # Training loop
    for epoch in range(epochs):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
