
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import data_setup, save_model, training
import covid_aid, squeeze_net, efficient_cnn, gru_cnn, codnnet, bobnet
import torchmetrics

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
accuracy_fn = torchmetrics.Accuracy(task="multiclass", num_classes=3).to(device)

# HYPERPARAMETERS
SEED=42
BATCH_SIZE=32
NUM_WORKERS=4 #os.cpu_count()
LEARNING_RATE={ # Great results at 0.01
                "BobNet": 0.01,
                "CodnNet": 0.01, 
                "GRUCNN": 0.01,   
               "EfficientCNN": 0.01,
               "CovidAid": 0.01, 
               "SqueezeNet": 0.01
}
EPOCHS = 5

# Instantiniate seeds
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


# Varaibles
train_dir = "../covid-dataset/train/"
test_dir = "../covid-dataset/test/"
gen_dir = "../secondary-covid-dataset/"
data_transforms = {
                    "BobNet": transforms.Compose([transforms.Resize(256),
                                                  transforms.ToTensor()]),
                    "CodnNet": transforms.Compose([transforms.Resize(256),
                                                  transforms.ToTensor()]),
                    "GRUCNN": transforms.Compose([transforms.Resize(224),
                                                  transforms.ToTensor()]),
                    "EfficientCNN": transforms.Compose([transforms.Resize(150),
                                                  transforms.ToTensor()]),
                    "CovidAid":transforms.Compose([transforms.Resize(256),
                                                  transforms.ToTensor()]),
                    "SqueezeNet": transforms.Compose([transforms.Resize(224),
                                                    transforms.ToTensor()])
}
# Models
models = {
    "BobNet": bobnet.BobNet().to(device),
    "CodnNet": codnnet.CodnNet().to(device),
    "GRUCNN": gru_cnn.GRUCNN().to(device), 
          "EfficientCNN": efficient_cnn.EFFICIENT_CNN().to(device),
            "CovidAid": covid_aid.CovidAidModel().to(device), 
            "SqueezeNet": squeeze_net.SqueezeNet().to(device)
}

optimizers =  {
    "BobNet": torch.optim.SGD(models["BobNet"].parameters(),lr=LEARNING_RATE["BobNet"]),
    "CodnNet": torch.optim.SGD(models["CodnNet"].parameters(),lr=LEARNING_RATE["CodnNet"]),
    "GRUCNN": torch.optim.SGD(models["GRUCNN"].parameters(),lr=LEARNING_RATE["GRUCNN"]),
    "EfficientCNN": torch.optim.SGD(models["EfficientCNN"].parameters(), lr=LEARNING_RATE["EfficientCNN"]),
    "CovidAid": torch.optim.SGD(models["CovidAid"].parameters(), lr=LEARNING_RATE["CovidAid"]),
    "SqueezeNet": torch.optim.Adam(models["SqueezeNet"].parameters(), lr=LEARNING_RATE["SqueezeNet"])
}

loss_functions = {
            "BobNet": nn.CrossEntropyLoss(),
            "CodnNet": nn.CrossEntropyLoss(),
                "GRUCNN": nn.CrossEntropyLoss(),
               "EfficientCNN": nn.CrossEntropyLoss(),
               "CovidAid": nn.CrossEntropyLoss(), 
               "SqueezeNet": nn.CrossEntropyLoss()

}


def training_loop(data_transforms, models):
    for model_name in models:
        print(f"Model Name: {model_name}")
        # DATA
        train_dataloader, test_dataloader, _ , class_names = data_setup.create_dataloaders(train_dir, test_dir, gen_dir, data_transforms[model_name], BATCH_SIZE, NUM_WORKERS)

        # Loss Function and Optimizer
        loss_fn = loss_functions[model_name]
        optimizer = optimizers[model_name]

        # Start timer
        start_time = timer()

        # Train models
        model_results = training.train(models[model_name], train_dataloader, test_dataloader, optimizer, loss_fn, EPOCHS, device)

        # End timer
        end_time = timer()

        # Misc
        print(f"{model_name} training time: {end_time-start_time:.2f} seconds")
        pytorch_total_params = sum(p.numel() for p in models[model_name].parameters())
        print(f"Number of parameters: {pytorch_total_params}")

        # Save Model
        save_model.save_model(models[model_name], target_dir='models', model_name=f'{model_name}.pt')
        
        # Plot graphs
        plot(model_name, model_results)

def plot(model_name, model_results):
    x_axis = [i for i in range(EPOCHS)]
    train_loss= model_results["train_loss"]
    test_loss = model_results["test_loss"]
    train_acc = [tensor.cpu().numpy() for tensor in model_results["train_acc"]]
    test_acc = [tensor.cpu().numpy() for tensor in model_results["test_acc"]]
    f1_score = [tensor.cpu().numpy() for tensor in model_results["f1_score"]]
    recall = [tensor.cpu().numpy() for tensor in model_results["recall"]]
    precision = [tensor.cpu().numpy() for tensor in model_results["precision"]]

    # Loss
    fig, (ax1, ax2 )= plt.subplots(1,2, figsize=(15, 10))
    ax1.plot(x_axis, train_loss, color="r", label="Train loss")
    ax1.plot(x_axis, test_loss, color="b", label="Test loss")
    ax1.set(xlabel="Epochs", ylabel="Loss")
    ax1.legend()
    
    # Accuracy
    ax2.plot(x_axis, train_acc, color="r", label="Train accuracy")
    ax2.plot(x_axis, test_acc, color="b", label="Test accuracy")
    ax2.set(xlabel="Epochs", ylabel="Accuracy")
    ax2.legend()
    fig.suptitle(model_name)
    fig.savefig(f"z_{model_name}") 
    
    # Display F1, recall and precision
    fig, ax = plt.subplots()
    ax.plot(x_axis, f1_score, color='r', label="f1_score")
    ax.plot(x_axis, recall, color='g', label="recall")
    ax.plot(x_axis, precision, color='b', label="precision")
    ax.legend()
    fig.suptitle(model_name)
    fig.savefig(f"metric_{model_name}")

    
def generalise(models):
    for model_name in models:
        print(f"Model Name: {model_name}")
        model = models[model_name]
        model.load_state_dict(torch.load(f'/users/adbr117/Covid-Classification/Covid-Classificaton/models/{model_name}.pt'))
        model.eval()
        
        _, _, gen_dataloader, _ = data_setup.create_dataloaders(train_dir, test_dir, gen_dir, data_transforms[model_name], BATCH_SIZE, NUM_WORKERS)
        
        loss_fn = loss_functions[model_name]
        optimizer = optimizers[model_name]
        
        for epoch in range(EPOCHS):
            gen_loss, gen_acc = 0, 0 
            with torch.inference_mode():
                for batch, (X, y) in enumerate(gen_dataloader):
                    X, y = X.to(device), y.to(device)

                    # forward
                    y_pred = model(X)

                    # loss
                    loss = loss_fn(y_pred, y)
                    gen_loss += loss.item()

                    # accuracy across batch
                    gen_pred_label = torch.argmax(y_pred,dim=1)
                    gen_acc += accuracy_fn(gen_pred_label, y)

            # get average loss and acc per batch
            gen_loss = gen_loss / len(gen_dataloader)
            gen_acc = gen_acc / len(gen_dataloader)
                
            print(f"Epoch {epoch} | Gen Loss: {gen_loss:.4f} | Gen Accuracy: {gen_acc:.2f}")
        
training_loop(data_transforms, models)
generalise(models)
