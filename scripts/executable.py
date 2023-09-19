from timeit import default_timer as timer
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import data_setup, save_model, training
import covid_aid, squeeze_net, efficient_cnn, gru_cnn, codnnet

# HYPERPARAMETERS
SEED=42
BATCH_SIZE=32
NUM_WORKERS=4 #os.cpu_count()
LEARNING_RATE={
                "CodnNet": 0.001,
                "GRUCNN": 0.001,   
               "EfficientCNN": 0.01,
               "CovidAid": 0.001, 
               "SqueezeNet": 0.00001}
EPOCHS = 50

# Instantiniate seeds
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Varaibles
train_dir = "../covid-dataset/train/"
test_dir = "../covid-dataset/test/"
data_transforms = {
                    "CodnNet": transforms.Compose([transforms.Resize(256),
                                                  transforms.ToTensor()]),
                    "GRUCNN": transforms.Compose([transforms.Resize(224),
                                                  transforms.ToTensor()]),
                    "EfficientCNN": transforms.Compose([transforms.Resize(150),
                                                  transforms.ToTensor()]),
                    "CovidAid":transforms.Compose([transforms.Resize(256),
                                                  transforms.ToTensor()]),
                    "SqueezeNet": transforms.Compose([transforms.Resize(224),
                                                    transforms.ToTensor()])}
# Models
models = {
    "CodnNet": codnnet.CodnNet().to(device),
    "GRUCNN": gru_cnn.GRUCNN().to(device), 
          "EfficientCNN": efficient_cnn.EFFICIENT_CNN().to(device),
            "CovidAid": covid_aid.CovidAidModel().to(device), 
            "SqueezeNet": squeeze_net.SqueezeNet().to(device)}

optimizers =  {
    "CodnNet": torch.optim.SGD(models["CodnNet"].parameters(),lr=LEARNING_RATE["CodnNet"]),
    "GRUCNN": torch.optim.SGD(models["GRUCNN"].parameters(),lr=LEARNING_RATE["GRUCNN"]),
    "EfficientCNN": torch.optim.SGD(models["EfficientCNN"].parameters(), lr=LEARNING_RATE["EfficientCNN"]),
    "CovidAid": torch.optim.SGD(models["CovidAid"].parameters(), lr=LEARNING_RATE["CovidAid"]),
    "SqueezeNet": torch.optim.Adam(models["SqueezeNet"].parameters(), lr=LEARNING_RATE["SqueezeNet"])
}

loss_functions = {
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
        train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir, test_dir, data_transforms[model_name], BATCH_SIZE, NUM_WORKERS)

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

        # plot graph
        fig, ax = plt.subplots(1,2)
        x_axis = [i for i in range(EPOCHS)]
        train_loss= model_results["train_loss"]
        test_loss = model_results["test_loss"]
        train_acc = model_results["train_acc"]
        test_acc = model_results["test_acc"]

        ax[0, 0].plot(x_axis, train_loss, color="r", label="Train loss")
        ax[0, 0].plot(x_axis, test_loss, color="b", label="Test loss")
        ax[0, 0].set(xlabel="Epochs", ylabel="Loss")
        ax[0, 0].legend()

        ax[0, 1].plot(x_axis, train_acc, color="r", label="Train accuracy")
        ax[0, 1].plot(x_axis, test_acc, color="b", label="Test accuracy")
        ax[0, 1].set(xlabel="Epochs", ylabel="Accuracy")
        ax[0, 1].legend()
        fig.title(model_name)
        fig.savefig(f"loss_{model_name}")

training_loop(data_transforms, models)
