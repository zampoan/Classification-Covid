from timeit import default_timer as timer
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import data_setup, save_model, training
import covid_aid, squeeze_net, efficient_cnn, gru_cnn

# HYPERPARAMETERS
SEED=42
BATCH_SIZE=32
NUM_WORKERS=4 #os.cpu_count()
LEARNING_RATE={
                "GRUCNN": 0.001,   # Adam 0.0001
               "EfficientCNN": 0.01,
               "CovidAid": 0.001, 
               "SqueezeNet": 0.001}
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
    "GRUCNN": gru_cnn.GRUCNN().to(device), 
          "EfficientCNN": efficient_cnn.EFFICIENT_CNN().to(device),
            "CovidAid": covid_aid.CovidAidModel().to(device), 
            "SqueezeNet": squeeze_net.SqueezeNet().to(device)}

optimizers =  {
    "GRUCNN": torch.optim.SGD(models["GRUCNN"].parameters(),lr=LEARNING_RATE["GRUCNN"]),
    "EfficientCNN": torch.optim.SGD(models["EfficientCNN"].parameters(), lr=LEARNING_RATE["EfficientCNN"]),
    "CovidAid": torch.optim.SGD(models["CovidAid"].parameters(), lr=LEARNING_RATE["CovidAid"]),
    "SqueezeNet": torch.optim.SGD(models["SqueezeNet"].parameters(), lr=LEARNING_RATE["SqueezeNet"])
}


def training_loop(data_transforms, models):
    for model_name in models:
        print(f"Model Name: {model_name}")
        # DATA
        train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir, test_dir, data_transforms[model_name], BATCH_SIZE, NUM_WORKERS)

        # Loss Function and Optimizer
        loss_fn = nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(models[model_name].parameters(), lr=LEARNING_RATE[model_name])
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
        fig, ax = plt.subplots()
        x_axis = [i for i in range(EPOCHS)]
        y_axis = model_results["train_loss"]
        ax.plot(x_axis, y_axis)
        ax.set(xlabel="Epochs", ylabel="Train Loss", title=model_name)
        fig.savefig(f"testloss_{model_name}")

training_loop(data_transforms, models)
