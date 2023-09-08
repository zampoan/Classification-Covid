from timeit import default_timer as timer
import os
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import data_setup, save_model, training
import covid_aid, squeeze_net

# HYPERPARAMETERS
SEED=42
BATCH_SIZE=32
NUM_WORKERS=4 #os.cpu_count()
LEARNING_RATE={"CovidAid": 0.01, "SqueezeNet": 0.1}
EPOCHS = 5

# Instantiniate seeds
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Varaibles
train_dir = "../covid-dataset/train/"
test_dir = "../covid-dataset/test/"
data_transforms = {"CovidAid":transforms.Compose([transforms.Resize(256),
                                                  transforms.ToTensor()]),
                    "SqueezeNet": transforms.Compose([transforms.Resize(224),
                                                    transforms.ToTensor()])}
# Models
models = {"CovidAid": covid_aid.CovidAidModel().to(device), "SqueezeNet": squeeze_net.SqueezeNet().to(device)}

def training_loop(data_transforms, models):
    for model_name in models:
        print(f"Model Name: {model_name}")
        # DATA
        train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir, test_dir, data_transforms[model_name], BATCH_SIZE, NUM_WORKERS)

        # Loss Function and Optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(models[model_name].parameters(), lr=LEARNING_RATE[model_name])

        # Start timer
        start_time = timer()

        # Train models
        model_results = training.train(models[model_name], train_dataloader, test_dataloader, optimizer, loss_fn, EPOCHS, device)

        # End timer
        end_time = timer()
        print(f"{model_name} training time: {end_time-start_time:.2f} seconds")

        # Save Model
        save_model.save_model(models[model_name], target_dir='models', model_name=f'{model_name}.pt')

training_loop(data_transforms, models)
