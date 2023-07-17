from timeit import default_timer as timer
import os
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import data_setup, covid_aid, save_model, training

# Varaibles
train_dir = "../covid-dataset/train/"
test_dir = "../covid-dataset/test/"
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor()
])

# HYPERPARAMETERS
SEED=42
BATCH_SIZE=32
NUM_WORKERS=4 #os.cpu_count()
LEARNING_RATE=0.01
EPOCHS = 5
print(f"Learning Rate: {LEARNING_RATE} | Number of Epochs: {EPOCHS}")

# Instantiniate seeds
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# DATA
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir, test_dir, data_transforms, BATCH_SIZE, NUM_WORKERS)

# MODELS
model_0 = covid_aid.CovidAidModel().to(device)

# Loss Function and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_0.parameters(), lr=LEARNING_RATE)

# Start timer
start_time = timer()

# Train models
model_0_results = training.train(model_0, train_dataloader, test_dataloader, optimizer, loss_fn, EPOCHS, device)

# End timer
end_time = timer()
print(f"Total training time: {end_time-start_time:.2f} seconds")

# Save Model
save_model.save_model(model_0, target_dir='models', model_name='CovidAid.pt')
