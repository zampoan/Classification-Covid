import os
import torch

def save_model(model, target_dir, model_name):
    # Create directory to save models
    target_dir = os.path.join("covid-dataset", target_dir)
    os.makedirs(target_dir, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth")
    model_saved_path = target_dir / model_name

    # save model
    print(f"Saved model to: {model_saved_path}")
    torch.save(model.state_dict(), model_saved_path)
