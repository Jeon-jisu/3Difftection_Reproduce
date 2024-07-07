import torch
from torch.utils.data import DataLoader
from models import SemanticControlNet
from utils.data_loader import SemanticDataset
from utils.preprocessing import preprocess_semantic
import yaml


def train_semantic_controlnet(args):
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Initialize model
    model = SemanticControlNet(config["model"])
    model = model.to(config["device"])

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Initialize dataset and dataloader
    dataset = SemanticDataset(config["data_path"], transform=preprocess_semantic)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Training loop
    for epoch in range(config["num_epochs"]):
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(config["device"]), targets.to(config["device"])

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {loss.item()}")

    # Save model
    torch.save(model.state_dict(), config["save_path"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train_semantic_controlnet(args)
