import torch
import torch.nn as nn
import torch.optim as optim

from configs.config import Config
from datasets.mnist_dataset import get_mnist_dataloader
# from models.cnn import SimpleCNN
from models.cnn import LeNet
from utils.seed import set_seed
#假如TensorBoard
from utils.logger import get_writer
from engine.evaluator import evaluate
from engine.trainer import Trainer

def main():

    config = Config()

    set_seed(config.seed)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_mnist_dataloader(
        config.batch_size,
        config.num_workers
    )
    #week1  simpleCNN
    # model = SimpleCNN().to(device)
    #week2  LeNet
    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr
    )

    #step 2
    writer = get_writer(config.log_dir)

    trainer = Trainer(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        device,
        writer,
        config,
        evaluate
    )
    trainer.train()


if __name__ == "__main__":
    main()