import torch
import torch.nn as nn
import torch.optim as optim

from configs.config import Config
from datasets.mnist_dataset import get_mnist_dataloader
from models.cnn import SimpleCNN
from utils.seed import set_seed

#假如TensorBoard

from utils.logger import get_writer


from engine.evaluator import evaluate

def train():

    config = Config()

    set_seed(config.seed)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_mnist_dataloader(
        config.batch_size,
        config.num_workers
    )

    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr
    )

    #step 2
    writer = get_writer(config.log_dir)

    for epoch in range(config.epochs):

        model.train()

        total_loss = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        #print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        #平均损失
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        writer.add_scalar("Loss/train", avg_loss, epoch)
        #准确率计算
        acc = evaluate(model, test_loader, device)

        print(f"Test Accuracy: {acc:.4f}")

        writer.add_scalar("Accuracy/test", acc, epoch)



    torch.save(model.state_dict(), config.model_save_path)

    print("Model saved!")


if __name__ == "__main__":
    train()