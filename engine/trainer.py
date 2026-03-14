import torch


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        device,
        writer,
        config,
        evaluator
    ):

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.writer = writer
        self.config = config
        self.evaluate = evaluator

    def train(self):

        for epoch in range(self.config.epochs):

            self.model.train()

            total_loss = 0

            for images, labels in self.train_loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(images)

                loss = self.criterion(outputs, labels)

                loss.backward()

                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)

            acc = self.evaluate(self.model, self.test_loader, self.device)

            print(
                f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}"
            )

            self.writer.add_scalar("Loss/train", avg_loss, epoch)
            self.writer.add_scalar("Accuracy/test", acc, epoch)

        torch.save(
            self.model.state_dict(),
            self.config.model_save_path
        )

        print("Model saved!")