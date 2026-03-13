class Config:
    seed = 42

    batch_size = 64
    epochs = 10
    lr = 0.001
    num_workers = 4

    device = "cuda"

    model_save_path = "checkpoints/mnist_model.pt"

    log_dir = "runs/mnist"
