# MNIST CNN 入门项目

这是一个用于学习卷积神经网络基础流程的 MNIST 手写数字分类项目。项目从最基础的 CNN 发展到 LeNet，并配合训练、评估、日志记录和模型保存，便于理解深度学习项目的完整训练闭环。

## 项目目标

- 使用 PyTorch 训练 MNIST 数据集
- 理解图像分类任务的标准训练流程
- 学习 `LeNet` 这类经典 CNN 结构
- 记录训练过程中的 loss 和 accuracy
- 将训练好的模型保存到本地

## 目录结构

```text
mnist_project/
├── configs/
│   └── config.py          # 训练超参数和保存路径
├── datasets/
│   └── mnist_dataset.py   # MNIST 数据加载与预处理
├── engine/
│   ├── trainer.py         # 训练逻辑
│   └── evaluator.py       # 测试集评估逻辑
├── models/
│   └── cnn.py             # CNN / LeNet 模型定义
├── utils/
│   ├── logger.py          # TensorBoard 日志
│   ├── metrics.py         # 指标工具
│   └── seed.py            # 随机种子固定
├── train.py               # 训练入口
├── test.py                # 预留测试脚本
└── requirements.txt       # Python 依赖
```

## 环境依赖

建议使用 Python 3.8+，然后安装依赖：

```bash
pip install -r requirements.txt
```

依赖主要包括：

- `torch`
- `torchvision`
- `tensorboard`
- `numpy`
- `matplotlib`

## 数据集

项目使用 `torchvision.datasets.MNIST` 自动下载数据集，默认保存到 `./data`。

首次运行训练脚本时会自动下载训练集和测试集，无需手动准备数据。

## 训练流程

直接运行训练脚本：

```bash
python train.py
```

训练过程会：

1. 固定随机种子，保证结果更稳定
2. 加载 MNIST 训练集和测试集
3. 构建 `LeNet` 模型
4. 使用 `CrossEntropyLoss` 作为损失函数
5. 使用 `Adam` 优化器进行训练
6. 每个 epoch 后在测试集上计算 accuracy
7. 将 loss 和 accuracy 写入 TensorBoard
8. 保存模型参数到本地

## 配置说明

训练超参数集中在 `configs/config.py` 中：

- `seed`：随机种子
- `batch_size`：批大小
- `epochs`：训练轮数
- `lr`：学习率
- `num_workers`：数据加载线程数
- `device`：默认训练设备
- `log_dir`：TensorBoard 日志目录
- `model_save_path`：模型保存路径

当前默认保存路径为：

```text
checkpoints/mnist_cnn_LeNet.pth
```

## TensorBoard 日志

训练时会把指标写入 `runs/mnist`。启动 TensorBoard：

```bash
tensorboard --logdir runs
```

然后在浏览器中查看训练曲线。

## 模型说明

当前训练入口使用的是 `models/cnn.py` 中的 `LeNet`。

文件中还保留了更早版本的 `SimpleCNN` 以及加入 `BatchNorm`、`Dropout` 的扩展实现，适合做结构对比和实验迭代。

## 输出内容

训练结束后会得到：

- 控制台输出每个 epoch 的 loss 和 accuracy
- TensorBoard 记录的训练指标
- 本地保存的模型参数文件

## 备注

- `test.py` 目前为空，主要评估逻辑已经放在 `engine/evaluator.py`
- 如果本地没有 `checkpoints` 目录，保存模型时需要先创建该目录
- 如果使用 CPU 训练，会自动回退到 CPU

