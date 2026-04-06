# MNIST CNN 入门项目

这是一个用于深度学习入门训练流程验证的 MNIST 手写数字分类项目。项目围绕数据加载、模型训练、评估、日志记录和模型保存展开，完整呈现了一个基础图像分类任务的闭环。

## 项目定位

- 使用 PyTorch 完成 MNIST 分类训练
- 以 `LeNet` 为主模型进行实验
- 记录训练 loss 和测试 accuracy
- 保存训练后的模型权重
- 作为后续更复杂图像分类项目的基础

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

## 数据集

项目使用 `torchvision.datasets.MNIST` 自动下载数据集，默认保存到 `./data`。

训练和测试数据在首次运行时自动完成下载，无需手动准备数据。

## 训练流程

训练入口为：

```bash
python train.py
```

训练过程包含以下步骤：

1. 固定随机种子
2. 加载 MNIST 训练集和测试集
3. 构建 `LeNet` 模型
4. 使用 `CrossEntropyLoss`
5. 使用 `Adam` 优化器
6. 每个 epoch 后在测试集上评估 accuracy
7. 将指标写入 TensorBoard
8. 保存模型参数到本地

## 配置说明

训练超参数统一写在 `configs/config.py` 中：

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

## 模型说明

训练入口当前使用 `models/cnn.py` 中的 `LeNet`。

模型文件中保留了更早版本的 `SimpleCNN`，以及加入 `BatchNorm` 和 `Dropout` 的扩展实现，作为结构演进记录。

## 输出内容

训练结束后会得到三类输出：

- 控制台输出每个 epoch 的 loss 和 accuracy
- TensorBoard 记录的训练曲线
- 本地保存的模型参数文件

## 备注

- `test.py` 目前为空文件，评估逻辑已放在 `engine/evaluator.py`
- 如果本地没有 `checkpoints` 目录，保存模型前需要先创建该目录
- 如果使用 CPU 训练，代码会自动回退到 CPU
