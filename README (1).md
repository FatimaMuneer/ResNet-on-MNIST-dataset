# MNIST Handwritten Digit Classifier with a Mini ResNet (PyTorch)

A beginner-friendly project that builds a small **ResNet** (Residual Network) from scratch in PyTorch and trains it on the **MNIST** handwritten digits dataset. The goal is to understand every part of the flow: data loading, the residual block, the full network, training, testing, and saving the model.

> Expected result: roughly **99% test accuracy** after 5 epochs (your exact number will vary slightly from run to run).

---

## Table of Contents

1. [What is MNIST?](#what-is-mnist)
2. [What is a ResNet?](#what-is-a-resnet)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [How to Run](#how-to-run)
6. [Cell-by-Cell Code Explanation](#cell-by-cell-code-explanation)
7. [Model Architecture Summary](#model-architecture-summary)
8. [Key Terms Glossary](#key-terms-glossary)
9. [Troubleshooting](#troubleshooting)
10. [Experiments to Try](#experiments-to-try)
11. [License](#license)

---

## What is MNIST?

MNIST is a classic dataset of **70,000 grayscale images** of handwritten digits (0 to 9). Each image is **28 x 28 pixels**.

- **60,000** images are used for training
- **10,000** images are used for testing

The task: given an image, predict which digit (0-9) it shows. This is called **image classification**.

## What is a ResNet?

In a normal neural network layer, the output is:

```
output = F(x)
```

In a **residual block**, the layer's input is added back to its output:

```
output = F(x) + x
```

That `+ x` is called a **skip connection** (or shortcut). It helps because very deep networks are hard to train: the learning signal gets weaker as it travels backward through many layers. The skip connection gives that signal a shortcut, so each block only has to learn a small correction instead of everything from scratch.

---

## Project Structure

```
mnist-resnet/
├── mnist_resnet.ipynb    # the notebook with all the code cells
├── README.md             # this file
├── requirements.txt      # Python packages needed
├── .gitignore            # files Git should ignore
└── mnist_resnet.pth      # saved model weights (created after training)
```

The `data/` folder (MNIST download) is created automatically on the first run and is ignored by Git.

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/mnist-resnet.git
cd mnist-resnet
```

**2. (Recommended) Create a virtual environment**

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

**3. Install the requirements**

```bash
pip install -r requirements.txt
```

## How to Run

**Option A: Jupyter Notebook (locally)**

```bash
jupyter notebook mnist_resnet.ipynb
```

Then choose **Kernel -> Restart & Run All**.

**Option B: Google Colab**

Upload `mnist_resnet.ipynb` to [Google Colab](https://colab.research.google.com), optionally switch to a GPU (**Runtime -> Change runtime type -> GPU**), and choose **Runtime -> Run all**.

> **Important:** run the cells **in order, from top to bottom**. Each cell uses variables created by earlier cells.

---

## Cell-by-Cell Code Explanation

### Cell 1: Imports and Device Setup

```python
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
```

**What it does:** loads the tools we need and decides where the computation runs.

- `import torch` loads PyTorch, the deep learning library.
- `torch.nn as nn` gives us building blocks for neural networks (layers like `Conv2d`, `Linear`, etc.).
- `torch.nn.functional as F` gives us simple functions like `F.relu`.
- `DataLoader` feeds data to the model in small groups called batches.
- `datasets` and `transforms` come from `torchvision`. `datasets` downloads MNIST for us, and `transforms` prepares images.
- The `device` line checks whether a **GPU** is available. If yes, it uses `"cuda"` (fast). If not, it uses `"cpu"` (slower but works everywhere). A GPU can do thousands of small calculations in parallel, which is perfect for neural networks.

---

### Cell 2: Loading and Preparing the Data

```python
tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds = datasets.MNIST("data", train=True,  download=True, transform=tfm)
test_ds  = datasets.MNIST("data", train=False, download=True, transform=tfm)

train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size=256)
```

**What it does:** downloads MNIST and prepares it for the model.

- `transforms.Compose([...])` chains several preparation steps into one pipeline.
- `ToTensor()` converts each image into a PyTorch tensor (a grid of numbers) and rescales pixel values from `0-255` to `0.0-1.0`.
- `Normalize((0.1307,), (0.3081,))` shifts and scales the numbers using MNIST's known **mean (0.1307)** and **standard deviation (0.3081)**. Training is steadier when inputs are centered around 0.
- `datasets.MNIST("data", train=True, download=True, transform=tfm)`: the first argument is the folder to save in, `train=True` selects the 60,000 training images, `download=True` downloads them if missing, and `transform=tfm` applies our pipeline to every image.
- The second line does the same with `train=False`, which gives the 10,000 test images the model never trains on.
- `DataLoader(..., batch_size=128, shuffle=True)` hands the model **128 images at a time** instead of all 60,000 at once (which would run out of memory). `shuffle=True` mixes the order every epoch so the model doesn't memorize the sequence.
- The test loader uses `batch_size=256` and no shuffle, because for testing the order doesn't matter and larger batches are faster.

---

### Cell 3: The Residual Block (the heart of ResNet)

```python
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)
```

**What it does:** defines one reusable residual block.

**The class setup**

- `class ResBlock(nn.Module)` means we are creating our own neural network piece. Every PyTorch model inherits from `nn.Module`.
- `__init__` is where we **create the layers**. `super().__init__()` sets up the parent class properly.
- `in_ch` is the number of input channels, `out_ch` is the number of output channels, and `stride` controls how much the image shrinks (stride 2 halves its width and height).

**The layers**

- `nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)` is a **convolution layer**. It slides small 3x3 filters over the image to detect patterns such as edges and curves. The arguments are: input channels, output channels, kernel size 3, stride, padding 1 (keeps the size the same when stride is 1). `bias=False` is used because BatchNorm right after it makes the bias unnecessary.
- `nn.BatchNorm2d(out_ch)` is **batch normalization**. It rescales the numbers flowing through the network so they stay in a healthy range, which makes training faster and more stable.
- The block has **two** conv + batchnorm pairs (`conv1/bn1` and `conv2/bn2`).

**The shortcut (skip path)**

- `self.shortcut = nn.Sequential()` starts as an **empty** sequence, which simply passes `x` through unchanged (the identity).
- The `if` checks whether the shapes of the input and output differ (`stride != 1` changes the image size, `in_ch != out_ch` changes the channel count). If they do, we can't add `x` to `out` directly because the shapes won't match. So we use a small `1x1` convolution to reshape `x` to match.

**The forward pass** (what happens when data flows through)

1. `self.conv1(x)`: first convolution.
2. `self.bn1(...)`: normalize.
3. `F.relu(...)`: **ReLU** replaces negative numbers with 0. This lets the network learn non-linear patterns.
4. `self.conv2(out)` then `self.bn2(...)`: second convolution and normalize (no ReLU yet).
5. `out + self.shortcut(x)`: **the skip connection**. We add the original input back. This one line is the whole idea of ResNet.
6. `F.relu(out)`: final ReLU on the sum.

---

### Cell 4: The Full Network (MiniResNet)

```python
class MiniResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU())
        self.layer1 = ResBlock(16, 16)
        self.layer2 = ResBlock(16, 32, stride=2)
        self.layer3 = ResBlock(32, 64, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(64, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

model = MiniResNet().to(device)
```

**What it does:** stacks residual blocks into a complete network and creates the model.

**The pieces**

- `stem`: the first layer. A convolution takes the **1-channel** grayscale image and produces **16 feature maps**, followed by BatchNorm and ReLU.
- `layer1`: a `ResBlock` from 16 to 16 channels. Size stays 28x28.
- `layer2`: a `ResBlock` from 16 to 32 channels with `stride=2`, so the image shrinks to 14x14.
- `layer3`: a `ResBlock` from 32 to 64 channels with `stride=2`, so the image shrinks to 7x7.
- `pool = nn.AdaptiveAvgPool2d(1)`: **global average pooling**. It averages each 7x7 feature map down to a single number, so we get 64 numbers.
- `fc = nn.Linear(64, 10)`: a **fully connected layer** that turns those 64 numbers into **10 scores**, one for each digit 0-9.

**The forward pass**

The data flows through `stem`, then the three layers, then pooling. `.flatten(1)` turns the `64 x 1 x 1` output into a flat list of 64 numbers (keeping the batch dimension). Finally `fc` gives the 10 scores.

**Shape journey of one image**

```
Input:    1  x 28 x 28
stem:     16 x 28 x 28
layer1:   16 x 28 x 28
layer2:   32 x 14 x 14
layer3:   64 x  7 x  7
pool:     64 x  1 x  1  ->  64 numbers
fc:       10 scores
```

**The last line:** `model = MiniResNet().to(device)` builds the model and moves it to the GPU (or CPU) chosen in Cell 1. The model and the data must be on the same device.

---

### Cell 5: Loss Function and Optimizer

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

**What it does:** sets up how the model measures its mistakes and how it improves.

- `criterion = nn.CrossEntropyLoss()` is the **loss function**. It compares the 10 scores with the correct digit and produces one number: the **loss**. A high loss means the model is wrong or unsure, and a low loss means it is right and confident. (It applies softmax internally, so we don't need to add it to the model.)
- `torch.optim.Adam(...)` is the **optimizer**. It updates the model's weights to reduce the loss. Adam is a popular default because it adapts the step size automatically.
- `model.parameters()` tells the optimizer which numbers it is allowed to change (all the weights in our network).
- `lr=1e-3` is the **learning rate** (0.001). It is the size of each update step. Too big and training becomes unstable, too small and it learns very slowly.

> This is the cell that causes `NameError: name 'criterion' is not defined` if you skip it.

---

### Cell 6: Training Loop

```python
for epoch in range(5):
    model.train()
    for images, labels in train_dl:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"epoch {epoch+1} done, last loss {loss.item():.4f}")
```

**What it does:** teaches the model by showing it the training images again and again.

- `for epoch in range(5)`: one **epoch** is one full pass over all 60,000 training images. We do 5 epochs.
- `model.train()`: switches the model to training mode (BatchNorm uses the current batch's statistics).
- `for images, labels in train_dl`: takes one batch (128 images with their correct labels) at a time.
- `.to(device)`: moves the batch to the same device as the model.

**The five steps inside the loop** (the core of all deep learning):

1. `outputs = model(images)`: **guess.** The model predicts scores for each image.
2. `loss = criterion(outputs, labels)`: **measure the error** by comparing the guesses with the true labels.
3. `optimizer.zero_grad()`: **clear old gradients.** PyTorch accumulates gradients, so we must reset them each time.
4. `loss.backward()`: **work out how to improve.** This is **backpropagation**: it calculates how much each weight contributed to the error.
5. `optimizer.step()`: **update the weights** a little in the direction that reduces the error.

- `print(...)`: shows the epoch number and the loss of the last batch. `loss.item()` converts the loss tensor into a plain Python number, and `:.4f` shows 4 decimals. The loss should generally go down across epochs.

---

### Cell 7: Testing the Model

```python
model.eval()
correct = 0
with torch.no_grad():
    for images, labels in test_dl:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()

print(f"Test accuracy: {100 * correct / len(test_ds):.2f}%")
```

**What it does:** checks how well the model works on images it has **never trained on**.

- `model.eval()`: switches to evaluation mode (BatchNorm uses its stored statistics instead of the batch's).
- `correct = 0`: a counter for right answers.
- `with torch.no_grad()`: turns off gradient tracking. We're not learning now, so this saves memory and time.
- `model(images).argmax(dim=1)`: for each image, picks the **digit with the highest score**. That is the model's prediction.
- `(preds == labels)` gives True or False for each image. `.sum()` counts the Trues and `.item()` converts to a normal number, which we add to `correct`.
- The final line divides correct answers by the total (10,000), multiplies by 100, and prints the accuracy as a percentage.

---

### Cell 8: Predict One Image and Save the Model

```python
img, label = test_ds[0]
pred = model(img.unsqueeze(0).to(device)).argmax(1).item()
print("true:", label, "predicted:", pred)

torch.save(model.state_dict(), "mnist_resnet.pth")
```

**What it does:** tests the model on a single image and saves what it learned.

- `test_ds[0]` takes the first test image and its true label.
- `img.unsqueeze(0)` adds a batch dimension. The model expects a batch, so a single image of shape `1x28x28` becomes `1x1x28x28`.
- `.to(device)` moves it to the right device, `model(...)` gets the 10 scores, and `.argmax(1).item()` gives the predicted digit as a plain number.
- `print(...)` shows the true label and the prediction so you can compare them.
- `torch.save(model.state_dict(), "mnist_resnet.pth")` saves all the learned weights to a file. `state_dict()` is a dictionary of the model's weights.

**To load the model later:**

```python
model = MiniResNet().to(device)
model.load_state_dict(torch.load("mnist_resnet.pth", map_location=device))
model.eval()
```

(You need the `ResBlock` and `MiniResNet` class definitions available when loading.)

---

## Model Architecture Summary

| Stage | Output shape | Purpose |
|---|---|---|
| Input | 1 x 28 x 28 | grayscale digit image |
| Stem (Conv + BN + ReLU) | 16 x 28 x 28 | first pattern detection |
| ResBlock 1 | 16 x 28 x 28 | learn features with skip connection |
| ResBlock 2 (stride 2) | 32 x 14 x 14 | smaller image, richer features |
| ResBlock 3 (stride 2) | 64 x 7 x 7 | even richer features |
| Global Average Pool | 64 | summarize each feature map |
| Linear | 10 | one score per digit |

## Key Terms Glossary

| Term | Easy meaning |
|---|---|
| **Epoch** | one full pass through the training data |
| **Batch** | a small group of images processed together |
| **Convolution** | a small filter that scans the image for patterns |
| **BatchNorm** | keeps numbers in a stable range during training |
| **ReLU** | turns negative numbers into 0 |
| **Skip connection** | adds a layer's input to its output (`F(x) + x`) |
| **Loss** | a number measuring how wrong the model is |
| **Optimizer** | the algorithm that updates weights to reduce loss |
| **Learning rate** | the size of each update step |
| **Backpropagation** | calculating how each weight affected the error |
| **Gradient** | the direction and size of change needed for a weight |
| **Logits / scores** | the raw 10 numbers the model outputs before choosing a digit |
| **Overfitting** | memorizing training data instead of learning general patterns |

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `NameError: name 'criterion' is not defined` | The Step 5 cell was not run. Run cells in order, or use **Restart & Run All**. |
| `NameError: name 'model' / 'train_dl' is not defined` | The cell that defines it was skipped. Run it first. |
| `RuntimeError: Expected all tensors to be on the same device` | The model and data are on different devices. Make sure both use `.to(device)`. |
| Very low accuracy (below 95%) | Check that you ran every cell in order and that `model.train()` / `model.eval()` are present. |
| Download is slow or fails | Check your internet connection and re-run the data cell. |
| Training is slow | Use a GPU (Colab: **Runtime -> Change runtime type -> GPU**). |

## Experiments to Try

- Increase `range(5)` to `range(10)` and compare accuracy.
- Remove the skip connection (change `out + self.shortcut(x)` to `out`) and compare.
- Stack more `ResBlock(64, 64)` layers.
- Try learning rates `1e-2` and `1e-4`.
- Plot the loss and accuracy for every epoch.
- Show the images the model got wrong.
- Move on to CIFAR-10 or a pretrained ResNet-18 from `torchvision.models`.

## License

This project is released under the MIT License. Feel free to use it for learning.
