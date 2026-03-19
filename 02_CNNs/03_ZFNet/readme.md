# ZFNet: Visualizing and Understanding Convolutional Networks

ZFNet (Zeiler & Fergus Net) is a classic Convolutional Neural Network architecture that won the **ILSVRC 2013** competition. It is essentially a refined version of AlexNet, with architecture improvements derived directly from a novel visualization technique.

## 🚀 Key Improvements over AlexNet

While ZFNet follows the same 8-layer structure as AlexNet (5 Convolutional + 3 Fully Connected), it introduced critical hyperparameter changes:

1.  **Reduced Filter Size in Conv1**: AlexNet used 11x11 filters. ZFNet reduced this to **7x7**.
2.  **Reduced Stride in Conv1**: AlexNet used a stride of 4. ZFNet used a **stride of 2**.
3.  **Increased Stride in Conv2**: ZFNet uses a stride of 2 in the second layer to balance the higher resolution from Conv1.
4.  **The Result**: These changes significantly reduced "pixel-aliasing" artifacts. Smaller strides and filters in the early layers allow the network to retain much more fine-grained structural information from the input image.

---

## 🔍 What is DeconvNet (ZFDeconvNet)?

The most influential contribution of the ZFNet paper was the **Deconvolutional Network (DeconvNet)**. In this project, it is implemented as `ZFDeconvNet`.

### Purpose
DeconvNet is a visualization tool used to "reverse" the CNN process. It maps activations from a hidden layer back into the input pixel space, allowing researchers to see exactly which patterns (e.g., dog ears, textures, wheels) a specific neuron is looking for.

### How it Works
To reconstruct an image from a feature map, DeconvNet performs three main operations in reverse:
1.  **Unpooling**: Uses stored "switches" (max-pooling indices) from the forward pass to place activations back in their original relative positions.
2.  **ReLU (Rectified Linear Unit)**: Filters the reconstruction to ensure only positive feature signals are projected back.
3.  **Deconvolution (Transpose Convolution)**: Uses the **transposed weights** of the original filters to project the feature maps back toward the input dimensions.

---

## 🧩 Learning Takeaway
ZFNet is a perfect example of how **interpretability** leads to better engineering. By building a tool (DeconvNet) to "see" what the model was doing wrong, the authors were able to optimize the architecture and set a new state-of-the-art record in computer vision.
