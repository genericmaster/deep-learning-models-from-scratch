import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(r'C:\THEDO DONT TOUCH\machine learning basics\deep learning')
from neural_network_from_scratch.full_neural_net import Neural_net
from convolutional_network_from_scratch import Conv_network
from torchvision import datasets, transforms
import torch

# --- Class labels ---
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# --- Load data ---
transform = transforms.ToTensor()
test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)
data_loader_test = torch.utils.data.DataLoader(dataset=test_data, batch_size=64, shuffle=True)

# --- Build and load conv network ---
conv = Conv_network()
conv.add_conv_layer(filter=8, kernel=3, depth=1, stride=1, padding=0, pool=True, pool_size=2, pool_stride=2)
conv.add_conv_layer(filter=16, kernel=3, depth=8, stride=1, padding=0, pool=True, pool_size=2, pool_stride=2)

for i, layer in enumerate(conv.add_conv_layer_list):
    layer.weight_matrix = np.load(f'conv_layer_fashion_{i}_weights.npy')

# --- Build and load Neural_net ---
neural = Neural_net('categorical_cross_entropy')
neural.add_layer(neurons=128, features=400, activation='relu')
neural.result_layer(neurons=10, features=128, activation='softmax')

for i, layer in enumerate(neural.add_layer_list):
    layer.weight_matrix = np.load(f'scratch_dense_layer_{i}_weights.npy')
    layer.bias_matrix = np.load(f'scratch_dense_layer_{i}_bias.npy')

neural.output_layer.weight_matrix = np.load('scratch_output_layer_weights.npy')
neural.output_layer.bias_matrix = np.load('scratch_output_layer_bias.npy')

# --- Grab one batch ---
images, labels = next(iter(data_loader_test))
images_np = images.numpy()

# --- Run forward pass ---
features = conv.train(images_np)
output = neural.forward_prop(features)
predictions = np.argmax(output, axis=1)

# --- Plot 16 predictions ---
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, ax in enumerate(axes.flatten()):
    img = images[i].squeeze().numpy()
    pred = predictions[i]
    true = labels[i].item()
    ax.imshow(img, cmap='gray')
    color = 'green' if pred == true else 'red'
    ax.set_title(f'Pred: {class_names[pred]}\nTrue: {class_names[true]}', color=color, fontsize=8)
    ax.axis('off')
plt.suptitle('Scratch Network Predictions - Green = Correct, Red = Wrong', fontsize=12)
plt.tight_layout()
plt.show()

# --- Feature maps for one image ---
image = images_np[0]  # shape (1, 28, 28)

# Layer 1 feature maps
layer0 = conv.add_conv_layer_list[0]
layer0.conv_layer_computation(image)
layer0.activation()
feature_map1 = layer0.activation_map

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(feature_map1[i], cmap='gray')
    ax.set_title(f'Filter {i}')
    ax.axis('off')
plt.suptitle('Conv Layer 1 Feature Maps')
plt.tight_layout()
plt.show()

# Layer 2 feature maps
layer1 = conv.add_conv_layer_list[1]
layer1.conv_layer_computation(feature_map1)
layer1.activation()
feature_map2 = layer1.activation_map

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(feature_map2[i], cmap='gray')
    ax.set_title(f'Filter {i}')
    ax.axis('off')
plt.suptitle('Conv Layer 2 Feature Maps')
plt.tight_layout()
plt.show()