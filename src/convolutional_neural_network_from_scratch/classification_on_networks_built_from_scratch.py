import numpy as np
import torch
import sys
sys.path.append(r'C:\THEDO DONT TOUCH\machine learning basics\deep learning')
from neural_network_from_scratch.full_neural_net import Neural_net
from convolutional_network_from_scratch import Conv_network

from torchvision import datasets, transforms

transform = transforms.ToTensor()
train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)

data_loader_train = torch.utils.data.DataLoader(dataset=train_data,batch_size=64,shuffle=True)
data_loader_test = torch.utils.data.DataLoader(dataset=test_data,batch_size=64)
conv =Conv_network()
conv.add_conv_layer(filter=8,kernel=3,depth=1,stride=1,padding=0,pool=True,pool_size=2,pool_stride=2)
conv.add_conv_layer(filter=16,kernel=3,depth=8,stride=1,padding=0,pool=True,pool_size=2,pool_stride=2)
neural=Neural_net('categorical_cross_entropy')
layer1=neural.add_layer(neurons=128,features=400,activation='relu')
layer2=neural.result_layer(neurons=10,features=128,activation='softmax')

all_features = []
all_labels = []

for images, labels in data_loader_train:
    images = images.numpy()
    features = conv.train(images)
    all_features.append(features)
    all_labels.append(labels.numpy())

x = np.vstack(all_features)      # shape (60000, 400)
Y = np.concatenate(all_labels)
Y_onehot = np.eye(10)[Y]  # shape (60000, 10))# shape (60000,)

loss = neural.train(X=x,labels=Y_onehot,batch_size=64,learning_rate=0.001,epoch=1,optimizer='adam')

all_features_test = []
all_labels_test = []

for images, labels in data_loader_test:
    images = images.numpy()
    features = conv.train(images)
    all_features_test.append(features)
    all_labels_test.append(labels.numpy())

X_test = np.vstack(all_features_test)      
Y_test = np.concatenate(all_labels_test)
Y_onehot_test = np.eye(10)[Y_test]

batch_size = 64
test_accuracies_scratch = []

for i in range(0, X_test.shape[0], batch_size):
    X_batch = X_test[i:i+batch_size]
    Y_batch = Y_test[i:i+batch_size]
    output = neural.forward_prop(X_batch)
    prediction = np.argmax(output, axis=1)
    accuracy = np.mean(prediction == Y_batch)
    test_accuracies_scratch.append(accuracy)

np.save('scratch_test_accuracies.npy', np.array(test_accuracies_scratch))
np.save('scratch_train_losses.npy', np.array(loss))

# save dense layer weights
for i, layer in enumerate(neural.add_layer_list):
    np.save(f'scratch_dense_layer_{i}_weights.npy', layer.weight_matrix)
    np.save(f'scratch_dense_layer_{i}_bias.npy', layer.bias_matrix)

# save output layer weights
np.save('scratch_output_layer_weights.npy', neural.output_layer.weight_matrix)
np.save('scratch_output_layer_bias.npy', neural.output_layer.bias_matrix)