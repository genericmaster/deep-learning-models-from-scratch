import numpy as np
from recurrent_network import Recurrent_net  # adjust import

np.random.seed(42)

# Single-sequence training data: list of sequences and labels
num_sequences = 20
timesteps = 5
input_dim = 3
hidden_dim = 8
num_classes = 2

# Each X[i] is shape (timesteps, input_dim) — 2D, no batch axis
X = [np.random.randn(timesteps, input_dim) for _ in range(num_sequences)]
labels = [np.eye(num_classes)[np.random.randint(0, num_classes)] for _ in range(num_sequences)]

# Build the model
model = Recurrent_net(input_dim=input_dim, hidden_dim=hidden_dim, loss='categorical_cross_entropy')
model.net.add_layer(neurons=8, features=hidden_dim, activation='relu')
model.net.add_layer(neurons=num_classes, features=8, activation='softmax')

# Train one sequence at a time
epochs = 10
learning_rate = 0.01
optimizer = 'adam'

for epoch in range(epochs):
    epoch_loss = 0
    for i in range(num_sequences):
        single_X = X[i]                          # shape (timesteps, input_dim)
        single_label = labels[i].reshape(1, -1)  # shape (1, num_classes) — head needs a batch axis
        
        final_hidden = model.rnn.forward(single_X)
        prediction = model.net.forward(final_hidden.reshape(1, -1))  # add batch axis for head
        model.net.backward(single_label, prediction,loss_type='average')
        input_grad = model.net.incoming_gradient_list[-1] if model.net.incoming_gradient_list else None
        # Actually use the return value from backward instead:
        # input_grad = model.net.backward(single_label, prediction)
        
        model.rnn.BPTT(input_grad.flatten())  # RNN expects 1D incoming gradient
        
        model.net.optimizers(learning_rate, optimizer)
        model.rnn.rnn_optimizer(learning_rate, optimizer)
    
    print(f"Epoch {epoch}: latest loss = {model.net.loss_track_average}")