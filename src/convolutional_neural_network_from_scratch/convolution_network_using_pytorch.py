import torchvision as py
import torch
import PIL.Image
import matplotlib.pyplot as plt

image=PIL.Image.open(r"C:\Users\rakhi\Downloads\WhatsApp Image 2026-06-05 at 01.00.45.jpeg").convert('L')

transform = py.transforms.ToTensor()
tensor = transform(image)
tensor = torch.unsqueeze(tensor,0)

conv= torch.nn.Conv2d(1,1,kernel_size=3)

# PART 1 — create the kernel values
kernel = torch.tensor([
    [1.,  0., -1.],
    [1.,  0., -1.],
    [1.,  0., -1.]
])

# PART 2 — assign it to the conv layer
with torch.no_grad():
    conv.weight = torch.nn.Parameter(kernel.reshape(1, 1, 3, 3))
    conv.bias.fill_(0.)

with torch.no_grad():
    output = conv(tensor)
    

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)  # 1 row, 2 columns, first plot
plt.imshow(image, cmap='gray')

plt.subplot(1,2,2)  # second plot
plt.imshow(output.squeeze().numpy(), cmap='gray')# show feature map here

plt.show()

import numpy as np
np.save('image_array1.npy', tensor.squeeze().numpy())
np.save('feature_map1.npy', output.squeeze().detach().numpy())
