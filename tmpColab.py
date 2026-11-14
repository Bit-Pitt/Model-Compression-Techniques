
!git clone https://github.com/Bit-Pitt/Model-Compression-Techniques
%cd /content/Model-Compression-Techniques/notebooks/

import torchvision
torchvision.datasets.CIFAR10(root="../data", train=True, download=True)
torchvision.datasets.CIFAR10(root="../data", train=False, download=True)


!cd .. && zip -r /content/outputs.zip checkpoints results
from google.colab import files
files.download("/content/outputs.zip")