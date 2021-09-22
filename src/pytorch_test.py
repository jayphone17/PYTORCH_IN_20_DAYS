import torch
import cv2

print(cv2.__version__)

x = torch.empty(5, 3)
print(x)

print("torch version is : {}".format(torch.__version__))
print("cuda is available : {}".format(torch.cuda.is_available()))