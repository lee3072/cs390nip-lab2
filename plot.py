import numpy as np
import matplotlib.pyplot as plt
# reference: https://matplotlib.org/tutorials/introductory/pyplot.html
dataset = ["mnist_d","mnist_f","cifar_10","cifar_100f","cifar_100c"]
ann_accuracy = [97.68,87.05,10.00,1.00,5.00]
cnn_accuracy = [98.70,88.30,70.21,25.15,38.94]
plt.figure(figsize=(25, 3))
plt.subplot(131)
plt.bar(dataset,ann_accuracy)   # reference: https://stackoverflow.com/questions/6282058/writing-numerical-values-on-the-plot-with-matplotlib
for a,b in zip(dataset, ann_accuracy): 
    plt.text(a, b, str(b))
plt.title("ann")
plt.subplot(132)
plt.bar(dataset,cnn_accuracy)
for a,b in zip(dataset, cnn_accuracy): 
    plt.text(a, b, str(b))
plt.title("cnn")
plt.show()