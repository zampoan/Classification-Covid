print("This is debug.py")
#from torchmetrics.classification import MulticlassAccuracy
import torchmetrics
a = torchmetrics.Accuracy(task="multiclass", num_classes=3)