# Simple CNN Implementation For Emotion Classification

This implements a simple shallow PyTorch CNN which can predict emotions when given a headshot. The model achieved an F1 score of $\approx 0.385$. A huge limitation of the model is how shallow it was, and perhaps, given greater depth, it could have modeled the data better.


Simple data augmentation through the PyTorch torchvision library was performed, which should have helped the model generalize. 