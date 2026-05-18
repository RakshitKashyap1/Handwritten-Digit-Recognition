# Handwritten Digit Recognition with MNIST

A TensorFlow/Keras CNN achieving ~99% accuracy on MNIST.

## Google Colab Setup

1. Go to [Google Colab](https://colab.research.google.com)
2. Create new notebook
3. Run: `!pip install tensorflow numpy matplotlib seaborn scikit-learn`
4. Copy the code from `mnist_digit_recognition.py`

## Architecture
```
Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(64) → Flatten → Dense(64) → Dropout(0.5) → Dense(10)
```

## Skills Demonstrated
- **Data Preprocessing**: Normalization, reshaping for CNN
- **CNN Design**: Conv2D, MaxPooling, Dropout regularization
- **Training**: Adam optimizer, validation split
- **Evaluation**: Accuracy, confusion matrix, classification report