# MNIST Classifier

## Overview
This project implements multiple classifiers for the MNIST dataset, including:
- **Random Forest Classifier**
- **Feed-Forward Neural Network (FNN)**
- **Convolutional Neural Network (CNN)**

Each classifier follows a common interface to allow easy switching between different models.

## Project Structure
```
├── mnist_classifier.py  # Main script containing classifier implementations
├── requirements.txt     # Dependencies for the project
└── README.md            # Project documentation
```

## Setup
### Prerequisites
Ensure you have Python 3.8+ installed. You also need the following dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies
The required packages are:
- `torch`
- `torchvision`
- `numpy`
- `scikit-learn`

## Usage
### Running the Classifier
The script `mnist_classifier.py` allows you to train and evaluate classifiers. It automatically loads the MNIST dataset and trains the selected model.

Example usage:

```python
from mnist_classifier import MnistClassifier, load_mnist
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = load_mnist()

# Choose classifier: 'rf' (Random Forest), 'nn' (Feed-Forward NN), 'cnn' (Convolutional NN)
classifier = MnistClassifier('cnn')
classifier.train(X_train, y_train)
predictions = classifier.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
```

### Model Selection
You can switch between different models by passing one of the following options:
- `'rf'` for Random Forest
- `'nn'` for Feed-Forward Neural Network
- `'cnn'` for Convolutional Neural Network

## Explanation of Models
1. **RandomForestMnist**: A basic ensemble learning model using a Random Forest classifier.
2. **FNNMnist**: A simple fully connected neural network with three layers.
3. **CNNMnist**: A Convolutional Neural Network with two convolutional layers, max pooling, and fully connected layers.

## GPU Support
The neural network models (`FNNMnist` and `CNNMnist`) automatically use GPU if available:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Results
The accuracy of each model may vary based on hyperparameters and training conditions. Typically:
- Random Forest: ~95%
- Feed-Forward NN: ~97%
- CNN: ~99%

## License
This project is licensed under the MIT License.

