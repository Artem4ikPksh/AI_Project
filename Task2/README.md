# Named Entity Recognition and Image Classification Pipeline

## Project Overview
This project integrates a Named Entity Recognition (NER) model with an Image Classification model to verify if the detected animal in a text matches the animal in an image. It consists of the following components:
- **NER Model**: Extracts animal names from a given text.
- **Image Classification Model**: Classifies images into predefined animal categories.
- **Verification Pipeline**: Compares the extracted entity from the text with the predicted image classification.

## Project Structure
```
project_directory/
│── animal_data/             # Directory containing image dataset
│── models/                  # Directory to save trained models
│── scripts/
│   ├── train_ner.py         # Script to train the NER model
│   ├── train_classifier.py  # Script to train the image classification model
│   ├── inference.py         # Script for running inference and verification
│── requirements.txt         # Required dependencies
│── README.md                # Documentation
```

## Setup Instructions
### 1. Clone the Repository
```bash
git clone <repository_url>
cd project_directory
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed. Install dependencies using:
```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset
Store images in the `animal_data/` directory with the following structure:
```
animal_data/
│── dog/
│   ├── image1.jpg
│   ├── image2.jpg
│── cat/
│   ├── image1.jpg
│   ├── image2.jpg
...
```

### 4. Train the Models
#### Train the NER Model:
```bash
python scripts/train_ner.py
```
#### Train the Image Classifier:
```bash
python scripts/train_classifier.py
```

### 5. Running Inference
To verify if an animal mentioned in the text matches the one in the image:
```bash
python scripts/inference.py --text "There is a tiger in the picture." --image "test_image.jpg"
```

## Model Details
### Named Entity Recognition (NER)
- Uses **BERT (dbmdz/bert-large-cased-finetuned-conll03-english)** for extracting animal names.
- Tokenizes input text and assigns entity labels.
- Outputs detected animal names.

### Image Classification
- Uses **ResNet18** as the backbone.
- Trained on labeled images in `animal_data/`.
- Normalizes images and resizes them to **224x224**.
- Outputs the most probable class.

### Verification Process
1. The text is processed to extract the animal name(s).
2. The image is classified into an animal category.
3. A comparison is made between the detected animal and the predicted image class.
4. Returns `True` if the extracted animal matches the classified image, else `False`.

## Notes
- Ensure the dataset is well-organized with clear class labels.
- You can modify the model hyperparameters in respective scripts.
- Extend the project by adding more animal classes or improving the NER pipeline.

## License
This project is open-source under the MIT License.
