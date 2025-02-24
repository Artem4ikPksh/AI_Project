import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
import torchvision.models as models
import torch.nn as nn

# Loading the NER model and tokenizer
ner_model_path = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(ner_model_path)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_path)
ner_model.eval()

# Using a pipeline for NER
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer)

def extract_animal_from_text(text):
    entities = ner_pipeline(text)
    extracted_animals = []
    current_animal = ""

    for entity in entities:
        word = entity["word"].replace("##", "") # Remove tokenization artifacts
        if entity["entity"].startswith("B-"):
            if current_animal:
                extracted_animals.append(current_animal)
            current_animal = word
        elif entity["entity"].startswith("I-") and current_animal:
            current_animal += word
    if current_animal:
        extracted_animals.append(current_animal)

    return " ".join(extracted_animals) if extracted_animals else None # Return None if nothing is found


# Loading image classification model
image_model_path = "image_classifier"  
num_classes = 15  

# Initialize the ResNet model
image_model = models.resnet18(weights=None)
image_model.fc = nn.Linear(image_model.fc.in_features, num_classes)

# Loading scales
try:
    checkpoint = torch.load(image_model_path, map_location=torch.device('cpu'), weights_only=True)
    image_model.load_state_dict(checkpoint, strict=False)
    image_model.eval()
except Exception as e:
    print(f"Error loading model image: {e}")

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def classify_image(image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = image_model(image_tensor)
    predicted_class = torch.argmax(outputs, dim=1).item()
    return predicted_class

def verify_animal_in_image(text, image_path, class_names):
    extracted_animal = extract_animal_from_text(text)
    if not extracted_animal:  
        print("Couldn't find the animal in the image!")
        return False
    
    predicted_class = classify_image(image_path)
    predicted_animal = class_names[predicted_class]

    print(f"Extracted animal: {extracted_animal}")
    print(f"Predicted animal: {predicted_animal}")

    return extracted_animal.lower() in predicted_animal.lower() or predicted_animal.lower() in extracted_animal.lower()

if __name__ == "__main__":
    class_names = ["dog", "cat", "cow", "elephant", "tiger", "lion", "horse", "zebra", "giraffe", "bear"]
    text_input = "There is a cow in the picture."
    image_path = "test_image.jpg"
    result = verify_animal_in_image(text_input, image_path, class_names)
    print(f"Verification result: {result}")