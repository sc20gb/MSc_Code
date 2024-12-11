import torchvision.transforms as transforms
from Data_Loading.data_loading import load_test_data
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

def load_llava_med_7b_model():
    # Load model directly
    tokenizer = AutoTokenizer.from_pretrained("PharMolix/BioMedGPT-LM-7B")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("PharMolix/BioMedGPT-LM-7B")
    return model, tokenizer


# Adjust the evaluate function for LLaVA-Med-7B
def evaluate_llava_med(model, tokenizer, test_loader, max_question_length=128, max_answer_length=128, device='cpu'):
    model.eval()  # Set model to evaluation mode
    model.to(device)
    
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            images, masks, questions, answers, categories = batch
            
            images = images.to(device)
            masks = masks.to(device)
            
            # Tokenize questions and answers
            encoded_questions = tokenizer(questions, padding="max_length", truncation=True, max_length=max_question_length, return_tensors="pt").to(device)
            encoded_answers = tokenizer(answers, padding="max_length", truncation=True, max_length=max_answer_length, return_tensors="pt").to(device)

            # Run forward pass
            outputs = model(input_ids=encoded_questions["input_ids"], attention_mask=encoded_questions["attention_mask"])

            # Compute loss
            loss = torch.nn.functional.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), encoded_answers["input_ids"].view(-1))
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct_predictions += (predictions == encoded_answers["input_ids"]).sum().item()
            total_samples += encoded_answers["input_ids"].numel()
    
    # Calculate average loss and accuracy
    average_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_samples
    
    print(f"Evaluation - Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return average_loss, accuracy

# Example Usage
if __name__ == "__main__":
    # Load the model and tokenizer
    model, tokenizer = load_llava_med_7b_model()

    # Initialize the test loader with your transform and directory
    test_loader = load_test_data(transform=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ]), batchSize=1, seed=42, dataDir=os.path.join(os.getcwd(), 'Slake1.0'))

    # Run the evaluation
    evaluate_llava_med(model, tokenizer, test_loader)
