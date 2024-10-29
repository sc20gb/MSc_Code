import torch
import os
import csv
from transformers import AutoTokenizer, CLIPProcessor
from Model_Defs.CLIP import CLIP
import torch.nn.functional as F
import torchvision.transforms as transforms
from Data_Loading.data_loading import load_combined_text_data, load_test_data
from Model_Defs.CLIP_with_LORA import CLIPWithLoRA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_clip_model(model_path, data_path, clip_path, BATCHSIZE=16, MAX_LENGTH=256, IMAGESIZE=224, device='cuda'):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"), do_sample=False)

    clip = CLIP(vocab_size=tokenizer.vocab_size, transformer_width=512, context_length=MAX_LENGTH, transformer_layers=6,
                transformer_heads=8, embed_dim=512, vision_width=512, image_resolution=IMAGESIZE, vision_patch_size=56,
                vision_layers=6, device=device)

    # Load trained model weights
    state_dict = torch.load(clip_path)
    clip.load_state_dict(state_dict,strict=True)
    clip.to(device)
    clip.eval()

    # Load the evaluation dataset (image-text pairs)
    eval_loader, _ = load_combined_text_data(transforms.Compose([
        transforms.Resize((IMAGESIZE, IMAGESIZE)),
        transforms.ToTensor()
    ]), BATCHSIZE, 42, data_path)

    correct_predictions = 0
    total_samples = 0

    # Evaluate the model on image-text pairs
    with torch.no_grad():
        for image_tensor, mask_tensor, text in eval_loader:
            # Tokenize text
            text_tensor = torch.cat([tokenizer(a + "</s>", return_tensors="pt", padding='max_length', max_length=MAX_LENGTH).input_ids for a in text], 0).to(device)
            image_tensor = image_tensor.to(device)

            # Get image and text features from CLIP model
            image_features, text_features = clip(image_tensor, text_tensor)

            # Normalize the features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # Cosine similarity
            logit_scale = clip.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()

            # Get the top predictions (argmax of the cosine similarity matrix)
            predicted_labels = torch.argmax(logits_per_image, dim=1)
            true_labels = torch.arange(len(logits_per_image)).to(device)

            # Count correct predictions
            correct_predictions += (predicted_labels == true_labels).sum().item()
            total_samples += len(image_tensor)

    # Calculate accuracy
    accuracy = correct_predictions / total_samples
    print(f"Evaluation Accuracy: {accuracy * 100:.2f}%")

    return accuracy

def save_eval_results(version, accuracy):
    # Save evaluation results to a CSV file
    results_dir = os.path.join(os.getcwd(), "Eval_Results", f"V_{version}")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    filename = os.path.join(results_dir, "eval_results.csv")
    
    with open(filename, mode='w', newline='\n') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['Version', 'Accuracy'])
        # Write data
        writer.writerow([version, accuracy])

    print(f"Evaluation results saved to '{filename}'.")

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import CLIPProcessor, CLIPModel

#evaluate the clip modelson the dataset provided
def universal_eval(model, dataset):
    results = {"accuracy": 0.0, "prec": 0.0, "recall": 0.0, "f1": 0.0}
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in dataset:
            # Check if this is the first dataloader format
            if len(batch) == 5:
                images, masks, questions, answers, categories = batch
                # Concatenate questions and answers for text input
                texts = [q + " " + a for q, a in zip(questions, answers)]
            
            # Otherwise, assume it's the second dataloader format
            elif len(batch) == 3:
                images, masks, texts = batch

            #tokenisation
            texts_pre = model.pre_process_texts(texts)
            
            image_embeddings, text_embeddings = model(images.to(model.device),texts_pre.to(model.device))
            
            # Normalize embeddings for cosine similarity
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            
            # Compute similarity scores between each image and text embedding
            logit_scale = model.logit_scale.exp()
            similarities = logit_scale * image_embeddings @ text_embeddings.T  # (batch_size, batch_size)
            
            # Predicted class is the index with the highest similarity score
            predicted_class_indices = similarities.argmax(dim=1).cpu().tolist()
            
            # True labels as indices, assuming each (image, text) pair should match positionally
            true_class_indices = torch.arange(len(texts)).cpu()
            
            # Append predictions and true labels for metrics
            all_preds.extend(predicted_class_indices)
            all_labels.extend(true_class_indices)

    # Calculate metrics
    results["accuracy"] = accuracy_score(all_labels, all_preds) * 100

    return results

# Handle the assigining of model to the GPUs
def handle_devices(cpu_only=False):
    if torch.cuda.is_available() and not cpu_only:
        gpu_count = torch.cuda.device_count()
        if gpu_count >= 2:
            print(f"CUDA is available with {gpu_count} GPU(s)!")
            
            # Assign the first GPU for the visual encoder
            device_vit = torch.device("cuda:0")
            print(f"Visual encoder will run on GPU 0: {torch.cuda.get_device_name(0)}")

            # Assign the second GPU for the connector LLM
            device_llm = torch.device("cuda:1")
            print(f"Connector LLM will run on GPU 1: {torch.cuda.get_device_name(1)}")
        else:
            print("Only one GPU available, models are split between GPU 0")
            device_vit = torch.device("cuda:0")
            device_llm = torch.device("cuda:0")
    else:
        print("CUDA is not available. Training will proceed on CPU.")
        device_vit = torch.device("cpu")
        device_llm = torch.device("cpu")

    return device_vit, device_llm

if __name__ == "__main__":
    # Configure parameters
    BATCHSIZE = 32
    MAX_LENGTH = 256
    IMAGESIZE = 224

    #Devices
    device, _ = handle_devices()

    #Initilise the model trained on the slake data
    vicuna_path = os.path.join(os.getcwd(), "Models", "vicuna-7b-v1.5")
    slake_trained_model = os.path.join(os.getcwd(),"Models","clip_model_23.pth")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(vicuna_path, "tokenizer"),do_sample=True)
    clip = CLIP(vocab_size=tokenizer.vocab_size, transformer_width=512, context_length=MAX_LENGTH, transformer_layers=6,
                transformer_heads=8, embed_dim=512, vision_width=512, image_resolution=IMAGESIZE, vision_patch_size=56,
                vision_layers=6, device=device)
    state_dict = torch.load(slake_trained_model)
    clip.load_state_dict(state_dict,strict=True)

    clip.set_tokenizer(tokenizer,256)

    #Initilise the pre-trained model
    pretrained_model = CLIPWithLoRA(device=device)

    #Load data
    data_path = os.path.join(os.getcwd(), 'Slake1.0')
    test_dataset_pre = load_test_data(transforms.Compose([transforms.Resize((IMAGESIZE, IMAGESIZE)),transforms.ToTensor()]), BATCHSIZE, 42, data_path)
    test_dataset_no_pre = load_test_data(transforms.Compose([pretrained_model.pre_process_images]),BATCHSIZE,42,data_path)

    # Load the evaluation dataset (image-text pairs)
    eval_loader_pre, _ = load_combined_text_data(transforms.Compose([transforms.Resize((IMAGESIZE, IMAGESIZE)),transforms.ToTensor()]), BATCHSIZE, 42, data_path)
    test_dataset_no_pre, _ = load_combined_text_data(transforms.Compose([pretrained_model.pre_process_images]), BATCHSIZE, 42, data_path)
  
    # Run the evals
    pretrained_model_results_test =  universal_eval(pretrained_model,test_dataset_no_pre)
    pretrained_model_results_eval = universal_eval(pretrained_model,test_dataset_no_pre)
    slake_trained_model_results = universal_eval(clip,eval_loader_pre)
    slake_trained_model_results_test = universal_eval(clip,test_dataset_pre)

    # Outpput results
    print(pretrained_model_results_test)
    print(pretrained_model_results_eval)
    print(slake_trained_model_results)
    print(slake_trained_model_results_test)