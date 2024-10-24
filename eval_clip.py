import torch
import os
import csv
from transformers import AutoTokenizer
from Model_Defs.CLIP import CLIP
import torch.nn.functional as F
import torchvision.transforms as transforms
from Data_Loading.data_loading import load_combined_text_data

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

if __name__ == "__main__":
    # Configure parameters
    BATCHSIZE = 32
    MAX_LENGTH = 256
    IMAGESIZE = 224
   # model_path = os.path.join("/nobackup", "sc20gwb", "Models", "vicuna-7b-v1.5")
    model_path = os.path.join(os.getcwd(), "Models", "vicuna-7b-v1.5")

    data_path = os.path.join(os.getcwd(), 'Slake1.0')
    path1 = os.path.join(os.getcwd(),"Models","clip_model_23.pth")#os.path.join("/nobackup","sc20gwb","Models", "Models_to_upload" , "V_" + str(10320005),"clip_model_" + str(23) + ".pth")


    # Evaluate model
    accuracy = evaluate_clip_model(model_path=model_path, data_path=data_path,clip_path=path1, BATCHSIZE=BATCHSIZE, MAX_LENGTH=MAX_LENGTH, IMAGESIZE=IMAGESIZE)

    # Save results
    save_eval_results(version=10320000, accuracy=accuracy)
