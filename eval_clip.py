import torch
import os
from transformers import AutoTokenizer
from Model_Defs.CLIP import CLIP
import torchvision.transforms as transforms
from Data_Loading.data_loading import load_combined_text_data, load_test_data
from Model_Defs.CLIP_with_LORA import CLIPWithLoRA
from sklearn.metrics import accuracy_score

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
            similarities = image_embeddings @ text_embeddings.T  # (batch_size, batch_size)
            
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
    eval_dataset_no_pre, _ = load_combined_text_data(transforms.Compose([pretrained_model.pre_process_images]), BATCHSIZE, 42, data_path)
  
    # Run the evals
    pretrained_model_results_eval = universal_eval(pretrained_model,eval_dataset_no_pre)
    pretrained_model_results_test =  universal_eval(pretrained_model,test_dataset_no_pre)
    slake_trained_model_results = universal_eval(clip,eval_loader_pre)
    slake_trained_model_results_test = universal_eval(clip,test_dataset_pre)

    # Outpput results
    print(pretrained_model_results_test)
    print(pretrained_model_results_eval)
    print(slake_trained_model_results)
    print(slake_trained_model_results_test)