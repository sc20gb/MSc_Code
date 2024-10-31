import torch
import os
from transformers import AutoTokenizer
from Model_Defs.CLIP import CLIP
import torchvision.transforms as transforms
from Data_Loading.data_loading import load_combined_text_data, load_test_data, load_clip_eval_test_data, load_chest_mnist_data, load_data
from Model_Defs.CLIP_with_LORA import CLIPWithLoRA
from sklearn.metrics import accuracy_score, f1_score

#evaluate the clip modelson the dataset provided
def universal_eval_not_classification(model, dataset):
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
                # True labels as indices, assuming each (image, text) pair should match positionally
                true_class_indices = torch.arange(len(texts)).cpu()
            
            # Otherwise, assume it's the second dataloader format
            elif len(batch) == 3:
                images, masks, texts = batch
                # True labels as indices, assuming each (image, text) pair should match positionally
                true_class_indices = torch.arange(len(texts)).cpu()

            elif len(batch) == 2:
                images, true_class_indices = batch
                texts = dataset.dataset.get_classes()

            elif len(batch) == 4:
                images, masks, question, answer = batch
                texts = [q + " " + a  for q,a in zip(question,answer)]
                true_class_indices = torch.arange(len(texts)).cpu()

                

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
            
            # Append predictions and true labels for metrics
            all_preds.extend(predicted_class_indices)
            all_labels.extend(true_class_indices)

    # Calculate metrics
    results["accuracy"] = accuracy_score(all_labels, all_preds) * 100
    results["f1"] = f1_score(all_labels, all_preds, average='weighted')  # Use 'macro' or 'micro' if appropriate


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

    #Load single record test data
    data_path = os.path.join(os.getcwd(), 'Slake1.0')
    test_dataset_pre = load_test_data(transforms.Compose([transforms.Resize((IMAGESIZE, IMAGESIZE)),transforms.ToTensor()]), BATCHSIZE, 42, data_path)
    test_dataset_no_pre = load_test_data(transforms.Compose([pretrained_model.pre_process_images]),BATCHSIZE,42,data_path)

    # Load the evaluation dataset (image-text pairs)
    train_loader_pre1, eval_loader_pre= load_combined_text_data(transforms.Compose([transforms.Resize((IMAGESIZE, IMAGESIZE)),transforms.ToTensor()]), BATCHSIZE, 42, data_path)
    train_loader_no_pre, eval_dataset_no_pre = load_combined_text_data(transforms.Compose([pretrained_model.pre_process_images]), BATCHSIZE, 42, data_path)
  
    #load combined records test data
    test_dataset_combined_pre = load_clip_eval_test_data(transforms.Compose([transforms.Resize((IMAGESIZE, IMAGESIZE)),transforms.ToTensor()]), BATCHSIZE, 42, data_path)
    test_dataset_combined_no_pre = load_clip_eval_test_data(transforms.Compose([pretrained_model.pre_process_images]), BATCHSIZE, 42, data_path)

    #Load external data from https://github.com/MedMNIST/MedMNIST we use the chest version of size 224 by 224
    external_data_pre = load_chest_mnist_data(transforms.Compose([transforms.Resize((IMAGESIZE, IMAGESIZE)),transforms.ToTensor(),transforms.Lambda(lambda x: x.repeat(3, 1, 1))]), 14, 42)
    external_data_no_pre = load_chest_mnist_data(transforms.Compose([pretrained_model.pre_process_images]), 14, 42)

    #Load train and eval single record data
    train_loader_pre, validate_loader_pre = load_data(transforms.Compose([transforms.Resize((IMAGESIZE, IMAGESIZE)),transforms.ToTensor()]), BATCHSIZE, 42, data_path)
    train_loader_no_pre, validate_loader_no_pre = load_data(transforms.Compose([pretrained_model.pre_process_images]), BATCHSIZE, 42, data_path)

    # Run the evals
    pretrained_model_results_eval = universal_eval_not_classification(pretrained_model,eval_dataset_no_pre)
    pretrained_model_results_test =  universal_eval_not_classification(pretrained_model,test_dataset_no_pre)
    pretrained_model_results_combined_test = universal_eval_not_classification(pretrained_model,test_dataset_combined_no_pre)
    pretrained_model_results_train = universal_eval_not_classification(pretrained_model,train_loader_no_pre)
    pretrained_model_results_not_combined_train = universal_eval_not_classification(pretrained_model,train_loader_no_pre)
    pretrained_model_results_not_combined_eval = universal_eval_not_classification(pretrained_model,validate_loader_no_pre)
    pretrained_model_results_MINST = universal_eval_not_classification(pretrained_model,external_data_no_pre)

    slake_trained_model_results = universal_eval_not_classification(clip,eval_loader_pre)
    slake_trained_model_results_test = universal_eval_not_classification(clip,test_dataset_pre)
    slake_trained_model_results_combined_test = universal_eval_not_classification(clip,test_dataset_combined_pre)
    slake_trained_model_results_train = universal_eval_not_classification(clip,train_loader_pre1)
    slake_trained_model_results_MINST = universal_eval_not_classification(clip,external_data_pre)
    slake_trained_model_results_not_combined_train = universal_eval_not_classification(pretrained_model,train_loader_pre)
    slake_trained_model_results_not_combined_eval = universal_eval_not_classification(pretrained_model,validate_loader_pre)

    # Output results
    print("Pretrained Clip model:")
    print("eval combined:", pretrained_model_results_eval)
    print("test single records:", pretrained_model_results_test)
    print("test combined records:", pretrained_model_results_combined_test)
    print("train combined records:", pretrained_model_results_train)
    print("external data minst:", pretrained_model_results_MINST)
    print("train data single records",pretrained_model_results_not_combined_train)
    print("eval data single records", pretrained_model_results_not_combined_eval)
    
    # Output results
    print("Slake Trained Clip model:")
    print("eval combined:", slake_trained_model_results)
    print("test single records:", slake_trained_model_results_test)
    print("test combined records:", slake_trained_model_results_combined_test)
    print("train combined records:", slake_trained_model_results_train)
    print("external data  minst:", slake_trained_model_results_MINST)
    print("train data single records",slake_trained_model_results_not_combined_train)
    print("eval data single records", slake_trained_model_results_not_combined_eval)

# Pretrained Clip model:
# eval combined: {'accuracy': 13.541666666666666, 'prec': 0.0, 'recall': 0.0, 'f1': 0.0963837568520231}
# test single records: {'accuracy': 8.011310084825636, 'prec': 0.0, 'recall': 0.0, 'f1': 0.07739885808284036}     
# test combined records: {'accuracy': 11.458333333333332, 'prec': 0.0, 'recall': 0.0, 'f1': 0.07262806637806636}  
# train combined records: {'accuracy': 7.277902012604189, 'prec': 0.0, 'recall': 0.0, 'f1': 0.07170543410702383}  
# external data minst: {'accuracy': 1.306111532117862, 'prec': 0.0, 'recall': 0.0, 'f1': 0.004407566555936079}    
# train data single records {'accuracy': 7.501524700142305, 'prec': 0.0, 'recall': 0.0, 'f1': 0.07353493514439682}
# eval data single records {'accuracy': 7.502374169040836, 'prec': 0.0, 'recall': 0.0, 'f1': 0.07140372979427836} 
# Slake Trained Clip model:
# eval combined: {'accuracy': 11.458333333333332, 'prec': 0.0, 'recall': 0.0, 'f1': 0.09771669302919302}
# test single records: {'accuracy': 3.016022620169651, 'prec': 0.0, 'recall': 0.0, 'f1': 0.02868925005250092}
# test combined records: {'accuracy': 2.083333333333333, 'prec': 0.0, 'recall': 0.0, 'f1': 0.021875000000000002}
# train combined records: {'accuracy': 2.8664362675340516, 'prec': 0.0, 'recall': 0.0, 'f1': 0.028327924177837496}
# external data  minst: {'accuracy': 0.7399812775821335, 'prec': 0.0, 'recall': 0.0, 'f1': 0.00015080903719492622}
# train data single records {'accuracy': 7.745476722911161, 'prec': 0.0, 'recall': 0.0, 'f1': 0.07653284629603808}
# eval data single records {'accuracy': 7.977207977207977, 'prec': 0.0, 'recall': 0.0, 'f1': 0.07913604810606456}