from utils.device_handler import handle_devices
from Model_Defs.connector_LLM_with_gen import Connector_LLM_With_Gen
from utils.model_loaders import load_image_encoder
from utils.half import handle_half_for_layer_Norm
from utils.scheduer import CustomSchedulerWithWarmup
from utils.metrics import Metrics, calc_loss_and_metrics, MetricsList
from Data_Loading.data_loading import load_data_cross_val
from torch.utils.data import random_split
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

import os
import math
import torch
import wandb
import torchvision.transforms as transforms

# TODO: Create a new version of the feature_aginment function that allows for three diffrent traning stages:
# 1. Training the connector from scratch on General Images
# 2. Training the connector from scratch on Medical Images
# 3. Training the connector and LLM (with LoRA) on Medical Images
# The funnction should be able to handle:
# stage 1 then stage 2 then stage 3
# stage 1 then stage 3
# stage 3 only
# stage 1 then stage 3
def feature_aliginment_training(**model_args):
    """
    Trains the feature alignment model between the visual encoder and LLM.
    This function has been refactored to accept parameters via **model_args.
    
    Required keys:
      'vicuna_path', 'connector_layers', 'embed_dim', 'image_resolution', 'VERSION',
      'lr', 'eps', 'weight_decay', 'per_warm', 'batch_size', 'vir_batch_size', 'rand_seed',
      'MAX_EPOC', 'pre_trained_connector_path', 'lora_rank', 'lora_dropout', 'lora_alpha',
      'hidden_layer_from_end', 'training_step', 'visual_encoder_type', 'use_half'
      
    Optional keys: 'save', 'cpu_only', 'train_dataset', 'val_dataset'
    
    Raises:
        KeyError: If any required key is missing.
        
    Returns:
        tuple: (training_list, validate_list) - Training metrics
    """
    required_keys = [
        'vicuna_path', 'connector_layers', 'embed_dim', 'image_resolution', 'VERSION',
        'lr', 'eps', 'weight_decay', 'per_warm', 'batch_size', 'vir_batch_size', 'rand_seed',
        'MAX_EPOC', 'pre_trained_connector_path', 'lora_rank', 'lora_dropout', 'lora_alpha',
        'hidden_layer_from_end', 'training_step', 'visual_encoder_type', 'use_half'
    ]
    missing = [key for key in required_keys if key not in model_args]
    if missing:
        raise KeyError(f"Missing required model_args keys: {missing}")

    # Setup optional parameters with defaults
    save = model_args.get('save', False)
    cpu_only = model_args.get('cpu_only', False)
    train_dataset = model_args.get('train_dataset', None)
    val_dataset = model_args.get('val_dataset', None)
    
    # Setup devices
    device_image_encoder, device_llm = handle_devices(cpu_only)
    
    # Unpack necessary values
    vicuna_path = model_args['vicuna_path']
    connector_layers = model_args['connector_layers']
    embed_dim = model_args['embed_dim']
    image_resolution = model_args['image_resolution']
    VERSION = model_args['VERSION']
    lr = model_args['lr']
    eps = model_args['eps']
    weight_decay = model_args['weight_decay']
    per_warm = model_args['per_warm']
    batch_size = model_args['batch_size']
    vir_batch_size = model_args['vir_batch_size']
    rand_seed = model_args['rand_seed']
    MAX_EPOC = model_args['MAX_EPOC']
    pre_trained_connector_path = model_args['pre_trained_connector_path']
    lora_rank = model_args['lora_rank']
    lora_dropout = model_args['lora_dropout']
    lora_alpha = model_args['lora_alpha']
    hidden_layer_from_end = model_args['hidden_layer_from_end']
    training_step = model_args['training_step']
    visual_encoder_type = model_args['visual_encoder_type']
    use_half = model_args['use_half']
    
    # Calculate gradient accumulation steps
    accumulation_steps = vir_batch_size // batch_size

    # Load connector and LLM model
    connector_llm = Connector_LLM_With_Gen(
        image_emded_dim=embed_dim,
        llm_path=vicuna_path,
        connector_layers=connector_layers,
        device=device_llm
    )

    # Load image encoder and appropriate data loaders (pass any extra model_args)
    img_encoder, train_loader, validate_loader = load_image_encoder(
        visual_encoder_type,
        device_image_encoder,
        val_dataset,
        train_dataset,
        image_resolution,
        batch_size,
        rand_seed,
        **model_args
    )
    
    # Freeze image encoder
    img_encoder.eval()

    # Enable half precision if requested
    if use_half:
        connector_llm.half()
        handle_half_for_layer_Norm(connector_llm)
        img_encoder.half()
        handle_half_for_layer_Norm(img_encoder)

    # Ensure models are on correct devices
    img_encoder.to(device_image_encoder)
    connector_llm.to(device_llm)

    # If training_step==2, load connector weights and apply LoRA
    if training_step == 2:
        if pre_trained_connector_path is not None:
            print("Loading the connector MLP")
            connector_llm.load_connector(pre_trained_connector_path)
        else:
            print("No connector given, training from scratch")
        connector_llm.apply_lora(rank=lora_rank, dropout=lora_dropout, alpha=lora_alpha)
    else:
        connector_llm.llm.eval()

    total_training_steps = math.ceil(len(train_loader.dataset) / vir_batch_size) * MAX_EPOC
    num_warmup_steps = math.ceil(total_training_steps * per_warm)

    print(f"{num_warmup_steps} warmup steps")
    print(f"{total_training_steps} total steps")
    print(f"{len(train_loader.dataset)/batch_size} train batches")

    if training_step == 2:
        optim = torch.optim.AdamW(connector_llm.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
    else:
        optim = torch.optim.AdamW(connector_llm.connector.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
        
    scheduler = CustomSchedulerWithWarmup(optim, num_warmup_steps=num_warmup_steps,
                                          num_training_steps=total_training_steps, training_step=training_step)

    training_list = MetricsList()
    validate_list = MetricsList()
    
    for epoch in range(1, MAX_EPOC + 1):
        metrics_training = Metrics()
        metrics_validate = Metrics()
        
        if training_step == 2:
            connector_llm.train()
        else:
            connector_llm.connector.train()

        count_t = 0
        optim.zero_grad()

        for image_tensor, mask_tensor, question, answer in train_loader:
            with torch.no_grad():
                _, hidden_states = img_encoder(
                    image_tensor.to(device_image_encoder),
                    return_hidden_states=True
                )
            image_features = hidden_states[(len(hidden_states) - 1) - hidden_layer_from_end]
            answer_ = connector_llm.tokenizer(
                answer,
                padding='longest',
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True
            ).input_ids.to(device_llm)[:, 1:]
            # Add EOS token
            eos_tensor = torch.full(
                (answer_.size(0), 1),
                connector_llm.tokenizer.eos_token_id,
                dtype=torch.long,
                device=device_llm
            )
            answer_ = torch.cat([answer_, eos_tensor], dim=1)

            output, loss = connector_llm(image_features.to(device_llm), question, answer_)
            metrics = Metrics(loss, **calc_loss_and_metrics(
                list(output.to('cpu')), list(answer_.to('cpu')), tokenizer=connector_llm.tokenizer
            ))
            metrics_training += metrics     
            count_t += 1

            loss.backward()
            if count_t % accumulation_steps == 0:
                optim.step()
                optim.zero_grad()
                scheduler.step()
                if connector_llm.llm.training:
                    connector_llm.llm.zero_grad()

        if (count_t + 1) % accumulation_steps != 0:
            optim.step()
            optim.zero_grad()
            scheduler.step()

        count = 0
        connector_llm.eval()
        with torch.no_grad():
            for image_tensor, mask_tensor, question, answer in validate_loader:
                _, hidden_states = img_encoder(
                    image_tensor.to(device_image_encoder),
                    return_hidden_states=True
                )
                image_features = hidden_states[(len(hidden_states) - 1) - hidden_layer_from_end]
                answer_ = connector_llm.tokenizer(
                    answer,
                    padding='longest',
                    truncation=True,
                    return_tensors='pt',
                    add_special_tokens=True
                ).input_ids.to(device_llm)[:, 1:]
                output, loss = connector_llm(image_features.to(device_llm), question, answer_)
                metrics = Metrics(loss, **calc_loss_and_metrics(
                    list(output), list(answer_), tokenizer=connector_llm.tokenizer
                ))
                metrics_validate += metrics     
                count += 1

        # Save checkpoints if needed
        if save:
            if training_step == 2:
                path = os.path.join(os.path.dirname(os.getcwd()), "SavedModels", f"MLLM_V_{VERSION}")
                os.makedirs(path, exist_ok=True)
                torch.save(connector_llm.state_dict(), os.path.join(path, f"MLLM_model{epoch}.pth"))
            else:
                path = os.path.join("/nobackup", "sc20gwb", "Models", "SavedModels", f"C_V_{VERSION}")
                os.makedirs(path, exist_ok=True)
                torch.save(connector_llm.connector.state_dict(), os.path.join(path, f"connector_LLM_model{epoch}.pth"))

        # Log metrics (using wandb as in the original)
        if val_dataset is None or train_dataset is None:
            if count and count_t:
                wandb.log((metrics_training/count_t).get_log("training_") |
                          (metrics_validate/count).get_log("validate_"))
            else:
                wandb.log(Metrics(-1, -1, -1, -1, -1, -1).get_log("training_") |
                          Metrics(-1, -1, -1, -1, -1, -1).get_log("validate_"))
        else:
            training_list.append(metrics_training / count_t)
            validate_list.append(metrics_validate / count)
            
    return training_list, validate_list

def cross_val_train(para, n_splits=3, per_data=1.0):
    """Performs k-fold cross validation training.
    
    Args:
        para (dict): Training parameters
        n_splits (int): Number of folds
        per_data (float): Proportion of data to use
        
    Returns:
        None: Logs results to wandb
    """
    # Load the train and val datasets concatnated
    dataset = load_data_cross_val( transforms.Compose([
            transforms.Resize((para["image_resolution"],para["image_resolution"] )),
            transforms.ToTensor()
        ]), os.path.join(os.path.dirname(os.getcwd()), 'Slake1.0'))
    
    if per_data != 1.0:
    
        # Define the split sizes
        train_size = int(per_data * len(dataset))
        discard_size = len(dataset) - train_size

        # Split the dataset into two parts (e.g., 80% and 20%)
        dataset, _ = random_split(dataset, [train_size, discard_size],torch.Generator().manual_seed(para["rand_seed"]))


    
    #Make sure to shuffle the data with the seed
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=para["rand_seed"])

    summed_list_training = MetricsList()
    summed_list_validate = MetricsList()

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold + 1}/{n_splits}')
        
        # Create data loaders for training and validation
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=para["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(para["rand_seed"]),num_workers=1,  prefetch_factor=1,pin_memory=False)
        val_loader = DataLoader(val_subset, batch_size=para["batch_size"],num_workers=1, prefetch_factor=1, pin_memory=False)
        training_list, validate_list = feature_aliginment_training(**para, train_dataset=train_loader, val_dataset=val_loader)

        if len(summed_list_training) == 0:
            summed_list_training = training_list
            summed_list_validate  = validate_list
        else:
            summed_list_training += training_list
            summed_list_validate  += validate_list

    avg_list_training = summed_list_training /  n_splits
    avg_list_validate = summed_list_validate / n_splits
        
    for (train, val) in zip(avg_list_training, avg_list_validate):
        wandb.log(wandb.log((train).get_log("training_") | (val).get_log("validate_")))
