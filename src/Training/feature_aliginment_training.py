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

# LEGACY CODE TODO: Remove or refactor
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
    save_dir = model_args.get('save_dir', os.getcwd())
    
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

            del output, answer_

            loss.backward()
            if count_t % accumulation_steps == 0:
                optim.step()
                optim.zero_grad()
                scheduler.step()
                if connector_llm.llm.training:
                    connector_llm.llm.zero_grad()
                # Detach the loss, output, and answer if not needed further
                loss = loss.detach()
                output = output.detach()
                answer_ = answer_.detach()
                # Optionally force synchronization then collect garbage
                torch.cuda.synchronize()
                import gc
                gc.collect()
                torch.cuda.empty_cache()

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
                path = os.path.join(save_dir, "SavedModels", f"MLLM_V_{VERSION}")
                os.makedirs(path, exist_ok=True)
                torch.save(connector_llm.state_dict(), os.path.join(path, f"MLLM_model{epoch}.pth"))
            else:
                path = os.path.join(save_dir, "SavedModels", f"C_V_{VERSION}")
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

# LEGACY CODE TODO: Remove or refactor
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


import torch

def check_gpu_memory(context=""):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        total_mem = props.total_memory  # in bytes
        reserved_mem = torch.cuda.memory_reserved()  # bytes reserved by the caching allocator
        allocated_mem = torch.cuda.memory_allocated()  # bytes actually allocated for tensors
        free_mem = total_mem - reserved_mem  # approximate free memory available for new allocations
        print(f"[GPU MEMORY {context}]: allocated = {allocated_mem/1024/1024:.2f} MB, "
              f"total = {total_mem/1024/1024:.2f} MB, free ≈ {free_mem/1024/1024:.2f} MB")
        torch.cuda.empty_cache()

def feature_alignment(**model_args):
    required_keys = [
        'vicuna_path', 'connector_layers', 'embed_dim', 'version',
        'lr', 'eps', 'weight_decay', 'per_warm', 'batch_size', 'vir_batch_size', 'rand_seed',
        'MAX_EPOC', 'pre_trained_connector_dict', 'lora_rank', 'lora_dropout', 'lora_alpha',
        'hidden_layer_from_end', 'training_step', 'use_half','train_LLM'
    ]
    missing = [key for key in required_keys if key not in model_args]
    if missing:
        raise KeyError(f"Missing required model_args keys: {missing}")

    # Setup optional parameters with defaults
    save = model_args.get('save', False)
    cpu_only = model_args.get('cpu_only', False)
    train_loader = model_args.get('train_loader', None)
    val_loader = model_args.get('val_loader', None)
    save_dir = model_args.get('save_dir', os.getcwd())

    # Setup devices
    _, device_llm = handle_devices(cpu_only)
    
    # Unpack necessary values
    vicuna_path = model_args['vicuna_path']
    connector_layers = model_args['connector_layers']
    embed_dim = model_args['embed_dim']
    VERSION = model_args['version']
    lr = model_args['lr']
    eps = model_args['eps']
    weight_decay = model_args['weight_decay']
    per_warm = model_args['per_warm']
    batch_size = model_args['batch_size']
    vir_batch_size = model_args['vir_batch_size']
    MAX_EPOC = model_args['MAX_EPOC']
    pre_trained_connector_dict = model_args['pre_trained_connector_dict']
    lora_rank = model_args['lora_rank']
    lora_dropout = model_args['lora_dropout']
    lora_alpha = model_args['lora_alpha']
    training_step = model_args['training_step']
    use_half = model_args['use_half']
    train_LLM = model_args['train_LLM']

    accumulation_steps = vir_batch_size // batch_size

    # Load connector and LLM model
    connector_llm = Connector_LLM_With_Gen(
        image_emded_dim=embed_dim,
        llm_path=vicuna_path,
        connector_layers=connector_layers,
        device=device_llm
    )
    

    # Enable half precision if requested
    if use_half:
        connector_llm.half()
        handle_half_for_layer_Norm(connector_llm)
    
# Ensure model is on correct device
    connector_llm.to(device_llm)
    
# load the connector weights if provided
    if pre_trained_connector_dict is not None:
        print("Loading the connector MLP using provided state dict")
        connector_llm.connector.load_state_dict(pre_trained_connector_dict)

    if train_LLM:
        print("Training the connector and LLM")
        connector_llm.apply_lora(rank=lora_rank, dropout=lora_dropout, alpha=lora_alpha)
        optim = torch.optim.AdamW(connector_llm.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
    else:
        print("Training the connector only")
        connector_llm.llm.eval()
        optim = torch.optim.AdamW(connector_llm.connector.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
    
# calculate the total training steps and the number of warmup steps
    total_training_steps = math.ceil(len(train_loader.dataset) / vir_batch_size) * MAX_EPOC
    num_warmup_steps = math.ceil(total_training_steps * per_warm)

    print(f"{num_warmup_steps} warmup steps")
    print(f"{total_training_steps} total steps")
    print(f"{len(train_loader.dataset)/batch_size} train batches")
    
    scheduler = CustomSchedulerWithWarmup(optim, num_warmup_steps=num_warmup_steps,
                                          num_training_steps=total_training_steps, training_step=training_step)
    

    check_gpu_memory(context="Before Training")
    
    for epoch in range(1, MAX_EPOC + 1):
        metrics_training = Metrics()
        metrics_validate = Metrics()
        
        if train_LLM:
            connector_llm.train()
        else:
            connector_llm.connector.train()
    
        count_t = 0
        count_v = 0
        optim.zero_grad()
    
        # training loop
        for batch in train_loader:
            check_gpu_memory(context=f"Training {count_t}")
            embeddings = batch[0]
            if training_step == 1:
                answers = batch[1]['batch_data']['data_1']
                questions = ["Generate a caption for the image" for _ in answers]
            else:
                questions = batch[1]['batch_data']['data_2']
                answers = batch[1]['batch_data']['data_3']
    
            answer_ = connector_llm.tokenizer(
                answers,
                padding='longest',
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True
            ).input_ids.to(device_llm)[:, 1:]
    
            eos_tensor = torch.full(
                (answer_.size(0), 1),
                connector_llm.tokenizer.eos_token_id,
                dtype=torch.long,
                device=device_llm
            )
            answer_ = torch.cat([answer_, eos_tensor], dim=1)
    
            output, loss = connector_llm(embeddings.to(device_llm), questions, answer_)
            metrics = Metrics(loss, **calc_loss_and_metrics(
                list(output.to('cpu')), list(answer_.to('cpu')), tokenizer=connector_llm.tokenizer
            ))
    
            metrics_training += metrics     
            count_t += 1
    
            loss.backward()
            if count_t % accumulation_steps == 0:
                print("Optimizing")
                optim.step()
                optim.zero_grad()
                scheduler.step()
                if connector_llm.llm.training:
                    connector_llm.llm.zero_grad()
                # Detach the loss, output, and answer if not needed further
                loss = loss.detach()
                output = output.detach()
                answer_ = answer_.detach()
                # Optionally force synchronization then collect garbage
                torch.cuda.synchronize()
                import gc
                gc.collect()
                torch.cuda.empty_cache()
    
        if (count_t + 1) % accumulation_steps != 0:
            print("Optimizing")
            optim.step()
            optim.zero_grad()
            scheduler.step()
    
        connector_llm.eval()
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch[0]
                if training_step == 1:
                    answers = batch[1]['batch_data']['data_1']
                    questions = ["Generate a caption for the image" for _ in answers]
                else:
                    questions = batch[1]['batch_data']['data_2']
                    answers = batch[1]['batch_data']['data_3']
    
                answer_ = connector_llm.tokenizer(
                    answers,
                    padding='longest',
                    truncation=True,
                    return_tensors='pt',
                    add_special_tokens=True
                ).input_ids.to(device_llm)[:, 1:]
    
                eos_tensor = torch.full(
                    (answer_.size(0), 1),
                    connector_llm.tokenizer.eos_token_id,
                    dtype=torch.long,
                    device=device_llm
                )
                answer_ = torch.cat([answer_, eos_tensor], dim=1)
    
                output, loss = connector_llm(embeddings.to(device_llm), questions, answer_)
                metrics = Metrics(loss, **calc_loss_and_metrics(
                    list(output), list(answer_), tokenizer=connector_llm.tokenizer
                ))
                metrics_validate += metrics     
                count_v += 1
    
        if save:
            if train_LLM:
                path = os.path.join(save_dir, "SavedModels", f"MLLM_V_{VERSION}")
                os.makedirs(path, exist_ok=True)
                torch.save(connector_llm.state_dict(), os.path.join(path, f"MLLM_model{epoch}.pth"))
            else:
                path = os.path.join(save_dir, "SavedModels", f"C_V_{VERSION}")
                os.makedirs(path, exist_ok=True)
                torch.save(connector_llm.connector.state_dict(), os.path.join(path, f"connector_LLM_model{epoch}.pth"))
    
        if count_v and count_t:
            wandb.log((metrics_training/count_t).get_log("training_") |
                      (metrics_validate/count_v).get_log("validate_"))
        else:
            wandb.log(Metrics(-1, -1, -1, -1, -1, -1).get_log("training_") |
                      Metrics(-1, -1, -1, -1, -1, -1).get_log("validate_"))
        
        # Check and log GPU memory usage at the end of each epoch.
        check_gpu_memory(context=f"Epoch {epoch}")
    
    state = connector_llm.connector.state_dict()
    # One final GPU memory check after training
    check_gpu_memory(context="Final")
            
    return state


def multi_stage_feature_aliginment_training(**model_args):
    """
    Multi-stage training function for feature alignment.

    Stages:
      Stage 1: Train the connector from scratch on General Images.
      Stage 2: Train the connector from scratch on Medical Images.
      Stage 3: Train the connector and LLM (with LoRA) on Medical Images.

    Allowed stage combinations: [1,2,3], [1,3], [3], etc.

    Requirements in model_args:
      - 'training_stages': list of stage numbers to run (e.g., [1,3])
      - For stage 1: 'general_dataset' (and optionally 'general_val_dataset')
      - For stage 2/3: 'medical_dataset' (and optionally 'medical_val_dataset')
      - Optionally, 'stage_params': a dict mapping stage number to a dict of:
            { 'lr': value, 'eps': value, 'weight_decay': value, 'per_warm': value, 'MAX_EPOC': value }
            Any missing value in a stage-specific dict will fall back to the global one.
    
    Other required keys are the same as for feature_aliginment_training.

    This function alters the VERSION per stage so that each stage's checkpoints
    are saved in distinct folders. For stages 2 and 3, if saving is enabled the latest
    checkpoint from a previous stage is chained as pre_trained_connector_path.
    Additionally, each stage creates its own wandb run (project "TinyLLama_CLIP_3_stage")
    to log the metrics.
    """
    import wandb

    if 'training_stages' not in model_args:
        raise KeyError("Missing required key: 'training_stages'")
    stages = model_args['training_stages']
    if not isinstance(stages, list) or not stages:
        raise ValueError("'training_stages' must be a non-empty list of stage numbers")

    base_version = model_args['version']
    latest_checkpoint = None

    stage_params = model_args.get("stage_params", {})
    save_dir = model_args.get('save_dir', os.getcwd())

    for stage in stages:
        stage_args = model_args.copy()
        print(f"\n===== Starting Training Stage {stage} =====")

        # Override global parameters with stage-specific ones if provided.
        for param in ['lr', 'eps', 'weight_decay', 'per_warm', 'MAX_EPOC']:
            if stage in stage_params and param in stage_params[stage]:
                stage_args[param] = stage_params[stage][param]
        
        if stage == 1:
            # Stage 1: Train connector from scratch on General Images.
            if 'general_dataset' not in stage_args:
                raise KeyError("Stage 1 requires key 'general_dataset'")
            stage_args['train_loader'] = stage_args['general_train_dataloader']
            stage_args['val_loader'] = stage_args.get('general_val_dataloader', None)
            stage_args['training_step'] = 1 
            stage_args['pre_trained_connector_dict'] = None  # Always start from scratch.
            stage_args['version'] = f"{base_version}_stage1"
            stage_args['train_LLM'] = False 
            
        elif stage == 2:
            # Stage 2: Train connector from scratch on Medical Images.
            if 'specific_train_dataloader' not in stage_args or 'specific_val_dataloader' not in stage_args:
                raise KeyError("Stage 2 requires key 'specific_train_dataloader' and 'specific_val_dataloader'")
            stage_args['train_loader'] = stage_args['specific_train_dataloader']
            stage_args['val_loader'] = stage_args.get('specific_val_dataloader', None)
            stage_args['training_step'] = 2  
            stage_args['pre_trained_connector_dict'] = latest_checkpoint
            stage_args['version'] = f"{base_version}_stage2"
            stage_args['train_LLM'] = False 

            
        elif stage == 3:
            # Stage 3: Train connector and LLM (with LoRA) on Medical Images.
            if 'specific_train_dataloader' not in stage_args or 'specific_val_dataloader' not in stage_args:
                raise KeyError("Stage 3 requires key 'specific_train_dataloader' and 'specific_val_dataloader'")
            stage_args['train_loader'] = stage_args['specific_train_dataloader']
            stage_args['val_loader'] = stage_args.get('specific_val_dataloader', None)
            stage_args['training_step'] = 3  # Train both connector and LLM.
            stage_args['pre_trained_connector_dict'] = latest_checkpoint
            stage_args['version'] = f"{base_version}_stage3"
            stage_args['train_LLM'] = True 

        else:
            raise ValueError(f"Unsupported stage: {stage}")
        
        stage_args['save_dir'] = save_dir

        # Initialize a new wandb run for this stage.
        wandb.init(project="TinyLLama_CLIP_3_stages", config=stage_args)
        
        # Run training for the current stage.
        latest_checkpoint = feature_alignment(**stage_args)

        print(f"Stage {stage} completed with VERSION: {stage_args['version']}")
        
        # Finish the wandb run after training.
        wandb.finish()
        

    print("\n===== Multi-Stage Training Completed =====")
    return 0

