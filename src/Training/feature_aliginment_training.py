from utils.device_handler import handle_devices
from Model_Defs.connector_LLM_gen_reg import Connector_LLM_With_Gen_Reg
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
import datetime


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

def feature_alignment(**model_args):
    required_keys = [
        'vicuna_path', 'connector_layers', 'embed_dim', 'version',
        'lr', 'eps', 'weight_decay', 'per_warm', 'batch_size', 'vir_batch_size', 'rand_seed',
        'MAX_EPOC', 'pre_trained_connector_dict', 'lora_rank', 'lora_dropout', 'lora_alpha',
        'hidden_layer_from_end', 'training_step', 'use_half','train_LLM','regulisation_constant', 'cross_val'
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
    regulisation_constant = model_args['regulisation_constant']
    cross_val = model_args['cross_val']

    accumulation_steps = vir_batch_size // batch_size

    # Load connector and LLM model
    connector_llm = Connector_LLM_With_Gen_Reg(
        image_emded_dim=embed_dim,
        llm_path=vicuna_path,
        connector_layers=connector_layers,
        device=device_llm,
        regularisation_constant=regulisation_constant
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


    connector_llm.apply_lora(rank=lora_rank, dropout=lora_dropout, alpha=lora_alpha)

    if train_LLM:
        print("Training the connector and LLM")
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
    
    metrics_train_list = MetricsList()
    metrics_val_list = MetricsList()
    for epoch in range(1, MAX_EPOC + 1):
        metrics_training = Metrics()
        metrics_validate = Metrics()
        
        if train_LLM:
            connector_llm.train()
        else:
            connector_llm.connector.train()
    
        count_t = 0
        count_v = 0
        # Initialize counters for caption lengths if in caption training mode.
        optim.zero_grad()
    
        # training loop
        for batch in train_loader:
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
    
            count_t += 1
    
            loss.backward()

            metrics = Metrics(loss.detach().to('cpu'), **calc_loss_and_metrics(
                list(output.to('cpu')), list(answer_.to('cpu')), tokenizer=connector_llm.tokenizer
            ))
            metrics_training += metrics     

            if count_t % accumulation_steps == 0:
                optim.step()
                optim.zero_grad()
                scheduler.step()
                if connector_llm.llm.training:
                    connector_llm.llm.zero_grad()
                loss = loss.detach()

        if (count_t + 1) % accumulation_steps != 0:
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
                ).input.ids.to(device_llm)[:, 1:]
    
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
                if not cross_val:
                    wandb.log((metrics_training/count_t).get_log("training_") |
                                (metrics_validate/count_v).get_log("validate_"))
                metrics_train_list.append(metrics_training / count_t)
                metrics_val_list.append(metrics_validate / count_v)
            else:
                wandb.log(Metrics(-1, -1, -1, -1, -1, -1).get_log("training_") |
                        Metrics(-1, -1, -1, -1, -1, -1).get_log("validate_"))
        
        # Check and log GPU memory usage at the end of each epoch.
    
    state = connector_llm.connector.state_dict()
    # One final GPU memory check after training
            
    return state, metrics_train_list, metrics_val_list

def multi_stage_feature_aliginment_training(**model_args):
    """
    Multi-stage training function for feature alignment.

    Stages:
      Stage 1: Train the connector from scratch on General Images.
      Stage 2: Train the connector from scratch on Medical Images.
      Stage 3: Train the connector and LLM (with LoRA) on Medical Images.

    Allowed stage combinations: [1,2,3], [1,3], [3], etc.
    Requirements in model_args: see original docstring.
    This function alters the VERSION per stage so that each stage's checkpoints
    are saved in distinct folders. For stages 2 and 3, if saving is enabled the latest
    checkpoint from a previous stage is chained as pre_trained_connector_dict.
    Additionally, each stage creates its own wandb run (project "TinyLLama_CLIP_3_stages")
    to log the metrics.
    """
    metrics_training_list, metrics_validate_list = MetricsList(), MetricsList()

    if 'training_stages' not in model_args:
        raise KeyError("Missing required key: 'training_stages'")
    stages = model_args['training_stages']
    if not isinstance(stages, list) or not stages:
        raise ValueError("'training_stages' must be a non-empty list of stage numbers")

    base_version = model_args['version']
    latest_checkpoint = None
    stage_params = model_args.get("stage_params", {})
    save_dir = model_args.get('save_dir', os.getcwd())

    # Create a unique wandb run name using a timestamp.
    unique_suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    for stage in stages:
        stage_args = model_args.copy()
        print(f"\n===== Starting Training Stage {stage} =====")

        # Override global parameters with stage-specific ones, if provided.
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
            if "general_batch_size" in stage_args:
                stage_args["batch_size"] = stage_args["general_batch_size"]
            if "general_vir_batch_size" in stage_args:
                stage_args["vir_batch_size"] = stage_args["general_vir_batch_size"]

        elif stage == 2:
            # Stage 2: Train connector from scratch on Medical Images.
            if 'specific_train_dataloader' not in stage_args or 'specific_val_dataloader' not in stage_args:
                raise KeyError("Stage 2 requires keys 'specific_train_dataloader' and 'specific_val_dataloader'")
            stage_args['train_loader'] = stage_args['specific_train_dataloader']
            stage_args['val_loader'] = stage_args.get('specific_val_dataloader', None)
            stage_args['training_step'] = 2
            stage_args['pre_trained_connector_dict'] = latest_checkpoint
            stage_args['version'] = f"{base_version}_stage2"
            stage_args['train_LLM'] = False
            if "specific_batch_size" in stage_args:
                stage_args["batch_size"] = stage_args["specific_batch_size"]
            if "specific_vir_batch_size" in stage_args:
                stage_args["vir_batch_size"] = stage_args["specific_vir_batch_size"]

        elif stage == 3:
            # Stage 3: Train connector and LLM (with LoRA) on Medical Images.
            if 'specific_train_dataloader' not in stage_args or 'specific_val_dataloader' not in stage_args:
                raise KeyError("Stage 3 requires keys 'specific_train_dataloader' and 'specific_val_dataloader'")
            stage_args['train_loader'] = stage_args['specific_train_dataloader']
            stage_args['val_loader'] = stage_args.get('specific_val_dataloader', None)
            stage_args['training_step'] = 3  # Train both connector and LLM.
            stage_args['pre_trained_connector_dict'] = latest_checkpoint
            stage_args['version'] = f"{base_version}_stage3"
            stage_args['train_LLM'] = True
            if "specific_batch_size" in stage_args:
                stage_args["batch_size"] = stage_args["specific_batch_size"]
            if "specific_vir_batch_size" in stage_args:
                stage_args["vir_batch_size"] = stage_args["specific_vir_batch_size"]

        else:
            raise ValueError(f"Unsupported stage: {stage}")

        stage_args['save_dir'] = save_dir
        run_name = f"Stage_{stage}_{unique_suffix}"

        if not stage_args['cross_val']:
            wandb.init(project="TinyLLama_CLIP_3_stages", config=stage_args, name=run_name)

        # Run training for the current stage.
        latest_checkpoint, metrics_training, metrics_validate = feature_alignment(**stage_args)

        # here: Concatenate the stage metric lists to the global lists.
        metrics_training_list.extend(metrics_training)
        metrics_validate_list.extend(metrics_validate)

        print(f"Stage {stage} completed with VERSION: {stage_args['version']}")
        
        if not stage_args['cross_val']:
            wandb.finish()

    print("\n===== Multi-Stage Training Completed =====")
    return metrics_training_list, metrics_validate_list

def cross_val_multi_stage_training(para, n_splits=3):
    """
    Performs k-fold cross validation for multi-stage feature alignment training.
    
    Args:
        para (dict): Training parameters dictionary
        n_splits (int): Number of folds for cross-validation
        
    Returns:
        tuple: (avg_training_metrics, avg_validation_metrics) - Average metrics across all folds
    """
    # Check required parameters
    if 'training_stages' not in para:
        raise KeyError("Missing required key: 'training_stages'")
    
    if 'cross_val' not in para:
        para['cross_val'] = True
    
    stages = para['training_stages']
    
    # Check for required datasets based on stages
    general_dataset = specific_dataset = None
    general_val_dataset = specific_val_dataset = None
    
    # Get datasets and combine training with validation for cross-validation
    if 1 in stages:
        if 'general_dataset' not in para:
            raise KeyError("Stage 1 requires 'general_dataset'")
        general_dataset = para['general_dataset']
        # Get validation dataset if provided
        general_val_dataset = para.get('general_val_dataset')
        if general_val_dataset:
            print(f"Combining general training dataset ({len(general_dataset)} samples) with validation dataset ({len(general_val_dataset)} samples) for cross-validation")
            from torch.utils.data import ConcatDataset
            general_dataset = ConcatDataset([general_dataset, general_val_dataset])
            print(f"Combined general dataset size: {len(general_dataset)}")
    
    if 2 in stages or 3 in stages:
        if 'specific_dataset' not in para:
            raise KeyError(f"Stages 2 or 3 require 'specific_dataset'")
        specific_dataset = para['specific_dataset']
        # Get validation dataset if provided
        specific_val_dataset = para.get('specific_val_dataset')
        if specific_val_dataset:
            print(f"Combining specific training dataset ({len(specific_dataset)} samples) with validation dataset ({len(specific_val_dataset)} samples) for cross-validation")
            from torch.utils.data import ConcatDataset
            specific_dataset = ConcatDataset([specific_dataset, specific_val_dataset])
            print(f"Combined specific dataset size: {len(specific_dataset)}")
    
    # Setup KFold cross-validation
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=para.get("rand_seed", 42))
    
    # Metrics collection
    all_training_metrics = MetricsList()
    all_validation_metrics = MetricsList()
    
    # Generate fold indices for the combined datasets
    if general_dataset:
        general_indices = list(range(len(general_dataset)))
        general_folds = list(kfold.split(general_indices))
    
    if specific_dataset:
        specific_indices = list(range(len(specific_dataset)))
        specific_folds = list(kfold.split(specific_indices))
    
    # Create a unique run ID for this cross-validation session
    unique_suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run training for each fold
    for fold in range(n_splits):
        print(f"\n===== Starting Fold {fold+1}/{n_splits} =====")
        
        # Create fold-specific parameters
        fold_para = para.copy()
        fold_para['version'] = f"{para.get('version', 'default')}_fold{fold+1}"
        
        # Setup dataloaders for this fold from the combined datasets
        if general_dataset:
            train_idx, val_idx = general_folds[fold]
            
            general_train_subset = Subset(general_dataset, train_idx)
            general_val_subset = Subset(general_dataset, val_idx)
            
            print(f"General dataset fold {fold+1}: {len(train_idx)} training samples, {len(val_idx)} validation samples")
            
            fold_para['general_train_dataloader'] = DataLoader(
                general_train_subset, 
                batch_size=para.get("general_batch_size", para.get("batch_size", 32)), 
                shuffle=True, 
                generator=torch.Generator().manual_seed(para.get("rand_seed", 42)),
                num_workers=para.get("num_workers", 1),
                pin_memory=para.get("pin_memory", False)
            )
            
            fold_para['general_val_dataloader'] = DataLoader(
                general_val_subset, 
                batch_size=para.get("general_batch_size", para.get("batch_size", 32)),
                num_workers=para.get("num_workers", 1),
                pin_memory=para.get("pin_memory", False)
            )
        
        if specific_dataset:
            train_idx, val_idx = specific_folds[fold]
            
            specific_train_subset = Subset(specific_dataset, train_idx)
            specific_val_subset = Subset(specific_dataset, val_idx)
            
            print(f"Specific dataset fold {fold+1}: {len(train_idx)} training samples, {len(val_idx)} validation samples")
            
            fold_para['specific_train_dataloader'] = DataLoader(
                specific_train_subset, 
                batch_size=para.get("specific_batch_size", para.get("batch_size", 32)), 
                shuffle=True, 
                generator=torch.Generator().manual_seed(para.get("rand_seed", 42)),
                num_workers=para.get("num_workers", 1),
                pin_memory=para.get("pin_memory", False)
            )
            
            fold_para['specific_val_dataloader'] = DataLoader(
                specific_val_subset, 
                batch_size=para.get("specific_batch_size", para.get("batch_size", 32)),
                num_workers=para.get("num_workers", 1),
                pin_memory=para.get("pin_memory", False)
            )
        
        # Run the multi-stage training for this fold
        print(f"Starting multi-stage training for fold {fold+1}")
        metrics_training, metrics_validate = multi_stage_feature_aliginment_training(**fold_para)
        
        # Accumulate metrics
        if len(all_training_metrics) == 0:  # First fold
            all_training_metrics = metrics_training
            all_validation_metrics = metrics_validate
        else:
            all_training_metrics += metrics_training
            all_validation_metrics += metrics_validate
    
    # Calculate average metrics across all folds
    avg_training_metrics = all_training_metrics / n_splits
    avg_validation_metrics = all_validation_metrics / n_splits
    
    # Log the average metrics to wandb
    wandb.init(project="TinyLLama_CLIP_CrossVal", config=para, name=f"CrossVal_{unique_suffix}")
    for train_metric, val_metric in zip(avg_training_metrics, avg_validation_metrics):
        wandb.log(train_metric.get_log("avg_training_") | val_metric.get_log("avg_validate_"))
    wandb.finish()
    
    return avg_training_metrics, avg_validation_metrics