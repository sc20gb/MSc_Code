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
        metrics_training = None 
        metrics_validate = None
        
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
    
            output, loss, token_prediction_loss, regularisation_loss,reconstructed_image_embeddings, projected_img_embeddings = connector_llm(embeddings.to(device_llm), questions, answer_)
    
            count_t += 1
    
            loss.backward()

            batch_metrics = Metrics(
                loss=loss.detach().to('cpu'),
                token_prediction_loss=token_prediction_loss.detach().to('cpu'),
                regularisation_loss=regularisation_loss.detach().to('cpu'),
                original_embedding=embeddings.detach().to('cpu'),
                restored_projected_embedding=reconstructed_image_embeddings.detach().to('cpu'),
                projected_embedding=projected_img_embeddings.detach().to('cpu'),
                **calc_loss_and_metrics(
                    list(output.to('cpu')), list(answer_.to('cpu')), tokenizer=connector_llm.tokenizer
                )
            )
            
            if metrics_training is None:
                metrics_training = batch_metrics
            else:
                metrics_training += batch_metrics

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
                ).input_ids.to(device_llm)[:, 1:]
        
                eos_tensor = torch.full(
                    (answer_.size(0), 1),
                    connector_llm.tokenizer.eos_token_id,
                    dtype=torch.long,
                    device=device_llm
                )
                answer_ = torch.cat([answer_, eos_tensor], dim=1)
        
                # Updated to capture all six return values from forward pass
                output, loss, token_prediction_loss, regularisation_loss, reconstructed_image_embeddings, projected_img_embeddings = connector_llm(embeddings.to(device_llm), questions, answer_)
                
                metrics = Metrics(
                loss=loss.detach().to('cpu'),
                original_embedding=embeddings.detach().to('cpu'),
                restored_projected_embedding=reconstructed_image_embeddings.detach().to('cpu'),
                projected_embedding=projected_img_embeddings.detach().to('cpu'),
                token_prediction_loss=token_prediction_loss.detach().to('cpu'),
                regularisation_loss=regularisation_loss,
                **calc_loss_and_metrics(list(output.to('cpu')), list(answer_.to('cpu')), tokenizer=connector_llm.tokenizer)
                )

                if metrics_validate is None:
                    metrics_validate = metrics
                else:
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
                    if metrics_training is not None and metrics_validate is not None:

                        wandb.log((metrics_training/count_t).get_log("training_") |
                                (metrics_validate/count_v).get_log("validate_"))
                        
                if metrics_training is not None and metrics_validate is not None:
                    metrics_train_list.append(metrics_training / count_t)
                    metrics_val_list.append(metrics_validate / count_v)

                else:
                    # throw an error if the metrics are not calculated
                    raise ValueError("Metrics were non and could not be passed to wandb or added to the metrics list")

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
            wandb.init(project="TinyLLama_CLIP_3_stages", config=stage_args, name=run_name, sync_tensorboard=True)

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

def cross_val_multi_stage_training(para, n_splits=3, per_data=1.0):
    """
    Performs k-fold cross validation for multi-stage feature alignment training.
    Works with pre-created dataloaders instead of datasets.
    
    Args:
        para (dict): Training parameters dictionary
        n_splits (int): Number of folds for cross-validation
        per_data (float): Percentage of data to use (between 0.0 and 1.0)
        
    Returns:
        tuple: (avg_training_metrics, avg_validation_metrics) - Average metrics across all folds
    """
    # Check required parameters
    if 'training_stages' not in para:
        raise KeyError("Missing required key: 'training_stages'")
    
    if not (0.0 < per_data <= 1.0):
        raise ValueError(f"per_data must be between 0.0 and 1.0, got {per_data}")
    
    if 'cross_val' not in para:
        para['cross_val'] = True
    
    stages = para['training_stages']
    
    # Check for required dataloaders based on stages
    if 1 in stages:
        if 'general_train_dataloader' not in para:
            raise KeyError("Stage 1 requires 'general_train_dataloader'")
    
    if 2 in stages or 3 in stages:
        if 'specific_train_dataloader' not in para:
            raise KeyError(f"Stages 2 or 3 require 'specific_train_dataloader'")
    
    # Get the datasets from dataloaders for cross-validation
    from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split
    
    # Extract datasets from dataloaders
    general_dataset = None
    specific_dataset = None
    
    if 1 in stages:
        general_train_dataset = para['general_train_dataloader'].dataset
        if 'general_val_dataloader' in para and para['general_val_dataloader'] is not None:
            general_val_dataset = para['general_val_dataloader'].dataset
            general_dataset = ConcatDataset([general_train_dataset, general_val_dataset])
            print(f"Combined general dataset: training ({len(general_train_dataset)}) + validation ({len(general_val_dataset)}) = {len(general_dataset)}")
        else:
            general_dataset = general_train_dataset
        
        # Apply per_data restriction if needed
        if per_data < 1.0:
            keep_size = int(per_data * len(general_dataset))
            # Ensure at least 5 samples
            original_size = len(general_dataset)
            keep_size = max(5, keep_size)
            keep_size = min(keep_size, original_size)  # Make sure we don't exceed dataset size
            discard_size = original_size - keep_size
            general_dataset, _ = random_split(
                general_dataset, 
                [keep_size, discard_size],
                generator=torch.Generator().manual_seed(para.get("rand_seed", 42))
            )
            print(f"Using {keep_size} samples ({keep_size/original_size*100:.4f}% of general data)")
    
    if 2 in stages or 3 in stages:
        specific_train_dataset = para['specific_train_dataloader'].dataset
        if 'specific_val_dataloader' in para and para['specific_val_dataloader'] is not None:
            specific_val_dataset = para['specific_val_dataloader'].dataset
            specific_dataset = ConcatDataset([specific_train_dataset, specific_val_dataset])
            print(f"Combined specific dataset: training ({len(specific_train_dataset)}) + validation ({len(specific_val_dataset)}) = {len(specific_dataset)}")
        else:
            specific_dataset = specific_train_dataset
        
        # Apply per_data restriction if needed
        if per_data < 1.0:
            keep_size = int(per_data * len(specific_dataset))
            # Ensure at least 5 samples
            original_size = len(specific_dataset)
            keep_size = max(5, keep_size)
            keep_size = min(keep_size, original_size)  # Make sure we don't exceed dataset size
            discard_size = original_size - keep_size
            specific_dataset, _ = random_split(
                specific_dataset, 
                [keep_size, discard_size],
                generator=torch.Generator().manual_seed(para.get("rand_seed", 42))
            )
            print(f"Using {keep_size} samples ({keep_size/original_size*100:.4f}% of specific data)")
    
    # Setup KFold cross-validation
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=para.get("rand_seed", 42))
    
    # Metrics collection - store separate lists for each fold
    all_fold_training_metrics = []
    all_fold_validation_metrics = []
    
    # Generate fold indices
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
        
        # Setup dataloaders for this fold
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
        
        # Store metrics from this fold
        all_fold_training_metrics.append(metrics_training)
        all_fold_validation_metrics.append(metrics_validate)
    
    # Calculate average metrics across all folds
    min_train_len = min(len(metrics) for metrics in all_fold_training_metrics)
    min_val_len = min(len(metrics) for metrics in all_fold_validation_metrics)

    # Create empty MetricsLists to store the averaged metrics
    avg_training_metrics = MetricsList()
    avg_validation_metrics = MetricsList()

    # Average each metric position across all folds
    for i in range(min_train_len):
        # Create a new Metrics object for this position
        avg_train_metric = Metrics()
        # Add up all metrics from the same position across folds
        for fold_metrics in all_fold_training_metrics:
            avg_train_metric += fold_metrics[i]
        # Divide by the number of folds to get the average
        avg_train_metric = avg_train_metric / n_splits
        avg_training_metrics.append(avg_train_metric)

    for i in range(min_val_len):
        # Create a new Metrics object for this position
        avg_val_metric = Metrics()
        # Add up all metrics from the same position across folds
        for fold_metrics in all_fold_validation_metrics:
            avg_val_metric += fold_metrics[i]
        # Divide by the number of folds to get the average
        avg_val_metric = avg_val_metric / n_splits
        avg_validation_metrics.append(avg_val_metric)

    # Log the average metrics to wandb
    wandb.init(project="TinyLLama_CLIP_CrossVal", config=para, name=f"CrossVal_{unique_suffix}", sync_tensorboard=True)
    for i, (train_metric, val_metric) in enumerate(zip(avg_training_metrics, avg_validation_metrics)):
        metrics_dict = train_metric.get_log(f"avg_training_") | val_metric.get_log(f"avg_validate_")
        wandb.log(metrics_dict)
    wandb.finish()
    
    return avg_training_metrics, avg_validation_metrics