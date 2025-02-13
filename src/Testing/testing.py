from utils.device_handler import handle_devices
from Model_Defs.connector_LLM_with_gen import Connector_LLM_With_Gen
from utils.model_loaders import load_image_encoder
from utils.half import handle_half_for_layer_Norm
from utils.metrics import Metrics, calc_loss_and_metrics
from collections import defaultdict

import wandb
import torch

def test(**model_args):
    """
    Test function that loads a full model from checkpoint.
    Expects model_args to contain the following required keys:
      'vicuna_path', 'connector_layers', 'embed_dim', 'image_resolution', 'VERSION',
      'lr', 'eps', 'weight_decay', 'per_warm', 'batch_size', 'vir_batch_size', 'rand_seed',
      'MAX_EPOC', 'pre_trained_connector_path', 'lora_rank', 'lora_dropout', 'lora_alpha',
      'mllm_checkpoint_path', 'hidden_layer_from_end', 'training_step', 'visual_encoder_type',
      'use_half'
      
    Optional keys: 'cpu_only', 'val_dataset', 'train_dataset'
    
    Raises:
        KeyError: If any required key is missing.
        
    Returns:
        tuple: (overall_metrics, category_results)
    """
    required_keys = [
        'vicuna_path', 'connector_layers', 'embed_dim', 'image_resolution', 'VERSION',
        'lr', 'eps', 'weight_decay', 'per_warm', 'batch_size', 'vir_batch_size', 'rand_seed',
        'MAX_EPOC', 'pre_trained_connector_path', 'lora_rank', 'lora_dropout', 'lora_alpha',
        'mllm_checkpoint_path', 'hidden_layer_from_end', 'training_step', 'visual_encoder_type',
        'use_half'
    ]
    missing = [key for key in required_keys if key not in model_args]
    if missing:
        raise KeyError(f"Missing required model_args keys: {missing}")
    
    # Setup devices
    cpu_only = model_args.get('cpu_only', False)
    device_image_encoder, device_llm = handle_devices(cpu_only)
    
    # Unpack required arguments
    vicuna_path              = model_args['vicuna_path']
    connector_layers         = model_args['connector_layers']
    embed_dim                = model_args['embed_dim']
    image_resolution         = model_args['image_resolution']
    VERSION                  = model_args['VERSION']
    lr                       = model_args['lr']
    eps                      = model_args['eps']
    weight_decay             = model_args['weight_decay']
    per_warm                 = model_args['per_warm']
    batch_size               = model_args['batch_size']
    vir_batch_size           = model_args['vir_batch_size']
    rand_seed                = model_args['rand_seed']
    MAX_EPOC                 = model_args['MAX_EPOC']
    pre_trained_connector_path = model_args['pre_trained_connector_path']
    lora_rank                = model_args['lora_rank']
    lora_dropout             = model_args['lora_dropout']
    lora_alpha               = model_args['lora_alpha']
    mllm_checkpoint_path     = model_args['mllm_checkpoint_path']
    hidden_layer_from_end    = model_args['hidden_layer_from_end']
    training_step            = model_args['training_step']
    visual_encoder_type      = model_args['visual_encoder_type']
    use_half                 = model_args['use_half']
    
    # Load model and encoder
    connector_llm = Connector_LLM_With_Gen(
        image_emded_dim=embed_dim,
        llm_path=vicuna_path,
        connector_layers=connector_layers,
        device=device_llm
    )
    connector_llm.to(device_llm)
    connector_llm.apply_lora(rank=lora_rank, dropout=lora_dropout, alpha=lora_alpha)
    
    # Load full model state dict
    if mllm_checkpoint_path:
        state_dict = torch.load(mllm_checkpoint_path)
        connector_llm.load_state_dict(state_dict)
        print(f"Loaded model checkpoint from {mllm_checkpoint_path}")
    else:
        raise ValueError("No model checkpoint provided")
    
    # Load image encoder and data loaders
    # Optional keys for datasets default to None if not provided.
    val_dataset   = model_args.get('val_dataset', None)
    train_dataset = model_args.get('train_dataset', None)
    
    img_encoder, _, _, test_loader = load_image_encoder(
        visual_encoder_type,
        device_image_encoder,
        val_dataset,
        train_dataset,
        image_resolution,
        batch_size,
        rand_seed,
        **model_args
    )
    
    # Handle half precision
    if use_half:
        connector_llm.half()
        handle_half_for_layer_Norm(connector_llm)
        img_encoder.half()
        handle_half_for_layer_Norm(img_encoder)
    
    # Load trained connector weights if provided
    if pre_trained_connector_path:
        connector_llm.load_connector(pre_trained_connector_path)
    
    # Set to evaluation mode
    connector_llm.eval()
    img_encoder.eval()
    
    # Initialize metric accumulators
    metrics_test    = Metrics()
    from collections import defaultdict
    category_metrics = defaultdict(Metrics)
    category_counts  = defaultdict(int)
    count = 0
    
    with torch.no_grad():
        for image_tensor, mask_tensor, question, answer, category in test_loader:
            # Get image features from the image encoder
            _, hidden_states = img_encoder(
                image_tensor.to(device_image_encoder),
                return_hidden_states=True
            )
            # Use the hidden state specified by hidden_layer_from_end
            image_features = hidden_states[(len(hidden_states) - 1) - hidden_layer_from_end]
    
            # Tokenize answer; remove first token for alignment
            answer_ = connector_llm.tokenizer(
                answer,
                padding='longest',
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True
            ).input_ids.to(device_llm)[:, 1:]
    
            # Get prediction and loss from the model
            output, loss = connector_llm(image_features.to(device_llm), question, answer_)
    
            # Calculate metrics
            metrics = Metrics(loss, **calc_loss_and_metrics(
                list(output), list(answer_), tokenizer=connector_llm.tokenizer
            ))
            metrics_test += metrics
    
            # Track per-category metrics
            for cat in category:
                category_metrics[cat] += metrics
                category_counts[cat] += 1
    
            count += 1
    
    # Compute average metrics
    overall_metrics = metrics_test / count
    category_results = {cat: category_metrics[cat] / category_counts[cat] for cat in category_metrics}
    
    return overall_metrics, category_results

def runtest(lamaCausalLM_path, modelpath,test_params):

    wandb.init(project="path_TinyLLama", config=test_params)

    test_params['vicuna_path'] = lamaCausalLM_path
    test_params['mllm_checkpoint_path'] = modelpath
    
    overall_metrics, category_metrics = test(**test_params)
    
    print("\nOverall Test Results:")
    print(overall_metrics)
    print("\nResults by Category:")
    for category, metrics in category_metrics.items():
        print(f"\n{category}:")
        print(metrics)
    
    # Combine all metrics into one dictionary for logging
    log_dict = {
        **overall_metrics.get_log("test_"),
        **{f"test_{cat}_{k}": v 
           for cat, metrics in category_metrics.items() 
           for k, v in metrics.get_log("").items()}
    }
    
    wandb.log(log_dict)
    wandb.finish()


