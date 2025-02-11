import os
import torch
from Model_Defs.connector_LLM_with_gen import Connector_LLM_With_Gen
import wandb
from collections import defaultdict
from utils.half import handle_half_for_layer_Norm
from utils.metrics import Metrics, calc_loss_and_metrics
from utils.device_handler import handle_devices
from utils.model_loaders import load_image_encoder


from Model_Defs.connector_LLM_with_gen import Connector_LLM_With_Gen
from utils.metrics import Metrics, calc_loss_and_metrics
from utils.half import handle_half_for_layer_Norm

from Training.feature_aliginment_training import feature_aliginment_training, cross_val_train

def test(
        vicuna_path,
        connector_layers,
        embed_dim,
        image_resolution,
        VERSION,
        lr,
        eps,
        weight_decay,
        per_warm,
        batch_size,
        vir_batch_size,
        rand_seed,
        MAX_EPOC,
        pre_trained_connector_path,
        lora_rank,
        lora_dropout,
        lora_alpha,
        mllm_checkpoint_path, # Path to full model checkpoint
        save=False,
        cpu_only=False,
        hidden_layer_from_end=0,
        training_step=2, 
        val_dataset = None,
        train_dataset = None,
        visual_encoder_type="CLIP-trained",
        use_half=True,
        **model_args
):
    """Test function that loads full model from checkpoint"""
    # Setup devices
    device_image_encoder, device_llm = handle_devices(cpu_only)
    
    # Load model and encoder
    connector_llm = Connector_LLM_With_Gen(
        image_emded_dim=embed_dim,
        llm_path=vicuna_path,
        connector_layers=connector_layers, 
        device=device_llm
    )

    connector_llm.to(device_llm)

    connector_llm.apply_lora(rank=lora_rank,dropout=lora_dropout,alpha=lora_alpha)
    
    # Load full model state dict
    if mllm_checkpoint_path:
        state_dict = torch.load(mllm_checkpoint_path)
        connector_llm.load_state_dict(state_dict)
        print(f"Loaded model checkpoint from {mllm_checkpoint_path}")
    else :
        raise ValueError("No model checkpoint provided")
    
    # Load image_encoder and correct data_loaders
    img_encoder, _, _, test_loader = load_image_encoder(visual_encoder_type,device_image_encoder,val_dataset,train_dataset, image_resolution,batch_size,rand_seed,**model_args)


    # Handle half precision
    if use_half:
        connector_llm.half()
        handle_half_for_layer_Norm(connector_llm)
        img_encoder.half()
        handle_half_for_layer_Norm(img_encoder)

    # Load trained weights if provided
    if pre_trained_connector_path:
        connector_llm.load_connector(pre_trained_connector_path)
        
    # Evaluate mode
    connector_llm.eval()
    img_encoder.eval()

    # Test metrics
    metrics_test = Metrics()
    category_metrics = defaultdict(Metrics)
    category_counts = defaultdict(int)
    count = 0
    
    with torch.no_grad():
        
        for image_tensor, mask_tensor, question, answer, category in test_loader:
            # Get image features from the img encoder
                _, hidden_states = img_encoder(image_tensor.to(device_image_encoder),return_hidden_states=True)                    

                # We want the hidden state at the specified layer (len(hidden_states) - 1) is the last layer, so 0 is 0 from the end, 1 one from the end
                image_features = hidden_states[(len(hidden_states) - 1) - hidden_layer_from_end]

                # Tokenize answer
                answer_ = connector_llm.tokenizer(answer, padding='longest', truncation=True, return_tensors='pt', add_special_tokens=True).input_ids.to(device_llm)[:, 1:]

                # Get MLLM prediction and NLL loss
                output, loss  = connector_llm(image_features.to(device_llm), question, answer_)

                # Eval
                metrics = Metrics(loss,**calc_loss_and_metrics(list(output),list(answer_),tokenizer=connector_llm.tokenizer))
                metrics_test += metrics

                # Track per-category metrics
                for cat, met in zip(category, [metrics] * len(category)):
                    category_metrics[cat] += met
                    category_counts[cat] += 1

                count = count + 1


        # Calculate averages
    overall_metrics = metrics_test/count
    category_results = {cat: metrics/category_counts[cat] 
                       for cat, metrics in category_metrics.items()}

    return overall_metrics, category_results

def runtest(lamaCausalLM_path,modelpath):
    
    # Test parameters
    test_params = {
            "vicuna_path": lamaCausalLM_path,
            "connector_layers": 2,
            "embed_dim": 768,
            "image_resolution": 224,
            "VERSION": 1,
            "lr": 0.001,
            "eps": 1e-8,
            "weight_decay": 0.01,
            "per_warm": 0.333,
            "batch_size": 4,
            "pre_trained_connector_path":None,
            "vir_batch_size": 32,
            "rand_seed": 42,
            "MAX_EPOC": 1,
            "mllm_checkpoint_path": modelpath,
            "lora_rank": 4,
            "lora_dropout": 0.3,
            "lora_alpha": 32,
            "save": False,
            "cpu_only": False,
            "hidden_layer_from_end": 1,
            "training_step": 2,
            "visual_encoder_type": "CLIP-pretrained",
            "use_half": False
        }
    
    wandb.init(project="path_TinyLLama",config=test_params)

    # Run test
    overall_metrics, category_metrics = test(**test_params)

    print("\nOverall Test Results:")
    print(overall_metrics)
    print("\nResults by Category:")
    for category, metrics in category_metrics.items():
        print(f"\n{category}:")
        print(metrics)


    # Combine all metrics into one dictionary
    log_dict = {
        **overall_metrics.get_log("test_"),
        **{f"test_{cat}_{k}": v 
           for cat, metrics in category_metrics.items() 
           for k, v in metrics.get_log("").items()}
    }

    # log metrics
    wandb.log(log_dict)
    wandb.finish()

if __name__ == '__main__':
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Modify as needed to run the training script, cross_val_train, or test script:
    
    #Test Script start ###########
    # Construct path to models directory
    path_TinyLLama_LOCAL = os.path.join(os.path.dirname(os.getcwd()), "Models", "TinyLLama-v1.0")

    path_TinyLLama_ARC = os.path.join("/nobackup", "sc20gwb", "Models", "TinyLLama-v1.0")

    # The path of the LLM to use. Must be a LlamaForCausalLM model
    lamaCausalLM_path = path_TinyLLama_LOCAL


    #for i in range(1, 21):
    # Test the model, only run testing code
    modelpath = os.path.join(os.path.dirname(os.getcwd()), "SavedModels", "MLLM_V_3000", "MLLM_model"  + str(20) + ".pth")
    runtest(lamaCausalLM_path,modelpath)
    # end script, remove testing code to run training script
    exit()

    #Training Script start ###########

    #Connector parameters
    HIDDEN_LAYER_LIST = [1]
    CONNECTOR_LAYERS_LIST = [2]

    #Training parameters
    LR_LIST = [0.001]
    WEIGHT_DECAY_LIST = [0.01]
    PERC_WARM_LIST = [0.3333]
    VIR_BATCH_SIZE_LIST = [64]

    #LoRA parameters
    DROPOUT_LIST = [0.3,0.0,0.5]
    RANK_LIST = [8,4,12]
    LORA_ALPHA_LIST =  [32,16,8]

    optim_list = [{
            "vicuna_path":lamaCausalLM_path,
            "connector_layers":cl,
            "embed_dim":768,
            "image_resolution":224,
            "lr": lr,
            "eps":1e-8,
            "weight_decay":wd,
            "per_warm": pw,
            "batch_size":4,
            "vir_batch_size":vb,
            "rand_seed":42,
            "MAX_EPOC":20,
            "VERSION":3000,
            "save":True,
            "cpu_only":False,
            "hidden_layer_from_end": hl,
            "training_step":2,
            "lora_dropout":do,
            "lora_rank":r,
            "pre_trained_connector_path":None,
            "lora_alpha": a,
            "visual_encoder_type": "CLIP-pretrained",
            "use_half" : False
                }
                for lr in LR_LIST 
                for wd in WEIGHT_DECAY_LIST 
                for cl in CONNECTOR_LAYERS_LIST
                for pw in PERC_WARM_LIST
                for vb in VIR_BATCH_SIZE_LIST
                for hl in HIDDEN_LAYER_LIST
                for do in DROPOUT_LIST
                for r in  RANK_LIST
                for a in LORA_ALPHA_LIST
                ]
    # Remove the first 3
    optim_list = optim_list[3:]

    for i, para in enumerate(optim_list):
        para['VERSION'] += i
        wandb.init(project="path_TinyLLama",config=para)
        #print("Cross Validation for VERSION ", para["VERSION"])
        feature_aliginment_training(**para)
        #cross_val_train(para,n_splits=3,per_data=1.0)
        wandb.finish()