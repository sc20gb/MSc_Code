import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from torch.utils.data import DataLoader
from torchvision import transforms 
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
from PIL import Image
import math
from transformers import set_seed, logging

logging.set_verbosity_error()


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_test_data(transform, batch_size, seed, data_dir):
    """
    Loads test data for evaluation. Assumes that the dataset is in JSON format.
    Args:
        transform (callable): Image transformation to be applied.
        batch_size (int): Number of samples per batch.
        seed (int): Random seed for shuffling.
        data_dir (str): Directory containing the test dataset JSON file.
    Returns:
        test_loader (DataLoader): DataLoader object for test data.
    """
    test_json_path = os.path.normpath(os.path.join(data_dir, 'test.json'))

    # Create Dataset objects
    test_dataset = JsonDatasetTest(test_json_path, transform)

    # Create DataLoader objects
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))

    return test_loader


def eval_model_with_dataloader(args, test_loader):
    set_seed(0)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    ans_file = open(args.answers_file, "w")

    for batch in tqdm(test_loader):
        images, masks, questions, answers, categories = batch
        for idx, question in enumerate(questions):
            image = images[idx]
            question_text = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
            cur_prompt = question_text

            if model.config.mm_use_im_start_end:
                question_text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question_text
            else:
                question_text = DEFAULT_IMAGE_TOKEN + '\n' + question_text

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], question_text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            # Process the image tensor using the image processor
            image_tensor = process_images([image], image_processor, model.config)[0]

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            # Inference
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                       "prompt": cur_prompt,
                                       "text": outputs,
                                       "answer_id": ans_id,
                                       "model_id": model_name,
                                       "metadata": {}}) + "\n")
            ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    # Dataloader-specific arguments
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing the test data.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data shuffling.")
    args = parser.parse_args()

    # Assuming you have defined a proper image transformation function:
    # Example: transform = some_transform_function()

    # Load test data
    test_loader = load_test_data(transforms.Compose([]), args.batch_size, args.seed, args.data_dir)

    eval_model_with_dataloader(args, test_loader)
