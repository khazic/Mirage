import os
import re
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str,
                    default='/mnt/workspace/fengli/checkpoints/public_models/Qwen2.5-VL-3B-Instruct')
parser.add_argument('--mission', type=str, default='classification')
parser.add_argument('--output_path', type=str, default='./sptial_qwen25_3b.json')
# 修改为支持多个 json 文件
parser.add_argument('--data_jsons', type=str, nargs='+',
                    default=['/mnt/workspace/fengli/data/1.json', '/mnt/workspace/fengli/data/2.json'])
parser.add_argument('--img_root', type=str, default='Mmoment/Mmoment_dataset/Absolute_relative_positions/')
args = parser.parse_args()

CLASSIFICATION_BASE_PROMPT = (
    'You should choose one of the provided options and output a json string with format {"answer": "<option>"}. '
    'Make sure your answer is exactly one of the options provided, and nothing else. '
    'For example, if the options are "left" and "right", you should output: '
    '```json{"answer": "left"}``` or ```json{"answer": "right"}```.\n'
    'Now the question is: '
)

model, processor = None, None

def load_model(model_path):
    """Load model and processor."""
    if 'Qwen2.5' in model_path:
        print('start loading Qwen2.5')
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
        )
        processor = AutoProcessor.from_pretrained(model_path)
        return model, processor
    else:
        raise ValueError('model_path is not supported')

def load_classification_info():
    """Load dataset info from multiple JSON files.
       Only the fields existing in the JSON file (id, question, options, answer, image) are retained."""
    info_list = []
    for json_file in args.data_jsons:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # data 是一个字典，键如 "P01_01", "P01_02"
        for group in data:
            items = data[group]
            for item in items:
                item['image'] = os.path.join(args.img_root, item['image'])
                info_list.append(item)
    return info_list

def infer_single_img(image_path, question):
    global model, processor
    if model is None or processor is None:
        model, processor = load_model(args.path)
    
    full_question = CLASSIFICATION_BASE_PROMPT + question
    image = Image.open(image_path)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": full_question},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    model_answer = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return model_answer

def parse_result(model_output):
    """Parse model output to extract json string."""
    pattern = r'```json\s*(\{.*?\})\s*```'
    match = re.search(pattern, model_output, re.DOTALL)
    if match:
        json_str = match.group(1)
        result = json.loads(json_str)
        return result
    else:
        return {}

def get_score(model_output, ground_truth):
    """Compare the model output with the ground truth answer (case insensitive)."""
    parsed_res = parse_result(model_output)
    if 'answer' in parsed_res:
        model_res = parsed_res['answer']
        if model_res.strip().lower() == ground_truth.strip().lower():
            return 1
        else:
            return 0
    else:
        print('Error parsing JSON, retry:', model_output)
        return 0

def main():
    final_res = []
    total_score = 0
    total_info = load_classification_info()
    for item in tqdm(total_info, desc='classification inference'):
        image_path = item['image']
        prompt = item['question'] + " Options: " + ", ".join(item['options'])
        ground_truth = item['answer']
        
        model_output = infer_single_img(image_path, prompt)
        score = get_score(model_output, ground_truth)
        total_score += score
        
        result_item = {
            "id": item['id'],
            "question": item['question'],
            "options": item['options'],
            "answer": item['answer'],
            "image": image_path,
            "prompt": prompt,
            "model_output": model_output,
            "score": score
        }
        final_res.append(result_item)
    
    overall_score = total_score / len(total_info) if total_info else 0
    with open(args.output_path, 'w', encoding='utf-8') as f:
        save_dict = {
            "model_name": "Qwen2.5-VL-3B-Instruct",
            "score": overall_score,
            "total_num": len(total_info),
            "detail": final_res
        }
        json.dump(save_dict, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
