import os
import re
import glob
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
# os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str,
                    default='/mnt/workspace/fengli/checkpoints/public_models/Qwen2.5-VL-3B-Instruct')
parser.add_argument('--mission', type=str, default='counting')
parser.add_argument('--output_path', type=str, default='./counting_qwen25_3b.json')
args = parser.parse_args()

counting_json_path = '/mnt/workspace/fengli/data/public_datasets/Mmoment_dataset/counting/counting_data.json'
counting_img_path = '/mnt/workspace/fengli/data/public_datasets/Mmoment_dataset/counting/images'
COUNTING_BASE_PROMPT = 'You should output a json string with format {"answer": a int number}.Your output should be directly parsed by json.loads function. eg.```json{"answer": 1}```.\nNow the question is:'

model, processor = None, None


def load_model(model_path):
    """load model and processor"""
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


def load_counting_info():
    """load dataset info"""
    with open(counting_json_path, 'r', encoding='utf-8') as f:
        json_info = json.load(f)
    for i in range(len(json_info)):
        json_info[i]['image_path'] = os.path.join(counting_img_path, json_info[i]['image_path'])
        assert os.path.exists(json_info[i]['image_path']),json_info[i]['image_path']
    return json_info


# Todo use api change this func
def infer_single_img(image_path, question):
    global model, processor
    if model is None or processor is None:
        model, processor = load_model(args.path)

    question = COUNTING_BASE_PROMPT + question
    image = Image.open(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": question},
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


def parse_result(model_output, ground_truth):
    """parse model output"""
    pattern = r'```json\s*(\{.*?\})\s*```'
    match = re.search(pattern, model_output, re.DOTALL)
    if match:
        json_str = match.group(1)
        result = json.loads(json_str)
        return result
    else:
        return {}


def get_score(model_output, ground_truth):
    """get score"""
    parsed_res = parse_result(model_output, ground_truth)
    if 'answer' in parsed_res:
        model_res = parsed_res['answer']
        if len(ground_truth) == 1:
            ground_res = int(list(ground_truth.keys())[0])
            if int(model_res) == ground_res:
                return 1
            else:
                return 0

        # for hard mission
        ground_count = ground_truth.keys()
        ground_count = [int(i) for i in ground_count]
        ground_score = list(ground_truth.values())

        if ground_count[0] <= model_res < ground_count[1]:
            return ground_score[0]
        elif model_res == ground_count[1]:
            return ground_score[1]
        else:
            return 0

    else:
        print('error json parse, retry:', model_output)
        return 0


def test():
    """unit test"""
    print('start unit test...')
    image1_path = '/mnt/workspace/fengli/data/public_datasets/Mmoment_dataset/counting/images/counting_1.jpg'
    image1_prompt = 'How many ship are in the image?'
    image1_grd = {
        "0": 1
    }

    image2_path = '/mnt/workspace/fengli/data/public_datasets/Mmoment_dataset/counting/images/counting_1.jpg'
    image2_prompt = 'How many plane are in the image?'
    image2_grd = {
        "4": 0.5,
        "5": 1
    }

    image3_path = '/mnt/workspace/fengli/data/public_datasets/Mmoment_dataset/counting/images/counting_1.jpg'
    image3_prompt = 'How many plane are in the image?'
    image3_grd = {
        "2": 0.5,
        "4": 1
    }

    model_output1 = infer_single_img(image1_path, image1_prompt)
    res1 = get_score(model_output1, image1_grd)

    model_output2 = infer_single_img(image2_path, image2_prompt)
    res2 = get_score(model_output2, image2_grd)

    model_output3 = infer_single_img(image3_path, image3_prompt)
    res3 = get_score(model_output3, image3_grd)

    breakpoint()


def main():
    final_res = []
    total_score = 0
    total_image_info = load_counting_info()
    for each_image_info in tqdm(total_image_info,desc='counting inference'):
        image_path = each_image_info['image_path']
        prompt = each_image_info['prompt']
        ground_truth = each_image_info['ground_truth']
        difficulty = each_image_info['difficulty']


        model_output = infer_single_img(image_path, prompt)
        
        score = get_score(model_output, ground_truth)

        total_score += score

        final_res.append({
            "id": each_image_info['id'],
            'image_path': image_path,
            'prompt': prompt,
            'ground_truth': ground_truth,
            'difficulty': difficulty,
            'model_output': model_output,
            'score': score
        })
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        save_dict = {
            "model_name":"Qwen2.5-VL-3B-Instruct",
            "score":total_score/len(total_image_info),
            "total_image_num":len(total_image_info),
            "detail":final_res
        }
        json.dump(save_dict, f, ensure_ascii=False, indent=4)

main()
