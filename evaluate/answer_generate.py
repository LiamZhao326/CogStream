import torch
import json
import os
import re
from torch.distributed import init_process_group
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
import argparse
# torchrun --nproc_per_node=8 answer_generate.py
def natural_sort_segments(folder_path):
    file_list = os.listdir(folder_path)

    pattern = re.compile(r"segment_(\d+)")

    def extract_number(filename):
        match = pattern.search(filename)
        if match:
            return int(match.group(1))
        else:
            return 999999

    sorted_files = sorted(file_list, key=extract_number)
    return sorted_files

def save_to_json(video_name, data, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{video_name}.json")
    data_dict = {"video_name": video_name, "Data": data}
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)

def get_selection_output(selection):
    if_visual = True
    selected_indices = []
    cleaned_str = selection.strip('[]')
    parts = cleaned_str.split(',')
    if parts and parts[0]:
        first_char = parts[0].strip()
        if first_char == 'no':
            if_visual = False
            parts = parts[1:]
        elif first_char == 'yes':
            if_visual = True
            parts = parts[1:]

    for part in parts:
        stripped_part = part.strip()
        if stripped_part:
            try:
                selected_indices.append(int(stripped_part))
            except ValueError:
                continue
    return selected_indices, if_visual

@torch.inference_mode()
def infer(conversation, model, processor, select=None, if_visual=None):
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16) # Match compute dtype
    model.set_adapter("language_module")
    inputs = model.qa_selection(**inputs, mode="FCC", select_gt=select, if_visual=if_visual)
    model.set_adapter("full_module")
    output_ids, selection_module_output = model.generate(**inputs, max_new_tokens=1024)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response, selection_module_output

class VideoDataset(Dataset):
    def __init__(self, video_dir, query_dir):
        self.video_and_querychain = []
        all_files = os.listdir(query_dir)
        for json_file in all_files:
            if json_file.endswith(".json"):
                video_name = json_file.replace(".json", "")
                video_path = os.path.join(video_dir, video_name)
                if os.path.exists(video_path):
                    with open(os.path.join(query_dir, json_file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.video_and_querychain.append({"video_path":video_path, "query_chains":data})
                else:
                    print(f"警告: 视频文件 {video_path} 未找到，对应的 JSON 文件为 {json_file}")
        print(f"总共{len(self.video_and_querychain)}个样本。")

    def __len__(self):
        return len(self.video_and_querychain)

    def __getitem__(self, idx):
        video_path = self.video_and_querychain[idx]["video_path"]
        query_chain = self.video_and_querychain[idx]["query_chains"][0]
        return {"video_path": video_path, "query_chain": query_chain}

def inference(model, dataloader, processor, save_dir):
    with tqdm(total=len(dataloader), desc=f"now generate") as pbar:
        for video_id, batch in enumerate(dataloader):
            data = []
            video_answer = []
            video_path = batch["video_path"][0]
            query_chain = batch["query_chain"]
            conversation = [{"role": "system", "content": "You are a helpful assistant."}]

            segments = {}
            for qa in query_chain:
                t = qa["info"]["Event_Time"].item()
                if t not in segments:
                    segments[t] = []
                segments[t].append(qa)

            sorted_times = sorted(segments.keys())
            hist_num = 0
            for current_query_time, file_name in zip(sorted_times, natural_sort_segments(video_path)):
                querys = [qa["Q"][0] for qa in segments[current_query_time]]
                full_path = os.path.join(video_path, file_name)
                cov = {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": {"video_path": full_path, "fps": 1, "max_frames": 180}},
                        {"type": "text", "text": querys[0]}
                        ]
                }
                for i, qa in enumerate(segments[current_query_time]):
                    if i == 0:
                        conversation.append(cov)
                    else:
                        conversation.append({"role": "user", "content": qa["Q"][0]})
                    output, selection = infer(conversation, model, processor)
                    if_visual = True
                    if hist_num > 0:
                        indices, if_visual = get_selection_output(selection)
                        relevance = [1 if i in indices else 0 for i in range(hist_num)]
                    else:
                        relevance = []
                        if_visual = True
                    video_answer.append({"qa_id":hist_num, "question":qa["Q"][0], "answer":qa["A"][0], "prediction":output, "predicted_coi":relevance, "predicted_visual":if_visual, "coi":qa["info"]["relevance"][0]})
                    hist_num += 1
                    conversation.append({
                        "role": "assistant",
                        "content": output,
                    })
            data.append(video_answer) 
            save_to_json(os.path.basename(video_path), data, save_dir)
            pbar.update(1)


def setup(rank, world_size):
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.manual_seed(42 + rank)
    random.seed(42 + rank)
    torch.cuda.set_device(rank)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VideoLLaMA3 inference with custom paths.")
    parser.add_argument("--model_path", type=str, help="Path to the base model directory.")
    parser.add_argument("--lora_adapter_1_path", type=str, help="Path to the first LoRA adapter.")
    parser.add_argument("--lora_adapter_2_path", type=str, help="Path to the second LoRA adapter.")
    parser.add_argument("--video_dir", type=str, help="Directory containing test video files.")
    parser.add_argument("--query_dir", type=str, help="Directory containing test query (QA) files.")
    parser.add_argument("--save_dir", type=str, help="Directory to save the result.")
    args = parser.parse_args()
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    setup(global_rank, world_size)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = PeftModel.from_pretrained(model, args.lora_adapter_1_path, adapter_name="full_module")
    model.load_adapter(args.lora_adapter_2_path, adapter_name="language_module")
    model.to(local_rank)
 
    dataset = VideoDataset(args.video_dir, args.query_dir)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)

    # 开始训练
    inference(model=model, dataloader=dataloader, processor=processor, save_dir=args.save_dir)
