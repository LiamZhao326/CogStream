import torch
import json
import os
from peft import LoraConfig
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import itertools
from tqdm import tqdm
from torch.amp import autocast
from transformers import AutoModel
from peft import get_peft_model, PeftModel
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM, AutoProcessor
import ast
import re
from transformers import BitsAndBytesConfig
import bitsandbytes.optim as bnb_optim
from accelerate import Accelerator
import warnings
import argparse
warnings.filterwarnings(
    action="ignore",
    message="None of the inputs have requires_grad=True. Gradients will be None",
    category=UserWarning,
    module="torch.utils.checkpoint"
)

warnings.filterwarnings(
    action="ignore",
    message=re.escape("`torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead."),
    category=FutureWarning,
    module="torch.utils.checkpoint"
)

# accelerate launch second_stage_training.py
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

def forward(conversation, model, processor, answer, cor, if_visual, accelerator):

    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    if "pixel_values" in inputs:
         inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)

    outputs = model.module.forward_train(**inputs, max_new_tokens=1024, temperature=0.5, answer=answer, cor=cor, if_visual=if_visual)

    return outputs.loss

def count_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"total param: {total_params:,}")
    print(f"trainable param: {trainable_params:,}")
    print(f"trainable ratio: {trainable_params / total_params * 100:.4f}%")
    return trainable_params, total_params

class VideoDataset(Dataset):
    def __init__(self, video_dir, query_dir, shuffle=True):
        self.video_and_querychain = []
        all_files = os.listdir(query_dir)
        
        if shuffle:
            random.shuffle(all_files)

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
        query_chain = random.choice(self.video_and_querychain[idx]["query_chains"])
        return {"video_path": video_path, "query_chain": query_chain}


def train(model, dataloader, optimizer, processor, scheduler, accelerator, lora_save_dir, num_epochs=1, accum_steps=4, resume_batch_idx=0, resume_step_idx=0, resume_epoch_idx=0):
    model.train()
    num_gradient_updates = (resume_step_idx // 8)
    save_every_n_steps = 30
    for epoch in range(resume_epoch_idx, num_epochs):
        total_loss = 0.0
        num_batches_processed_for_loss_avg = 0
        accum_step = 0
        accelerator.print(f"Starting epoch {epoch+1}/{num_epochs}")

        if epoch == resume_epoch_idx and resume_batch_idx > 0:
            if resume_batch_idx < len(dataloader):
                batches_to_skip = resume_batch_idx
                accelerator.print(f"Resuming epoch {epoch+1} by skipping first {batches_to_skip} batches.")
                effective_dataloader = itertools.islice(dataloader, batches_to_skip, None)
            else:
                accelerator.print(f"Warning: Resuming batch index {resume_batch_idx} is >= dataloader length {len(dataloader)}. Starting epoch {epoch+1} from batch 0.")
                effective_dataloader = dataloader
                batches_to_skip = 0
        else:
            effective_dataloader = dataloader
            batches_to_skip = 0

        with tqdm(desc=f"Epoch {epoch+1}/{num_epochs}, Update_steps={num_gradient_updates}", disable=not accelerator.is_local_main_process) as pbar:
            for video_id, batch in enumerate(effective_dataloader):
                video_id = video_id + batches_to_skip
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

                qa_num = 0
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
                        qa_num += 1
                        if i == 0:
                            conversation.append(cov)
                        else:
                            conversation.append({"role": "user", "content": qa["Q"][0]})
                        answer = qa["A"][0] + "<|im_end|>"
                        
                        try:
                            loss = forward(conversation, model, processor, answer, ast.literal_eval(qa["info"]["COI"][0]), qa["info"]["is_visual"][0], accelerator)
                        except Exception as e:
                            print(f"video {video_path} error {e}")
                        
                        loss = loss / accum_steps

                        accelerator.backward(loss)
                        total_loss += loss.item() * accum_steps
                        num_batches_processed_for_loss_avg += 1

                        accum_step += 1

                        if accum_step % accum_steps == 0:

                            optimizer.step()
                            if not accelerator.optimizer_step_was_skipped:
                                scheduler.step()
                            optimizer.zero_grad()
                            accum_step = 0
                            num_gradient_updates += 1
                            if (num_gradient_updates > 0) and (num_gradient_updates % save_every_n_steps == 0):
                                save_dir = lora_save_dir + f"/checkpoint_step_{num_gradient_updates}"
                                accelerator.wait_for_everyone()
                                try:
                                    accelerator.save_state(output_dir=save_dir)
                                    accelerator.print(f"Core state saved by accelerator to {save_dir}")
                                except Exception as e:
                                    accelerator.print(f"Error during accelerator.save_state: {e}")

                                total_loss = 0.0
                                num_batches_processed_for_loss_avg = 0
                                
                                if accelerator.is_main_process:
                                    os.makedirs(save_dir, exist_ok=True)
                                    unwrapped_model = accelerator.unwrap_model(model)
                                    unwrapped_model.save_pretrained(save_dir)
                                    accelerator.print(f"LoRA adapter weights saved to {save_dir}")
                                    processor.save_pretrained(save_dir)
                                    accelerator.print(f"Processor saved to {save_dir}")
                                
                                accelerator.wait_for_everyone()

                            current_avg_loss = total_loss / num_batches_processed_for_loss_avg if num_batches_processed_for_loss_avg > 0 else 0.0
                            current_lr = scheduler.get_last_lr()[0]
                            pbar.update(1)
                            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Video {video_id+1}/{len(dataloader)}, Update_steps={num_gradient_updates}, lr={current_lr}")
                            pbar.set_postfix(loss=f"{current_avg_loss:.4f}")
                        conversation.append({
                            "role": "assistant",
                            "content": qa["A"][0],
                        })
                try :
                    assert len(query_chain) == qa_num 
                except AssertionError as e:
                    print(f"Expected {len(query_chain)} queries, but got {qa_num}.{video_path}")
            avg_loss = total_loss / num_batches_processed_for_loss_avg if num_batches_processed_for_loss_avg > 0 else 0.0
            accelerator.print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")




if __name__ == "__main__":
    gradient_accumulation_steps = 4
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    target_modules = []
    parser = argparse.ArgumentParser(description="Run VideoLLaMA3 full module training.")
    parser.add_argument("--model_path", type=str, help="Path to the base model directory.")
    parser.add_argument("--video_dir", type=str, help="Directory containing train video files.")
    parser.add_argument("--query_dir", type=str, help="Directory containing train query (QA) files.")
    parser.add_argument("--num_epochs", type=str, help="Training epochs.")
    args = parser.parse_args()
    args.num_epochs = int(args.num_epochs)
    #############################################
    load_from_checkpoint = None
    RESUME_GRADIENT_UPDATES = 0
    RESUME_EPOCH = 0
    RESUME_BATCH_IDX = 0
    #############################################

    num_decoder_layers = 28
    for i in range(num_decoder_layers):
        target_modules.extend([
            f"model.layers.{i}.self_attn.q_proj",
            f"model.layers.{i}.self_attn.k_proj",
            f"model.layers.{i}.self_attn.v_proj",
            f"model.layers.{i}.self_attn.o_proj",
            f"model.layers.{i}.mlp.gate_proj",
            f"model.layers.{i}.mlp.up_proj",
            f"model.layers.{i}.mlp.down_proj"
        ])
    target_modules.extend([
            "model.mm_projector.readout.0",
            "model.mm_projector.readout.2"
        ])
    # LoRA 配置
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    with accelerator.main_process_first():
        quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
          args.model_path,
          trust_remote_code=True,
          torch_dtype=torch.bfloat16, # Compute dtype
          attn_implementation="flash_attention_2",
          quantization_config=quantization_config,
        )
        if load_from_checkpoint:
            processor = AutoProcessor.from_pretrained(load_from_checkpoint, trust_remote_code=True)
            accelerator.print(f"Processor loaded from checkpoint: {load_from_checkpoint}")
            accelerator.print(f"Loading PEFT adapters from {load_from_checkpoint}...")
            model = PeftModel.from_pretrained(model, load_from_checkpoint, is_trainable=True)
            accelerator.print("PEFT adapters loaded successfully.")
            model.enable_input_require_grads()
        else:
            processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
            accelerator.print(f"Processor loaded from base model path: {args.model_path}")


    if load_from_checkpoint == None:
        with accelerator.main_process_first():
            model.enable_input_require_grads()
            model = get_peft_model(model, lora_config)
    if accelerator.is_main_process:
        trainable_params, total_params = count_trainable_parameters(model)
        accelerator.print(f"Model loaded and PEFT applied. Trainable parameters: {trainable_params:,}")
 
    with accelerator.main_process_first():
        dataset = VideoDataset(args.video_dir, args.query_dir)
    dataloader = DataLoader(dataset, batch_size=1)

    optimizer = bnb_optim.AdamW8bit(model.parameters(), lr=1e-4)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    num_update_steps_per_epoch = ((len(dataloader) + gradient_accumulation_steps - 1) // gradient_accumulation_steps) * 20

    num_training_steps = args.num_epochs * num_update_steps_per_epoch
    accelerator.print(f"Total training steps (gradient updates): {num_training_steps}")
    scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=0)
    if RESUME_GRADIENT_UPDATES > 0:
        accelerator.print(f"Manually advancing scheduler state by {RESUME_GRADIENT_UPDATES} steps...")
        for _ in range(RESUME_GRADIENT_UPDATES):
            scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        accelerator.print(f"Scheduler advanced. Resumed learning rate approx: {current_lr:.2e}")
    # Prepare the scheduler
    lora_save_dir = "./stage2_lora_weights"
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    train(model, dataloader, optimizer, processor, scheduler, accelerator, lora_save_dir, num_epochs=args.num_epochs, accum_steps=gradient_accumulation_steps, resume_batch_idx=RESUME_BATCH_IDX, resume_epoch_idx=RESUME_EPOCH, resume_step_idx=RESUME_GRADIENT_UPDATES)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_save_dir = lora_save_dir
        if not os.path.exists(final_save_dir):
             os.makedirs(final_save_dir, exist_ok=True)

        # Save the final adapter weights
        unwrapped_model = accelerator.unwrap_model(model)
        if hasattr(unwrapped_model, "save_pretrained"):
             unwrapped_model.save_pretrained(final_save_dir)
             accelerator.print(f"Final LoRA adapter weights saved to {final_save_dir}")
        else:
             accelerator.save(unwrapped_model.state_dict(), os.path.join(final_save_dir, "pytorch_model.bin"))
             accelerator.print(f"Final full model state dict saved to {os.path.join(final_save_dir, 'pytorch_model.bin')}")

    accelerator.wait_for_everyone()

