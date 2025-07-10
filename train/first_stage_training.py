import torch
import json
import re
import os
import numpy as np
import random
from torch.distributed import init_process_group
from transformers import (
    AutoModelForCausalLM,
    Trainer, TrainingArguments,
    LogitsProcessor, LogitsProcessorList, AutoProcessor
)
from peft import LoraConfig, get_peft_model, PeftModel
import random
from transformers import TrainerCallback
import argparse

def count_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"total param: {total_params:,}")
    print(f"trainable param: {trainable_params:,}")
    print(f"trainable ratio: {trainable_params / total_params * 100:.4f}%")
    return trainable_params, total_params

# torchrun --nproc_per_node=8 first_stage_training.py
class EpochLossCallback(TrainerCallback):
    def __init__(self):
        self.epoch_losses = []
        self.stage_losses = []
    
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            epoch_loss = logs["loss"]
            self.epoch_losses.append(epoch_loss)
            self.stage_losses.append(epoch_loss)
            print(f"Epoch {state.epoch:.2f}: Average Loss = {epoch_loss:.4f}")

def load_json(folder_path):
    all_data = []
    for json_file in os.listdir(folder_path):
        if json_file.endswith(".json"):
            file_path = os.path.join(folder_path, json_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.append(data)
    
    return all_data
class StopTrainingCallback(TrainerCallback):
    def __init__(self, max_epochs=5):
        self.max_epochs = max_epochs

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch >= self.max_epochs:
            print(f"Epoch {state.epoch} reached {self.max_epochs}, stopping training.")
            control.should_terminate = True
        return control

# 1. 数据预处理函数
def format_example(example, include_demo=True):
    """
    Convert data sample to formatted prompt with instructions to output a list of indices
    of historical QA pairs that are helpful for answering the current question, prefixed
    with yes (requires additional visual info) or no (fully answerable by historical QA).
    """
    system_prompt = """<|im_start|>system
You are a QA-pair filtering assistant. Your task is to identify which of the historical QA pairs are helpful for answering the current question and determine if the historical QA pairs alone are sufficient to answer it.

A QA pair is considered helpful if it provides:
- Relevant background information, context, or details
- Additional facts or insights that can be used to answer the current question
- Matching roles, scenarios, or domain knowledge that could support the answer

Output a single bracketed sequence:
- Start with 'yes' if the historical QA pairs are insufficient to fully answer the question (additional visual information may be needed).
- Start with 'no' if the current question can be fully answered using only the historical QA pairs (no additional visual information needed).
- Follow with the indices (starting from 0) of the helpful QA pairs, e.g., [yes,0,5] or [no,0,5].
- If no QA pairs are helpful, output [yes] or [no] based on the question's dependency.
- Do not add extra text or explanation — only output the bracketed sequence.
<|im_end|>"""

    demo_section = ""
    if include_demo:
        demo_section = """\nExample:
Current Question: What causes earthquakes?
Historical QA Pairs:
0. Q: How to measure earthquakes? A: Using the Richter scale
1. Q: What is tectonic plate? A: Massive rock slabs beneath crust
2. Q: What is the weather like today? A: Sunny and warm
→ Output: [no,1]
------------------------------
Example:
Current Question: What does an earthquake look like?
Historical QA Pairs:
0. Q: How to measure earthquakes? A: Using the Richter scale
1. Q: What is tectonic plate? A: Massive rock slabs beneath crust
2. Q: What is the weather like today? A: Sunny and warm
→ Output: [yes]
------------------------------"""

    user_content = f"""{demo_section}
Current Question: {example['current_Q']}

Historical QA Pairs (ordered by time):"""

    for i, (q, a) in enumerate(zip(example['hist_Qs'], example['hist_As'])):
        user_content += f"\n{i}. Q: {q}\n   A: {a}"

    user_content += "\nGenerate a bracketed sequence (e.g., [yes,0,5] or [no,0,5]) indicating the dependency (yes or no) and the indices of helpful QA pairs. Only output the bracketed sequence."
    helpful_indices = [str(i) for i, label in enumerate(example['labels']) if label == 1]
    target = f",{','.join(helpful_indices)}]" if helpful_indices else "]"
    if example["if_visual"]:
        target = "[yes" + target
    else:
        target = "[no" + target
    full_text = (
        f"{system_prompt}"
        f"<|im_start|>user\n{user_content}<|im_end|>"
        f"<|im_start|>assistant\n{target}<|im_end|>"
    )

    
    return {
        "text": full_text,
        "prompt_part": f"{system_prompt}<|im_start|>user\n{user_content}<|im_end|><|im_start|>assistant\n",
        "target_part": f"{target}<|im_end|>"
    }

class CustomDataCollator:
    def __init__(self,
                 tokenizer,
                 max_length: int = 2048,
                 log_file="batch_log.txt",
                 eval_mode=False):
        self.eval_mode = eval_mode
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.log_file = log_file

    def __call__(self, batch_of_items):
        expansions = []
        log_entries = []

        for item_idx, item in enumerate(batch_of_items):
            if self.eval_mode:
                sample = self._make_sample(item["current_Q"], 
                                            item["hist_Qs"], item["hist_As"],
                                            item["labels_01"], item["if_visual"])
                expansions.append(sample)
            else:
                current_ID = item["ID"]
                if current_ID == 1:
                    sample = self._make_sample(item["current_Q"], 
                                            item["hist_Qs"], item["hist_As"],
                                            item["labels_01"], item["if_visual"])
                    expansions.append(sample)
                    log_entries.append(f"[Item {item_idx}] ID=1 => 1 sample.")
                
                elif current_ID == 2:
                    sample_normal = self._make_sample(item["current_Q"], 
                                                    item["hist_Qs"], item["hist_As"],
                                                    item["labels_01"], item["if_visual"])
                    expansions.append(sample_normal)
                    rev_Qs = list(reversed(item["hist_Qs"]))
                    rev_As = list(reversed(item["hist_As"]))
                    rev_labels = list(reversed(item["labels_01"]))
                    sample_rev = self._make_sample(item["current_Q"], rev_Qs, rev_As, rev_labels, item["if_visual"])
                    expansions.append(sample_rev)
                    log_entries.append(f"[Item {item_idx}] ID=2 => 2 samples.")
                
                else:
                    sample_normal = self._make_sample(item["current_Q"], 
                                                    item["hist_Qs"], item["hist_As"],
                                                    item["labels_01"], item["if_visual"])
                    expansions.append(sample_normal)

                    unique_shuffles = set()
                    while len(unique_shuffles) < 3:
                        indices = list(range(len(item["hist_Qs"])))
                        random.shuffle(indices)
                        shuf_Qs = [item["hist_Qs"][i] for i in indices]
                        shuf_As = [item["hist_As"][i] for i in indices]
                        if len(indices) != len(item["labels_01"]):
                            print(item["current_Q"])
                            print(indices)
                            print(item["labels_01"])
                        shuf_labels = [item["labels_01"][i] for i in indices]
                        unique_shuffles.add(tuple(zip(shuf_Qs, shuf_As, shuf_labels)))
                    
                    for shuf_tuple in unique_shuffles:
                        sq, sa, sl = zip(*shuf_tuple)
                        expansions.append(
                            self._make_sample(item["current_Q"], list(sq), list(sa), list(sl), item["if_visual"])
                        )
                    log_entries.append(f"[Item {item_idx}] ID={current_ID} => total {1 + 2} samples.")

        tokenized_batch = []
        target = []
        for exp_idx, s in enumerate(expansions):
            formatted_data = format_example(s)
            prompt_part = formatted_data["prompt_part"]
            target_part = formatted_data["target_part"]
            target.append(target_part)
            encoded_prompt = self.tokenizer(
                prompt_part,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,  
                padding=False,
                return_overflowing_tokens=False,
            )

            encoded_target = self.tokenizer(
                target_part,
                truncation=True,
                max_length=self.max_length - len(encoded_prompt["input_ids"]),
                add_special_tokens=False, 
                padding=False,
                return_overflowing_tokens=False,
            )

            input_ids = encoded_prompt["input_ids"] + encoded_target["input_ids"]

            labels = [-100] * len(encoded_prompt["input_ids"]) + encoded_target["input_ids"]
            tokenized_batch.append({
                "input_ids": input_ids,
                "labels": labels,
            })

        saved_labels = [item.pop("labels") for item in tokenized_batch]

        batch_dict = self.tokenizer.pad(
            tokenized_batch,
            padding="longest",
            return_tensors="pt"
        )
        max_seq_len = batch_dict["input_ids"].shape[1]

        padded_labels = []
        for lbl in saved_labels:
            needed = max_seq_len - len(lbl)
            if needed > 0:
                lbl = lbl + ([-100] * needed)
            elif needed < 0:
                lbl = lbl[:max_seq_len]
            padded_labels.append(lbl)
        batch_dict["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        batch_dict["label"] = target

        return batch_dict

    def _make_sample(self, current_Q, hist_Qs, hist_As, labels_01, if_visual):
        return {
            "current_Q": current_Q,
            "hist_Qs": hist_Qs,
            "hist_As": hist_As,
            "labels": labels_01,
            "if_visual": if_visual,
        }

class ConstrainedTrainer(Trainer):
    def __init__(self, *args, logits_processor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.logits_processor = logits_processor or LogitsProcessorList()

    def training_step(self, model, inputs, *args):
        try:
            loss = super().training_step(model, inputs)
            return loss
        except RuntimeError as e:
            torch.cuda.empty_cache()
            raise e
            
    def compute_loss(self, model, inputs, return_outputs=False, print_test=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        if self.logits_processor is not None:
            logits = outputs.logits
            processed_logits = logits.clone()
            for processor in self.logits_processor:
                processed_logits = processor(inputs["input_ids"], processed_logits)
        outputs.logits = processed_logits

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()
        if print_test:
                print(f"Shift Labels Length: {len(shift_labels[0])}")
                print(f"Shift Labels: {shift_labels[0].tolist()}")
                pred_ids = shift_logits[0].argmax(-1).tolist()
                print(f"Predicted IDs: {pred_ids}")
                print(f"Decoded Pred: {tokenizer.decode(pred_ids)}")
                valid_count = (shift_labels[0] != -100).sum().item()
                print(f"Valid tokens in shift_labels: {valid_count}")

        def decode_valid_predictions(shift_logits, shift_labels, tokenizer, num_samples=2):
            sample_idx = 0
            sample_labels = shift_labels[sample_idx]
            valid_positions = (sample_labels != -100).nonzero().squeeze(-1)
            
            if valid_positions.nelement() == 0:
                print(f"样本 {sample_idx}: 无有效标签")

                
            print(f"\n=== 样本 {sample_idx} 诊断 ===")
            for pos in valid_positions:
                true_token_id = sample_labels[pos].item()
                pred_token_id = shift_logits[sample_idx, pos].argmax(-1).item()

                input_token_id = inputs["input_ids"][sample_idx, pos+1].item()
                
                print(
                    f"位置 {pos} | "
                    f"输入 token: {tokenizer.decode(input_token_id)} | "
                    f"预测 token: {tokenizer.decode(pred_token_id)} ({pred_token_id}) | "
                    f"标签 token: {tokenizer.decode(true_token_id)} ({true_token_id}) | "
                    f"正确/错误: {pred_token_id == true_token_id}"
                )

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        if print_test:
            decode_valid_predictions(shift_logits, shift_labels, tokenizer)
            print(f"Loss: {loss.item()}")
        outputs.loss = loss
        return (outputs.loss, outputs) if return_outputs else outputs.loss

class FlattenedQADataset(torch.utils.data.Dataset):
    def __init__(self, all_data, shuffle=True):
 
        self.samples = []
        if shuffle:
            random.shuffle(all_data)
        for original_data in all_data:
            if shuffle:
                random.shuffle(original_data)
            for group in original_data:
                for i, qa_pair in enumerate(group):
                    current_Q = qa_pair["Q"]
                    current_ID = qa_pair["info"]["ID"]
                    if_visual = qa_pair["info"]["is_visual"]
                    relevance = qa_pair["info"]["relevance"]


                    if isinstance(relevance, str):
                        relevance = json.loads(relevance)
                    labels_01 = [int(r) for r in relevance]  # 0/1
                    if len(labels_01) != qa_pair["info"]["ID"]:
                        print(qa_pair["Q"])

                    hist_Qs = [x["Q"] for x in group[:i]]
                    hist_As = [x["A"] for x in group[:i]]
                    
                    if current_ID == 0:
                        continue
                    item = {
                        "current_Q": current_Q,
                        "hist_Qs": hist_Qs,
                        "hist_As": hist_As,
                        "labels_01": labels_01,   # [0,1,0,...]
                        "ID": current_ID,
                        "if_visual": if_visual,
                    }
                    self.samples.append(item)
        
        print(f"[FlattenedQADataset] total {len(self.samples)} items.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

class StructuredLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.allowed_token_ids = self._get_allowed_token_ids(tokenizer)

    def _get_allowed_token_ids(self, tokenizer):
        special_tokens = [str(i) for i in range(10)] + ['[', ']', ',', '<|im_end|>', 'no', 'yes']
        allowed_ids = set()
        for t in special_tokens:
            token_ids = tokenizer.encode(t, add_special_tokens=False)
            for idx in token_ids:
                if idx >= 0:
                    allowed_ids.add(idx)
        return sorted(list(allowed_ids))

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float('-inf'))
        mask[..., self.allowed_token_ids] = 0
        return scores + mask

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    # preds: [batch_size, seq_len, vocab_size]
    # labels: [batch_size, seq_len]
    preds = np.argmax(preds, axis=-1)

    exact_match = 0
    token_acc = 0
    total_tokens = 0

    for pred_seq, label_seq in zip(preds, labels):
        pred_text = tokenizer.decode(pred_seq, skip_special_tokens=True)
        true_text = tokenizer.decode(label_seq, skip_special_tokens=True)
        pred_digits = re.findall(r'\d', pred_text)
        true_digits = re.findall(r'\d', true_text)
        if pred_digits == true_digits:
            exact_match += 1
        min_len = min(len(pred_digits), len(true_digits))
        token_acc += sum(p == t for p, t in zip(pred_digits[:min_len], true_digits[:min_len]))
        total_tokens += min_len
    return {
        "exact_match": exact_match / len(preds),
        "token_accuracy": token_acc / total_tokens if total_tokens > 0 else 0
    }

def setup(rank, world_size):
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.manual_seed(42 + rank)

if __name__ == "__main__":
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    setup(global_rank, world_size)
    parser = argparse.ArgumentParser(description="Run VideoLLaMA3 QA selection module training.")
    parser.add_argument("--model_path", type=str, default="model", help="Path to the base model directory.")
    parser.add_argument("--QA_path", type=str, help="Path to the QA directory.")
    args = parser.parse_args()
    model_path = args.model_path
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer


    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    target_modules = []
    for i in range(28):
        target_modules.extend([
            f"model.layers.{i}.self_attn.q_proj",
            f"model.layers.{i}.self_attn.k_proj",
            f"model.layers.{i}.self_attn.v_proj",
            f"model.layers.{i}.self_attn.o_proj",
            f"model.layers.{i}.mlp.gate_proj",
            f"model.layers.{i}.mlp.up_proj",
            f"model.layers.{i}.mlp.down_proj"
        ])
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    base_model.enable_input_require_grads()
    model = get_peft_model(base_model, lora_config)
    model = model.to(local_rank)
    model.config.use_cache = False
    for name, param in model.named_parameters():
        if "lora" not in name:
            assert not param.requires_grad, f"Non-LoRA参数 {name} 未冻结！"
        else:
            assert param.requires_grad, f"LoRA参数 {name} 未启用梯度！"
    trainable_params, total_params = count_trainable_parameters(model)
    print("total_params:", total_params, "trainable_params:", trainable_params)
    train_dataset = load_json(args.QA_path)
    train_dataset = FlattenedQADataset(train_dataset)
    logits_processor = LogitsProcessorList([StructuredLogitsProcessor(tokenizer)])
    training_args = TrainingArguments(
        output_dir="./stage1_lora_weights",
        per_device_train_batch_size=1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        max_grad_norm=0.5,
        warmup_ratio=0.08,
        gradient_accumulation_steps=4,
        num_train_epochs=8,
        logging_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,  
        dataloader_num_workers=2,     
        ddp_find_unused_parameters=False,
        resume_from_checkpoint=False,
        report_to=None
    )
    trainer = ConstrainedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        logits_processor=logits_processor,
        data_collator=CustomDataCollator(tokenizer, eval_mode=False),
        callbacks=[EpochLossCallback()]
    )
    trainer.train()
    
