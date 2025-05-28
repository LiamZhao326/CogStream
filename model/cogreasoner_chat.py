# Adopted from https://github.com/haotian-liu/LLaVA.
# Below is the original copyright:
# Copyright 2023 Haotian Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CogReasoner model."""
import importlib.util
import os.path as osp
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import shutil
import os
from transformers import AutoModel, Qwen2ForCausalLM, Qwen2Model
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from .kmeans_with_time import kmeans_with_time_min_max
from .qaselect_module_predict import select_qas
from transformers.feature_extraction_utils import BatchFeature

try:
    from .configuration_videollama3 import Videollama3Qwen2Config
except ModuleNotFoundError:
    spec = importlib.util.spec_from_file_location(
        "configuration_videollama3",
        osp.join(osp.dirname(__file__), "configuration_videollama3.py"),
    )
    configuration_videollama3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(configuration_videollama3)
    Videollama3Qwen2Config = getattr(
        configuration_videollama3,
        "Videollama3Qwen2Config",
    )

def select_additional_frames(cls_feature, long_memory, cluster_assignments, additional_frame_num):  # shape: [long_memory_length, P, D], [T]
    cls_feature = cls_feature.to(dtype=torch.float32)
    long_memory_flattened = long_memory.view(long_memory.shape[0], -1)  # [long_memory_length, P * D]
    cls_flattened = cls_feature.view(cls_feature.shape[0], -1)
    selected_frame_indices = []
    for i in range(long_memory_flattened.shape[0]):
        mask = (cluster_assignments == i)
        features = cls_flattened[mask]
        if features.shape[0] <= additional_frame_num:
            selected_frame_indices.append(torch.nonzero(mask, as_tuple=True)[0])
        else:
            distances = torch.cdist(features, long_memory_flattened[i].unsqueeze(0))
            _, topk_indices = torch.topk(distances.squeeze(1), k=additional_frame_num, largest=False)
            selected_frame_indices.append(torch.nonzero(mask, as_tuple=True)[0][topk_indices])
    return selected_frame_indices     # type: list.   len: long_memory_length.

def copy_frame_file(source_dir, target_base_dir, indices, idx):
    if not os.path.exists(source_dir):
        print(f"错误：源文件夹 {source_dir} 不存在")
        return False
    target_dir = os.path.join(target_base_dir, str(idx))
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    for i in indices:
        source_filename = f"frame_{i:04d}_t{i:.3f}.png"
        source_path = os.path.join(source_dir, source_filename)

        target_path = os.path.join(target_dir, source_filename)

        if not os.path.exists(source_path):
            print(f"警告：源文件 {source_path} 不存在，跳过")
            continue

        try:
            shutil.copy2(source_path, target_path)
        except Exception as e:
            print(f"复制失败：{source_path} -> {target_path}，错误：{e}")
            return False

    return True

def create_visual_summary_prompt(P, timestamps, image_token="<image>"):

    instruction = "Concisely list the key points of the event shown in the timestamped images, adhering strictly and honestly to the visual content. For each key point, identify relevant objects or actions, note any visible text, and specify the approximate timestamp(s). Provide an overview focusing on these key timestamped points." 
    T = len(timestamps)
    prompt_parts = [
        f"<|im_start|>system\n"
        f"You are a helpful assistant specializing in summarizing events from timestamped visual data.<|im_end|>\n"
        f"<|im_start|>user\n" 
    ]
    image_placeholder_sequence = image_token * (P//T)

    frame_strings = []
    for t in range(T):
        ts_val = timestamps[t].item() if isinstance(timestamps[t], torch.Tensor) else float(timestamps[t])
        formatted_ts = f"{ts_val:.1f}s"
        current_frame_str = f"Time {formatted_ts}:{image_placeholder_sequence}"
        if t < T - 1:
            current_frame_str += ","
            
        frame_strings.append(current_frame_str)
    all_frames_string = "".join(frame_strings)
    prompt_parts.append(all_frames_string + "\n") 
    prompt_parts.append(instruction)
    prompt_parts.append("<|im_end|>\n")
    prompt_parts.append("<|im_start|>assistant") 
    final_prompt = "".join(prompt_parts)
    return final_prompt

def process_input_ids(text, if_visual, hist_qs, hist_as, current_question, tokenizer):
    if not if_visual:
        text = re.sub(r'Time \d+\.\d+s:(?:<image>)*,', '', text)
        text = re.sub(r'Time \d+\.\d+s:(?:<image>)*\n', '', text)

    segments = text.split('<|im_start|>')[1:]
    filtered_segments = []
    last_user_question = current_question
    
    for segment in segments:
        role_content = segment.split('\n', 1)
        if len(role_content) != 2:
            continue
        role, content = role_content
        role = role.strip()
        content = content.split('<|im_end|>')[0].strip()

        if role == 'system':
            filtered_segments.append(f'<|im_start|>{role}\n{content}<|im_end|>\n')
            continue

        if role == 'user':
            visual_content = ''
            question = content
            if if_visual:
                match = re.match(r'((?:(?:Time \d+\.\d+s:(?:<image>)*),?)*)\s*(.*)', content)
                if match:
                    visual_content = match.group(1).rstrip(',').strip()
                    question = match.group(2).strip()

            if question == last_user_question:
                filtered_segments.append(f'<|im_start|>{role}\n{content}<|im_end|>\n')
            elif question in hist_qs:
                filtered_segments.append(f'<|im_start|>{role}\n{content}<|im_end|>\n')
            elif if_visual and visual_content:
                filtered_segments.append(f'<|im_start|>{role}\n{visual_content}')
            continue

        if role == 'assistant':
            if content in hist_as:
                filtered_segments.append(f'<|im_start|>{role}\n{content}<|im_end|>\n')

    filtered_segments.append(f'<|im_start|>assistant\n')

    cleaned_segments = []
    for i, segment in enumerate(filtered_segments):
        if segment.startswith('<|im_start|>user\n'):
            if i == 0 or not filtered_segments[i-1].rstrip().endswith('<|im_end|>'):
                cleaned_segment = segment[len('<|im_start|>user\n'):]
                if cleaned_segment.strip():
                    cleaned_segments.append(cleaned_segment)
                continue
        cleaned_segments.append(segment)

    result = ''.join(cleaned_segments)
    
    return result

def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


def build_vision_projector(config, delay_load=False, **kwargs):
    # videollama3 projector only support image-wise operation now, i.e., prohibit the temporal aggregation
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    if projector_type == "linear":
        # NOTE: for both linear and mlp2x_gelu projector type, mean pooling is adopted to aggreate video features
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    elif projector_type.startswith("mlp"):
        return MlpGeluProjector(config, projector_type)
    else:
        raise ValueError(f'Unknown projector type: {projector_type}')


class MlpGeluProjector(nn.Module):

    def __init__(self, config, projector_type):
        super().__init__()

        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
        mlp_depth = int(mlp_gelu_match.group(1))

        self.readout = build_mlp(mlp_depth, config.vision_encoder_config.hidden_size, config.hidden_size)

    def forward(self, x):
        x = self.readout(x)
        return x


class Videollama3MetaModel:

    def __init__(self, config):
        super(Videollama3MetaModel, self).__init__(config)
        if config.vision_encoder is not None:
            self.vision_encoder = AutoModel.from_pretrained(
                config.vision_encoder,
                attn_implementation=self.config._attn_implementation,
                torch_dtype=self.dtype,
            )
            self.config.vision_encoder_config = self.vision_encoder.config
            self.config.vision_encoder = None
        elif config.vision_encoder_config is not None:
            self.vision_encoder = AutoModel.from_config(
                self.config.vision_encoder_config,
                attn_implementation=self.config._attn_implementation,
                torch_dtype=self.dtype,
            )
        else:
            raise ValueError("Vision encoder is not provided in config")
        self.mm_projector = build_vision_projector(config)

    def get_vision_encoder(self):
        return self.vision_encoder

    def get_mm_projector(self):
        return self.mm_projector


class Videollama3Qwen2Model(Videollama3MetaModel, Qwen2Model):

    config_class = Videollama3Qwen2Config

    def __init__(self, config: Videollama3Qwen2Config): # type: ignore
        super(Videollama3Qwen2Model, self).__init__(config)


class Videollama3MetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_encoder(self):
        return self.get_model().get_vision_encoder()

    def get_mm_projector(self):
        return self.get_model().get_mm_projector()
    

    def encode_images(
        self,
        pixel_values: torch.FloatTensor,
        grid_sizes: torch.LongTensor,
        merge_sizes: torch.LongTensor,
    ) -> torch.FloatTensor:
        mm_features = self.get_model().get_vision_encoder()(
            pixel_values=pixel_values,
            grid_sizes=grid_sizes,
            merge_sizes=merge_sizes,
        )
        mm_features = self.get_model().mm_projector(mm_features)
        return mm_features

    def select_events_based_on_summary(self, mm_features, total_image_num, timestamps):
        features = mm_features.view(total_image_num, mm_features.shape[0]//total_image_num, mm_features.shape[1])
        memory_length = math.ceil(features.shape[0] / 15)
        if memory_length <= 9:
            return []
        features_kmeans, _, cluster_assignments = kmeans_with_time_min_max(features, timestamps, memory_length)
        selected_indices = select_additional_frames(features, features_kmeans, cluster_assignments, 2)
        selected_indices_set = torch.cat(selected_indices, dim=0).view(-1).tolist()
        events = [[] for _ in range(memory_length)]
        all_timestamps = [[] for _ in range(memory_length)]
        list_of_input_ids = []
        list_of_prepared_features = []
        list_of_attention_mask = []
        for i in range(len(cluster_assignments)):
            events[cluster_assignments[i]].append(features[i])
            all_timestamps[cluster_assignments[i]].append(timestamps[i])
        for i in range(memory_length):
            events[i] = torch.cat(events[i], dim=0)
            all_timestamps[i] = torch.stack(all_timestamps[i], dim=0)
            prompt = create_visual_summary_prompt(events[i].shape[0], all_timestamps[i])
            inputs = self.tokenizer(prompt, return_tensors="pt")
            list_of_input_ids.append(inputs["input_ids"].to(features.device))
            list_of_attention_mask.append(inputs["attention_mask"].to(features.device))
            list_of_prepared_features.append(events[i].to(features.device))
        outputs = []
        for idx, input_ids in enumerate(list_of_input_ids):
            inputs_embeds = self.get_model().embed_tokens(input_ids).clone()
            image_selected = (input_ids == self.config.image_token_index)
            inputs_embeds[image_selected] = inputs_embeds[image_selected] * 0.0 + events[idx]
            B=1
            C = inputs_embeds.shape[-1]
            inputs_embeds = inputs_embeds.reshape(B, -1, C)
            if list_of_attention_mask[idx] is not None:
                attention_mask = list_of_attention_mask[idx].view(B, -1)
            outputs.append(self.get_model()(
                position_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            ))
        events_pooled_repr = [torch.mean(outputs[i].last_hidden_state, dim=1) for i in range(len(outputs))]
        events_pooled_repr = torch.cat(events_pooled_repr, dim=0)
        inputs = self.tokenizer(self.current_question, padding=True, truncation=True, return_tensors="pt", max_length=128)
        model_device = next(self.get_model().parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        outputs = self.get_model()(**inputs)
        question_embedding = torch.mean(outputs.last_hidden_state, dim=1)
        
        cosine_sim = F.cosine_similarity(question_embedding[0], events_pooled_repr, dim=1)
        nums = cosine_sim.shape[0]
        assert nums == memory_length
        
        threshold = 0.45
        top_indices = torch.where(cosine_sim < threshold)[0]
        unimportant_indices_list = [i for i in range(total_image_num) if cluster_assignments[i].item() in set(top_indices.tolist())]
        filtered_indices = [idx for idx in unimportant_indices_list if idx not in selected_indices_set]
        return filtered_indices


    def _get_valid_visual_tokens(
        self,
        mm_features: torch.FloatTensor,
        batched_num_patches: torch.LongTensor,
        modals: List[str],
    ):
        valid_masks = []
        for num_patches, modal in zip(batched_num_patches, modals):
            valid_mask = torch.full((num_patches, ), modal != "text", dtype=torch.bool, device=mm_features.device)
            valid_masks.append(valid_mask)
        mm_features = mm_features[torch.cat(valid_masks)]
        return mm_features

    def _maybe_truncate_visual_tokens(
        self,
        mm_features: torch.FloatTensor,
        compression_mask: torch.BoolTensor,
        batched_num_patches: torch.LongTensor,
        modals: List[str],
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        if position_ids is None or mm_features.shape[0] == input_ids.eq(self.config.image_token_index).sum():
            return mm_features, compression_mask

        truncation_mask = []
        for num_patches, modal in zip(batched_num_patches, modals):
            if modal == "text":
                truncation_mask.append(torch.ones((0,), dtype=torch.bool, device=input_ids.device))
            else:
                truncation_mask.append(torch.ones((num_patches,), dtype=torch.bool, device=input_ids.device))

        seq_end_indices = torch.nonzero(position_ids == 0)[:, 0]
        seq_end_indices = seq_end_indices[seq_end_indices > 0].tolist()+ [len(input_ids)]
        seq_start_indices = [0] + seq_end_indices[:-1]
        num_visual_tokens = [
            input_ids[start:end].eq(self.config.image_token_index).sum()
            for start, end in zip(seq_start_indices, seq_end_indices)
        ]

        for n, mask in zip(num_visual_tokens, truncation_mask):
            if len(mask) > 0:
                mask[n:] = False
        truncation_mask = torch.cat(truncation_mask)

        return mm_features[truncation_mask], compression_mask[truncation_mask]

    def _get_compression_mask(
        self,
        pixel_values: torch.FloatTensor,
        batched_num_patches: torch.LongTensor,
        grid_sizes: torch.LongTensor,
        merge_sizes: torch.LongTensor,
        modals: List[str],
        threshold: float = 0.1,
        min_tokens: int = 1,
        minor_frame_indices: Optional[List[int]] = None,
    ) -> torch.BoolTensor:
        batched_images = pixel_values.split(grid_sizes.prod(dim=1).tolist(), dim=0)
        compression_masks = []
        global_frame_count = 0
        for images, num_patches, grid_size, merge_size, modal in zip(
            batched_images, batched_num_patches, grid_sizes, merge_sizes, modals
        ):
            t, h, w = grid_size
            if modal == "image" or (modal == "video" and t == 1):
                compression_masks.append(torch.ones((num_patches,), dtype=torch.bool, device=images.device))
                global_frame_count += t

            elif modal == "video":
                # NOTE: video token compressor
                images = images.view(t, (h // merge_size) * (w // merge_size), -1)

                pixel_diff = images[1:] - images[:-1]
                pixel_diff = torch.abs(pixel_diff).mean(dim=-1) * 255
                pixel_diff = torch.cat([torch.full_like(pixel_diff[0:1], threshold + 1), pixel_diff], dim=0)
                mask = pixel_diff > threshold
                padding_ids = torch.nonzero(mask.sum(dim=1) < min_tokens)[:, 0]
                mask[padding_ids, :min_tokens] = 1

                for frame_t in range(t):
                    current_global_index = global_frame_count + frame_t

                    if current_global_index in minor_frame_indices:

                        mask[frame_t, 0] = True
                        mask[frame_t, 1:] = False

                compression_masks.append(mask.flatten())
                global_frame_count += t

            else:
                # in case of psuedo image
                compression_masks.append(torch.ones((0,), dtype=torch.bool, device=images.device))
                global_frame_count += t

        return torch.cat(compression_masks)

    def compress_unimportant_events(self,
                                    mm_features: torch.FloatTensor,
                                    patch_num: int,
                                    minor_frame_indices: List[int]
                                   ) -> torch.FloatTensor:
        total_patches, embedding_dim = mm_features.shape
        if total_patches % patch_num != 0:
            raise ValueError(f"总 patch 数 ({total_patches}) 不能被每帧 patch 数 ({patch_num}) 整除")
        num_frames = total_patches // patch_num
        features_reshaped = mm_features.view(num_frames, patch_num, embedding_dim).clone()
        for frame_idx in minor_frame_indices:
            pooled_feature = features_reshaped[frame_idx, :, :].mean(dim=0)
            features_reshaped[frame_idx, 0, :] = pooled_feature
        return features_reshaped.view(-1, embedding_dim)

    def _compress_visual_tokens(
        self,
        compression_mask: torch.BoolTensor,
        mm_features: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        mm_features = mm_features[compression_mask]
        image_selected = (input_ids == self.config.image_token_index)

        text_masks = torch.logical_not(image_selected)
        
        text_masks[image_selected] = compression_mask
        input_ids = input_ids[text_masks]

        if attention_mask is not None:
            attention_mask = attention_mask[text_masks]
        if labels is not None:
            labels = labels[text_masks]
        if position_ids is not None:
            position_ids = position_ids[text_masks]
            pos_start = [0] + torch.nonzero(position_ids == 0)[:, 0].tolist()
            pos_end = pos_start[1:] + [len(input_ids)]
            position_ids = torch.cat([torch.arange(end - start, device=input_ids.device) for start, end in zip(pos_start, pos_end)])

        return mm_features, input_ids, attention_mask, position_ids, labels
    
    def prepare_inputs(self, selection_module_output, original_text=None, answer=None):
        if_visual = True
        selected_indices = []

        cleaned_str = selection_module_output.strip('[]')
        parts = cleaned_str.split(',')

        if parts and parts[0]:
            first_char = parts[0].strip()
            if first_char == 'no':
                if_visual = False
                parts = parts[1:]
            elif first_char == 'yes':
                parts = parts[1:]
        
        for part in parts:
            stripped_part = part.strip()
            if stripped_part:
                try:
                    selected_indices.append(int(stripped_part))
                except ValueError:
                    continue
                
        hist_qs = [self.hist_qs[i] for i in selected_indices if i < len(self.hist_qs)]
        hist_as = [self.hist_as[i] for i in selected_indices if i < len(self.hist_qs)]
        new_prompt = process_input_ids(original_text, if_visual, hist_qs, hist_as, self.current_question, self.tokenizer)
        if answer is not None:
            original_input_length = len(self.tokenizer(new_prompt, padding=False, truncation=True, return_tensors='pt')["input_ids"][0])
            new_prompt = new_prompt + answer
            new_inputs = self.tokenizer(new_prompt, padding=False, truncation=True, return_tensors='pt')
            return new_inputs, original_input_length
        else:
            new_inputs = self.tokenizer(new_prompt, padding=False, padding_side='right', return_tensors='pt')
            return new_inputs, if_visual

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        total_image_num=0,
        if_visual=True,
    ):
        
        vision_encoder = self.get_vision_encoder()
        # NOTE: text-only situation
        if vision_encoder is None or pixel_values is None or input_ids.shape[1] == 1:
            return input_ids, attention_mask, position_ids, past_key_values, None, labels

        # 1. flatten text inputs
        B, N = input_ids.shape
        input_ids = input_ids.view(B * N)
        if attention_mask is not None:
            attention_mask = attention_mask.view(B * N)
        if position_ids is not None:
            position_ids = position_ids.view(B * N)
        if labels is not None:
            labels = labels.view(B * N)

        # 2. embed visual tokens
        if if_visual:
            batched_num_patches = grid_sizes.prod(dim=1).div(merge_sizes ** 2).long()
            mm_features = self.encode_images(pixel_values, grid_sizes, merge_sizes)
            mm_features = self._get_valid_visual_tokens(mm_features, batched_num_patches, modals)
            
            assert mm_features.shape[0] % total_image_num == 0, f"mm_features.shape[0] % patch_num != 0, {mm_features.shape[0]} % {total_image_num} != 0"
            frame_indices = self.select_events_based_on_summary(mm_features, total_image_num, self.all_timestamps)
            mm_features = self.compress_unimportant_events(mm_features, mm_features.shape[0] // total_image_num, frame_indices)
            compression_mask = self._get_compression_mask(
                pixel_values, batched_num_patches, grid_sizes, merge_sizes, modals, minor_frame_indices=frame_indices
            )
            mm_features, compression_mask = self._maybe_truncate_visual_tokens(
                mm_features, compression_mask, batched_num_patches, modals, input_ids, position_ids
            )

            # 3. compress visual tokens
            if self.config.use_token_compression:
                assert B == 1, "Token compression is only supported for batch_size=1"
                mm_features, input_ids, attention_mask, position_ids, labels = self._compress_visual_tokens(
                    compression_mask, mm_features, input_ids, attention_mask, position_ids, labels
                )

        # 4. embed text tokens
        inputs_embeds = self.get_model().embed_tokens(input_ids).clone()
        if if_visual:
            # 5. replace multimodal tokens with features
            image_selected = (input_ids == self.config.image_token_index)

            inputs_embeds[image_selected] = inputs_embeds[image_selected] * 0.0 + mm_features.to(inputs_embeds.dtype)   

        # 6. reshape back to batched format
        C = inputs_embeds.shape[-1]
        inputs_embeds = inputs_embeds.reshape(B, -1, C)
        if attention_mask is not None:
            attention_mask = attention_mask.view(B, -1)
        if labels is not None:
            labels = labels.view(B, -1)
        if position_ids is not None:
            position_ids = position_ids.view(B, -1)

        return None, attention_mask, position_ids, past_key_values, inputs_embeds, labels


class Videollama3Qwen2ForCausalLM(Qwen2ForCausalLM, Videollama3MetaForCausalLM):

    config_class = Videollama3Qwen2Config

    def __init__(self, config, **kwargs):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = Videollama3Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    # NOTE: arguments are copied from transformers==4.46.3
    def forward_train(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        # multimodal inputs
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        tokenizer = None,
        hist_qs = None,
        hist_as = None,
        current_question = None,
        all_timestamps = None,
        total_image_num = None,
        answer = None,
        cor = None,
        if_visual = True,
        original_text = None,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        self.tokenizer = tokenizer
        self.hist_qs = hist_qs
        self.hist_as = hist_as
        self.current_question = current_question
        self.all_timestamps = torch.tensor(all_timestamps, device=input_ids.device)
        prefix = "yes" if if_visual else "no"
        cor_str_list = [str(num) for num in cor]
        combined_list = [prefix] + cor_str_list
        inner_content = ",".join(combined_list)
        result_string = f"[{inner_content}]"
        new_inputs, query_token_count = self.prepare_inputs(result_string, answer=answer, original_text=original_text)
        new_input_ids = new_inputs["input_ids"].to(input_ids.device)
        new_attention_mask = new_inputs["attention_mask"].to(input_ids.device)
        answer_encoding = self.tokenizer(answer, padding=False, truncation=True, return_tensors="pt")
        answer_ids_tensor = answer_encoding["input_ids"]
        if answer_ids_tensor.numel() > 0:
            answer_ids_list = answer_ids_tensor[0].tolist()
        else:
            answer_ids_list = []
        assert new_input_ids.shape[1] == query_token_count + len(answer_ids_list)
        labels_list = [-100] * query_token_count + answer_ids_list
        labels = torch.tensor(labels_list, device=input_ids.device).long().unsqueeze(0)
        if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                pixel_values=pixel_values,
                grid_sizes=grid_sizes,
                merge_sizes=merge_sizes,
                modals=modals,
                total_image_num=total_image_num,
                if_visual=if_visual,
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **loss_kwargs)

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        # multimodal inputs
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        **loss_kwargs,
        ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            labels,
            ) = self.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            pixel_values=pixel_values,
            grid_sizes=grid_sizes,
            merge_sizes=merge_sizes,
            modals=modals,
            )

        return super().forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        num_logits_to_keep=num_logits_to_keep,
        **loss_kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        # multimodal inputs
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        new_input_ids = None,
        new_attention_mask = None,
        selection_module_output = "",
        if_visual = True,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
    
        input_ids = kwargs.pop("input_ids", None)
        past_key_values = kwargs.pop("past_key_values", None)
        attention_mask = kwargs.pop("attention_mask", None)
        position_ids = kwargs.pop("position_ids", None)
        total_image_num = kwargs.pop("total_image_num", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if pixel_values is not None:
            (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=None,
                pixel_values=pixel_values,
                grid_sizes=grid_sizes,
                merge_sizes=merge_sizes,
                modals=modals,
                total_image_num=total_image_num,
                if_visual=if_visual
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(new_input_ids)
            attention_mask = new_attention_mask

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        ), selection_module_output
    
    @torch.no_grad()
    def qa_selection(self, current_question=None, hist_qs=None, hist_as=None, tokenizer=None, original_text=None, input_ids=None, attention_mask=None, mode="FCC", select_gt=None, if_visual=None, **kwargs):
        
        self.tokenizer = tokenizer
        self.hist_qs = hist_qs
        self.hist_as = hist_as
        self.current_question = current_question
        self.all_timestamps = torch.tensor(kwargs.pop("all_timestamps", None), device=input_ids.device)
        if mode == "FCC":
            if len(hist_qs) == 0:
                new_input_ids = input_ids
                new_attention_mask = attention_mask
                selection_module_output  = ""
                if_visual = True
            else:
                selection_module_output = select_qas(current_question, hist_qs, hist_as, self, tokenizer)
                new_inputs, if_visual = self.prepare_inputs(selection_module_output, original_text=original_text)
                new_input_ids = new_inputs["input_ids"].to(input_ids.device)
                new_attention_mask = new_inputs["attention_mask"].to(input_ids.device)
        elif mode == "AC":
            new_input_ids = input_ids
            new_attention_mask = attention_mask
            selection_module_output  = ""
            if_visual = True
        elif mode == "NC":
            if len(hist_qs) == 0:
                new_input_ids = input_ids
                new_attention_mask = attention_mask
                selection_module_output  = ""
                if_visual = True
            else:
                selection_module_output = "[yes]"
                new_inputs, if_visual = self.prepare_inputs(selection_module_output, original_text=original_text)
                new_input_ids = new_inputs["input_ids"].to(input_ids.device)
                new_attention_mask = new_inputs["attention_mask"].to(input_ids.device)
        elif mode == "gt":
            assert select_gt != None, "in gt mode, you should provide selection gt"

            prefix = "yes" if if_visual else "no"
            cor_str_list = [str(num) for num in select_gt]
            combined_list = [prefix] + cor_str_list
            inner_content = ",".join(combined_list)
            result_string = f"[{inner_content}]"

            if len(hist_qs) == 0:
                new_input_ids = input_ids
                new_attention_mask = attention_mask
                selection_module_output  = ""
                if_visual = True
            else:
                selection_module_output = result_string
                new_inputs, if_visual = self.prepare_inputs(selection_module_output, original_text=original_text)
                new_input_ids = new_inputs["input_ids"].to(input_ids.device)
                new_attention_mask = new_inputs["attention_mask"].to(input_ids.device)


        return BatchFeature(data={"new_input_ids": new_input_ids, "new_attention_mask": new_attention_mask, "selection_module_output": selection_module_output, "input_ids": input_ids, "attention_mask": attention_mask, "if_visual": if_visual, **kwargs})

    
    def generate_base(self, position_ids, attention_mask, inputs_embeds, max_new_tokens=1024, temperature=0.5):
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    
    def generate_language_module(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, **kwargs):
            if input_ids is None and inputs_embeds is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")

            if input_ids is not None and inputs_embeds is None:
                try:
                    inputs_embeds = self.get_model().embed_tokens(input_ids)
                except AttributeError as e:
                    raise AttributeError("Failed to generate inputs_embeds from input_ids. Ensure model is properly initialized.") from e

            if attention_mask is None and input_ids is not None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)

            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            if position_ids is not None:
                position_ids = position_ids.to(self.device)
            kwargs["do_sample"] = kwargs.get("do_sample", False)
            kwargs["temperature"] = kwargs.get("temperature", 1.0)
            kwargs["top_p"] = kwargs.get("top_p", 1.0)
            kwargs["top_k"] = kwargs.get("top_k", None)
            kwargs["num_beams"] = kwargs.get("num_beams", 1)
            with torch.no_grad():
                return super(Videollama3Qwen2ForCausalLM, self).generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    **kwargs
                )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs
