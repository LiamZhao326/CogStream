# -*- coding: utf-8 -*-
import json
import os, copy, re
import random
from collections import defaultdict

from tqdm import tqdm
import argparse
import time
import random

import re
import random


def remix(video_data):
    N_seg = len(video_data) - 1 # 10duan -> 9
    changes_to_apply = []
    temporal_count = 0  # Renamed from 'temperal' to 'temporal_count' for clarity
    K = 2  # Minimum required temporal perception QAs
    ORI_SEG,ORI_QAID =[],[]

    # First pass: Process existing Temporal Perception and Dialogue Recalling
    for seg_id, seg_info in enumerate(video_data): # 0~9
        qa_info = seg_info['QA_pairs']

        # Process Dialogue Recalling
        if "Dialogue Recalling" in qa_info:
            tempt_qa = qa_info["Dialogue Recalling"].copy()
            tempt_qa["Original_seg_ID"]=seg_id
            target_seg_id = random.randint(seg_id + 1, N_seg) if seg_id < N_seg else N_seg
            changes_to_apply.append(
                (target_seg_id, "Dialogue Recalling", tempt_qa)
            )
            ORI_SEG.append(tempt_qa["Original_seg_ID"])
            ORI_QAID.append(tempt_qa['Original_QA_ID'])
            qa_info.pop("Dialogue Recalling")


    for seg_id, seg_info in enumerate(video_data):  # 0~9
        qa_info = seg_info['QA_pairs']

        # Process Temporal Perception
        if 'L1' in qa_info:
            L1_pairs = qa_info['L1']
            keys_to_remove = set()

            for l1_key in list(L1_pairs.keys()):
                if l1_key.startswith("Q"):
                    question = L1_pairs[l1_key]
                    # Check if it's a Temporal Perception question
                    if question.startswith("[Temporal Perception]"):
                        # Check if it contains a timestamp
                        if not re.search(r'\d+s', question):
                            # Replace with Actions if no timestamp
                            L1_pairs[l1_key] = question.replace("[Temporal Perception]", "[Actions]")
                            continue

                        temporal_count += 1
                        q_key = l1_key
                        a_key = l1_key.replace('Q', 'A')

                        if a_key in L1_pairs:
                            tempt_qa = {
                                "Original_seg_ID":seg_id, # if seg = 5(No.6)
                                "QA_pairs":{
                                q_key: question.replace("[Temporal Perception]", ""),
                                a_key: L1_pairs[a_key]
                                }
                            }
                            keys_to_remove.update([q_key, a_key])
                            target_seg_id = random.randint(seg_id + 1, N_seg) if seg_id < N_seg else N_seg # then 6~9
                            changes_to_apply.append(
                                (target_seg_id, "Temporal Perception", tempt_qa)
                            )

            # Remove processed QA pairs
            for key in keys_to_remove:
                L1_pairs.pop(key, None)



    # Second pass: Find more temporal questions if needed
    if temporal_count < K:
        for seg_id, seg_info in enumerate(video_data):

            qa_info = seg_info['QA_pairs']

            L1_pairs = qa_info['L1']
            keys_to_remove = set()

            for l1_key in list(L1_pairs.keys()):
                if l1_key.startswith("Q"):
                    if (seg_id in ORI_SEG) and (l1_key[-1] in ORI_QAID):
                        continue

                    question = L1_pairs[l1_key]
                    # Find questions with timestamps but not marked as Temporal Perception
                    if re.search(r'\d+s', question) and not question.startswith("[Temporal Perception]"):
                        # Extract content in parentheses if exists
                        match = re.search( r'\[.*?\]', question)
                        if match:
                            content = match.group(0)
                            q_key = l1_key
                            a_key = l1_key.replace('Q', 'A')

                            if a_key in L1_pairs:
                                tempt_qa = {
                                    "Original_seg_ID": seg_id,
                                    "QA_pairs": {
                                        q_key: question.replace(f"{content}", ""),
                                        a_key: L1_pairs[a_key]
                                    }
                                }
                                keys_to_remove.update([q_key, a_key])
                                target_seg_id = random.randint(seg_id + 1, N_seg) if seg_id < N_seg else N_seg
                                target_seg_id = min(target_seg_id, seg_id + 2)
                                changes_to_apply.append(
                                    (target_seg_id, "Temporal Perception", tempt_qa)
                                )
                                temporal_count += 1
                        break
            # Remove processed QA pairs
            for key in keys_to_remove:
                L1_pairs.pop(key, None)

            if seg_id == N_seg or temporal_count >= K:
                break

    # Apply all changes
    for target_seg_id, qa_type, qa_content in changes_to_apply:
        if qa_type not in video_data[target_seg_id]['QA_pairs']:
            video_data[target_seg_id]['QA_pairs'][qa_type] = {}
        video_data[target_seg_id]['QA_pairs'][qa_type].update(qa_content)

    return video_data


def shrim(video_data):
    '''
    部分片段的QA过多了，限制L1 QA = 5， L3 QA = 4, L4 QA=2;
    若数量大于限制，则随机删除一组QA对
    '''
    for seg_id, seg_info in enumerate(video_data):
        qa_info = seg_info['QA_pairs']

        # 定义每个类别的 QA 对数量限制
        limits = {'L1': 5, 'L3': 4, 'L4': 2}

        # 遍历需要限制的类别
        for category, limit in limits.items():
            if category in qa_info:
                pairs = qa_info[category]

                # 找出所有问题键（以 "Q" 开头的键）
                question_keys = [key for key in pairs.keys() if key.startswith("Q")]
                N = len(question_keys)

                # 如果 QA 对数量超过限制
                if N > limit:
                    print('!!')
                    num_to_delete = N - limit  # 计算需要删除的 QA 对数量
                    keys_to_delete = random.sample(question_keys, num_to_delete)  # 随机选择要删除的问题键

                    # 删除选中的 QA 对
                    for q_key in keys_to_delete:
                        del pairs[q_key]  # 删除问题
                        a_key = q_key.replace('Q', 'A')  # 找到对应的回答键
                        if a_key in pairs:
                            del pairs[a_key]  # 删除回答（如果存在）

    return video_data

def main(inputPath,output_folder):
    os.makedirs(output_folder,exist_ok=True)
    conut = 0
    for vid_num, video_qa_file in tqdm(enumerate(os.listdir(inputPath))):

        vid_name = os.path.splitext(video_qa_file)[0]
        json_path = os.path.join(inputPath, vid_name) + '.json'
        save =  os.path.join(output_folder, vid_name) + '.json'
        if os.path.exists(save):continue
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as jsonfile:
            video_data = json.load(jsonfile)
            video_data_ = remix(video_data) # 核心函数
            video_data_ = shrim(video_data_)

        with open(save, 'w', encoding='utf-8') as jsonfile:
            json.dump(video_data_, jsonfile,indent=4)
        conut += 1
    print(conut)

def dynamic_updating(inputPath,output_folder):
    conut = 0
    with open(inputPath,'r') as f:
        dynamic_qa = json.load(f)
    os.makedirs(output_folder + '2', exist_ok=True)
    for vid_num, dy_info in tqdm(enumerate(dynamic_qa)):
        dy_qas = dy_info["Dynamic_updating"] # 默认时间戳都与分割后的视频对齐。
        dy_qas = sorted(dy_qas,key = lambda x: x['time'])
        vid_name = dy_info['name']
        if not vid_name == 'VMFu3vr8jUg':continue
        new_qa_json_path = os.path.join(output_folder, vid_name) + '.json'
        save = os.path.join(output_folder+'2', vid_name) + '.json'

        if not os.path.exists(new_qa_json_path):
            print(f'找不到 {vid_name} QA文件')
            continue
        if vid_name =='7iUyB7UNzdE':
            a = 1
        # 读取JSON文件
        with open(new_qa_json_path, 'r', encoding='utf-8') as jsonfile:
            video_data = json.load(jsonfile)
        for dy_qa in dy_qas:
            dy_time = dy_qa['time']
            for seg_id, seg_info in enumerate(video_data):
                st = seg_info['segment_timestamp'] # qa_seg_time
                # 确认动态更新QA的位置
                if st[0] <= dy_time < st[1]:
                    # 由于时间戳人工标注，认为在分割点±3s内都算误差，重新赋值为分割点；否则则认为故意而为。避免与分割点间隔太短时还要额外分割视频，意义不大。
                    if dy_time < st[0] + 3: dy_qa['time'] = st[0]
                    elif dy_time > st[1] - 3: dy_qa['time'] = st[1]
                    if 'Dynamic Updating' not in seg_info['QA_pairs']:
                        seg_info['QA_pairs']['Dynamic Updating'] = [dy_qa]
                    else:
                        seg_info['QA_pairs']['Dynamic Updating'].append(dy_qa)
                    break
        with open(save, 'w', encoding='utf-8') as jsonfile:
            json.dump(video_data, jsonfile,indent=4)
        conut += 1
    print(conut)

if __name__ == "__main__":
    inputPath = r'C:\Users\COG27\Desktop\code\code\2\QA_pairs'
    dynamicPath = r'C:\Users\COG27\Desktop\code\code\2\1\ALL_dynamic.json'
    output_folder = r'C:\Users\COG27\Desktop\code\code\2\QA_pairs_new'
    # main(inputPath,output_folder)
    if os.path.exists(dynamicPath): dynamic_updating(dynamicPath, output_folder)