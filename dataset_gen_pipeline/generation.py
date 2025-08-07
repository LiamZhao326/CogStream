# -*- coding: utf-8 -*-
from glob import glob
import os
import json
import re
import random
from tqdm import tqdm
import argparse

from tools.MLLMs import GPT
from tools.all_prompt import *

def write_json_data(data, json_path):
    """实时写入 JSON 数据到文件"""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    try:
        if not os.path.exists(json_path):
            with open(json_path, 'w', encoding='utf-8') as json_file:
                json.dump([data], json_file, ensure_ascii=False, indent=4)
        else:
            with open(json_path, 'r+', encoding='utf-8') as json_file:
                try:
                    existing_data = json.load(json_file)
                except json.JSONDecodeError:
                    existing_data = []
                if not isinstance(existing_data, list):
                    existing_data = []
                existing_data.append(data)
                json_file.seek(0)
                json.dump(existing_data, json_file, ensure_ascii=False, indent=4)
                json_file.truncate()
    except Exception as e:
        print(f"写入 JSON 文件失败: {e}")

def check_qa_pairs(data, last_key=0):
    """检查 QA 对的结构和连续性"""
    if isinstance(data, dict):
        for key, value in data.items():
            if not value:
                return False
            if key.startswith('Q') and key[1:].isdigit():
                current_id = int(key[1:])
                if current_id == 1:
                    last_key = 0
                if current_id != last_key + 1:
                    return False
                last_key = current_id
                expected_a_key = f"A{key[1:]}"
                if expected_a_key not in data or not check_qa_pairs(data[expected_a_key], last_key):
                    return False
                if not check_qa_pairs(data[key], last_key):
                    return False
            elif not check_qa_pairs(value, last_key):
                return False
        return True
    elif isinstance(data, list):
        return all(check_qa_pairs(item, last_key) for item in data)
    return True

def gpt_response(prompt, gpt_client, images_url=None, max_retries=5):
    """调用 GPT API 并处理重试"""
    retry_count = 0
    while retry_count <= max_retries:
        try:
            response = gpt_client.vision(prompt, imagesFetched=images_url) if images_url else gpt_client.chat(prompt)
            if response_format := QA_format(response):
                if check_qa_pairs(response_format):
                    return response_format
            print(f"QA 格式不正确，重试中 ({retry_count + 1}/{max_retries}): {response_format}")
            retry_count += 1
        except Exception as e:
            print(f"GPT 调用失败: {e}")
            retry_count += 1
    raise ConnectionError(f"GPT 调用失败超过 {max_retries} 次")

def merge_dicts(a: dict, b: dict, key_prefixes=('Q', 'A')) -> dict:
    """合并字典并自动递增键序号"""
    def _process_item(target, source):
        for key, value in source.items():
            if any(key.startswith(prefix) for prefix in key_prefixes):
                prefix = next(p for p in key_prefixes if key.startswith(p))
                existing_keys = [k for k in target.keys() if k.startswith(prefix)]
                new_num = max([int(k[len(prefix):]) for k in existing_keys], default=0) + 1
                new_key = f"{prefix}{new_num}"
            else:
                new_key = key

            if isinstance(value, dict):
                target.setdefault(new_key, {})
                _process_item(target[new_key], value)
            else:
                target[new_key] = value if new_key not in target else [target[new_key], value]
    
    result = a.copy()
    _process_item(result, b)
    return result

def QA_format(text):
    """解析 GPT 响应中的 JSON 数据"""
    if not text:
        return None
    text = re.sub(r'("[^"]+":\s*"[^"]*")\s*("[^"]+":\s*"[^"]*")', r'\1, \2', text)
    try:
        if text.startswith('```'):
            parts = text.split('```')
            for part in parts:
                if part.strip().startswith('json'):
                    return json.loads(part.replace("json", "", 1).strip())
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败: {e}")
        return None

def main(args):
    """主函数：处理视频并生成 QA 对"""
    root_dir = args.root_dir
    videos_file = os.path.join(root_dir, 'segments')
    keyframe_file = os.path.join(root_dir, 'keyframes')
    output_file = os.path.join(root_dir, 'QA_pairs')
    seg_json_file = args.seg_json_file

    gpt_client = GPT(model=args.GPT_model)

    with open(seg_json_file, 'r', encoding='utf-8') as json_file:
        vid_infos = json.load(json_file)

    for vid_id, vid_info in enumerate(tqdm(vid_infos, desc="Processing Videos")):
        vid_name = vid_info['name']
        seg_timestamp = vid_info['time_stamps']
        vid_path = os.path.join(videos_file, vid_name)
        json_path = os.path.join(output_file, f"{vid_name}.json")

        if os.path.exists(json_path):
            print(f"Video_{vid_name} 已处理，跳过")
            continue

        if not os.path.isdir(vid_path):
            print(f"未找到视频目录: {vid_path}")
            continue

        print(f"处理 Video_{vid_name}")
        cr_info = vid_info.get("co-reference", [])
        cr_all_params = {cr["object"]: {'frames_url': [], 'first': False, 'second': False, 'cr_qa': None, 'p_context': ''} 
                         for cr in cr_info}
        
        segment_files = sorted(glob(os.path.join(vid_path, f'{vid_name}_segment_*.mp4')) or 
                              glob(os.path.join(vid_path, 'segment_*.mp4')))
        if not segment_files:
            print(f"未找到视频文件: {vid_path}")
            continue

        lucky_seg = random.randint(1, len(segment_files) - 1)
        print(f"Dialogue Recall片段: {lucky_seg}/{len(segment_files)}")
        
        global_summary = ""
        qa_pairs = {"title": "", "QA_pairs": {"L1": {}, "L3": {}}}

        for seg_id, seg_path in enumerate(segment_files, 1):
            kf_path_lst = sorted(glob(os.path.join(keyframe_file, vid_name, f'keyframe_{seg_id}_*.jpg')),
                                 key=lambda x: int(re.search(r'keyframe_\d+_(\d+)', x).group(1)))
            if not kf_path_lst:
                print(f"未找到关键帧: {seg_id}")
                continue

            kf_path_lst = [x.replace('\\', '/') for x in kf_path_lst]
            cur_seg_time = seg_timestamp[seg_id - 1]
            duration = cur_seg_time[-1] - cur_seg_time[0]
            frame_ts = [cur_seg_time[0] + duration / len(kf_path_lst) * i for i in range(len(kf_path_lst))]

            frame_ts_list = '[' + ", ".join(f"{round(t)}s" for t in frame_ts) + ']'
            images_url = kf_path_lst

            # 生成 QA 对
            prompt = creat_prompt(first=True, time=frame_ts_list)
            qa_pairs = gpt_response(prompt, gpt_client, images_url)
            qa_pairs_l3 = gpt_response(creat_prompt(history=global_summary, time=frame_ts_list), gpt_client, images_url)
            qa_pairs_l3 = gpt_response(polish(qa_pairs_l3), gpt_client)
            qa_pairs['QA_pairs'].update(qa_pairs_l3)

            # 生成总结
            prompt = creat_prompt(history=None, qa_pairs=qa_pairs['QA_pairs'], summarize=True)
            summary = gpt_response(prompt, gpt_client)
            t = f'{cur_seg_time[0]:.2f}s to {cur_seg_time[-1]:.2f}s'
            global_summary += f"({t}) Clip {seg_id}-{qa_pairs['title']}: [{', '.join(map(str, summary.values()))}]\n"

            # 处理全局 QA 和回归问题
            if seg_id == len(segment_files):
                prompt = creat_prompt(Global=global_summary, time=[f"{t[0]:.2f}s to {t[-1]:.2f}s" for t in seg_timestamp])
                qa_global = gpt_response(polish(gpt_response(prompt, gpt_client), CR=True), gpt_client)
                qa_pairs['QA_pairs'].update(qa_global)

            if seg_id == lucky_seg:
                qas = {k: re.sub(r'\[.*?\]', '', v) for k, v in qa_pairs['QA_pairs']['L1'].items()}
                qa_recall = gpt_response(recall(qas), gpt_client)
                qa_pairs['QA_pairs']['Dialogue Recalling'] = qa_recall

            # 处理共指问题
            for cr in cr_info:
                cr_object = cr["object"]
                cr_time = cr["appearance_time"]
                cr_params = cr_all_params[cr_object]
                current_first_frames = [t for t in frame_ts if cr_time[0][0] <= t <= cr_time[0][1]]
                current_second_frames = [t for t in frame_ts if len(cr_time) > 1 and cr_time[1][0] <= t <= cr_time[1][1]]

                if current_first_frames and not cr_params['first']:
                    current_urls = [images_url[idx] for idx, t in enumerate(frame_ts) if t in current_first_frames]
                    time_str = f"{int(cr_time[0][0])}s to {int(cr_time[0][1])}s"
                    p_context = f"({time_str}) segment {seg_id}-{qa_pairs['title']}: {summary}"
                    cr_qa = gpt_response(cr_prompt1(cr_object, global_summary), gpt_client, current_urls)
                    qa_pairs['QA_pairs'].setdefault('Object Tracking', {}).update({cr_object: cr_qa})
                    cr_params.update({'cr_qa': cr_qa, 'p_context': p_context, 'frames_url': current_urls, 'first': True})
                elif current_second_frames and not cr_params['second']:
                    final_answer = gpt_response(cr_prompt2(cr_object, cr_params['cr_qa'], cr_params['p_context']), 
                                              gpt_client, cr_params['frames_url'])
                    qa_pairs['QA_pairs'].setdefault('Object Tracking', {}).update({cr_object: final_answer})
                    cr_params['second'] = True

            # 存储数据
            vid_data = {
                "segment_path": seg_path,
                "segment_id": seg_id,
                "summary": summary,
                "title": qa_pairs['title'],
                "segment_timestamp": cur_seg_time,
                "QA_pairs": qa_pairs['QA_pairs']
            }
            write_json_data(vid_data, json_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Video processing and QA generation")
    parser.add_argument("--seg_json_file", default=r"video_segments_train.json", 
                        type=str, help="Path to segment JSON file")
    parser.add_argument("--root_dir", default="./Dataset", type=str, help="Root directory of dataset")
    parser.add_argument("--GPT_model", default="", type=str, help="GPT model to use")
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    main(args)