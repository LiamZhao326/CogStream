# -*- coding: utf-8 -*-
import json
import os
import re
import copy
import time
import logging
from tqdm import tqdm
import argparse
from tools.MLLMs import GPT
from tools.all_prompt import *

"""
Relevance QA Set (RS):
1. RS Storage Structure
   - QA Pairs: Stored in a dictionary with structure "segment_idx":"Level":"QA_idx", representing QA pairs across segments and levels.
   - Scores: Relevance scores for each CQA with PQAs stored in a similar structure: "segment_idx":"Level":"(QA_idx, scores)".

2. link_raw
   - link_raw (dict): Stores raw relevance scores for PQAs and CQAs, enabling dynamic filtering to select highly relevant PQAs for RS.
   - Two-layer nested dict structure:
     - First layer: "segment_idx":"Level":"CQA_idx" for current QA pairs.
     - Second layer: "segment_idx":"Level":"[PQA_idx, scores]" for associated indices.
"""

def setup_logging(output_folder):
    """Configure logging to record errors in a file."""
    log_file = os.path.join(output_folder, 'video_processing_errors.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - Video: %(video_name)s - %(message)s',
        filemode='a'
    )

def integrate_data(input_data):
    """Integrate and clean QA pairs for processing."""
    output = {}
    allowed_levels = {'L1', 'L3', 'L4'}

    for level in input_data:
        if level not in allowed_levels:
            continue

        current_level = input_data[level]
        new_level = {}
        q_keys = sorted(
            [k for k in current_level if k.startswith('Q')],
            key=lambda x: int(x[1:])
        )

        for q_key in q_keys:
            a_key = q_key.replace('Q', 'A')
            q_text = current_level[q_key]
            a_text = current_level.get(a_key, '')
            if 'co-reference' in q_text.lower():
                continue
            clean_q = re.sub(r'\[.*?\]', '', q_text).strip()
            clean_a = re.sub(r'\s*\([^)]*\)', '', a_text).strip()
            new_level[q_key] = f"{clean_q} {clean_a}"

        output[level] = new_level

    return output

def validate_format(name, valid_prefixes, description):
    """Validate the format of a key."""
    if not any(name.startswith(prefix) for prefix in valid_prefixes) or not name[-1].isdigit():
        print(f"Invalid format for {description}: {name}")

def process_scores(clevel, scores, cqa_seg_id, link_raw, pqa_seg_id=None, plevel=None):
    """Process and store relevance scores."""
    def extract_score():
        ok_raw = {}
        segments_dict = {f"segment {pqa_seg_id}": qa_info} if pqa_seg_id else qa_info

        for seg_id, seg_data in segments_dict.items():
            seg_id_num = pqa_seg_id if pqa_seg_id else seg_id.split()[-1]
            if not pqa_seg_id:
                validate_format(seg_id, ["segment"], "segment")

            if plevel:
                preQAs_raw = [(f"QA{qa_id[-1]}", qa_score) for qa_id, qa_score in seg_data.items()
                              if validate_format(qa_id, ["QA", "Q", "P", "PQA"], "preQA") or True]
                ok_raw.setdefault(f"segment {seg_id_num}", {})[plevel] = preQAs_raw
            else:
                for qa_lv, qa_scores in seg_data.items():
                    if qa_lv not in keys:
                        raise ValueError(f"Invalid level: {qa_lv}")
                    preQAs_raw = [(f"QA{qa_id[-1]}", qa_score) for qa_id, qa_score in qa_scores.items()
                                  if validate_format(qa_id, ["QA", "Q", "P", "PQA"], "preQA") or True]
                    ok_raw.setdefault(f"segment {seg_id_num}", {})[qa_lv] = preQAs_raw

        if not all([clevel, scores, cqa_seg_id]):
            raise ValueError('Parameters must be specified')
        cqa_seg_key = f"segment {cqa_seg_id}"
        nested_dict = link_raw.setdefault(cqa_seg_key, {}).setdefault(clevel, {})
        qa_key = f"QA{qa[-1]}"
        if qa_key in nested_dict:
            pqa_seg_key = f"segment {pqa_seg_id}"
            nested_dict[qa_key].setdefault(pqa_seg_key, {}).update(ok_raw.get(pqa_seg_key, ok_raw))
        else:
            nested_dict[qa_key] = ok_raw

    if clevel == 'L2':
        keys = ['L1']
        for qa, qa_info in scores.items():
            validate_format(qa, ["QA", "Q", "C", "CQA"], "response")
            extract_score()
    elif clevel in ['L3', 'L4']:
        keys = ['L1', 'L3']
        for qa, qa_info in scores.items():
            validate_format(qa, ["QA", "Q", "C", "CQA"], "response")
            extract_score()
    else:
        raise ValueError(f"Invalid level: {clevel}")

def check_coreference(video_data, link_raw):
    """Extract and score co-reference QA pairs."""
    for seg_info in video_data:
        qa_info = seg_info['QA_pairs']
        seg_num = seg_info['segment_id']
        score_l2 = {}

        for l2_key, l2_value in qa_info.get("L2", {}).items():
            if l2_key.startswith("Q") and l2_value.startswith("[Co-reference]"):
                matches = re.findall(r'\(([^)]+)\)', l2_value)
                l1_key = int(matches[0][-1])
                score_l2[l2_key] = {
                    f'segment {seg_num}': {
                        'L1': {f'QA{l1_key}': 7}
                    }
                }
        process_scores(clevel='L2', scores=score_l2, cqa_seg_id=seg_num, link_raw=link_raw)

def QA_format(text):
    """Parse JSON data from GPT response."""
    if not text:
        return None
    text = re.sub(r'("[^"]+":\s*"[^"]*")\s*("[^"]+":\s*"[^"]*")', r'\1, \2', text)
    text = text.replace('},\n    }', '}\n    }')
    try:
        if text.startswith('```'):
            parts = text.split('```')
            for part in parts:
                if part.strip().startswith('json'):
                    return json.loads(part.replace("json", "", 1).strip())
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}\n{text}")
        return None

system_prompt = """
You are a VQA question chain evaluation expert. Analyze the logic and information dependency strength of **Current Question-Answer pairs (CQA)** on **Previous Question-Answer pairs (PQA)** using only the provided QA text.
"""

def q_to_p(text):
    """Replace 'Q' with 'P' in dictionary keys."""
    return {re.sub(r'Q(\d+)', r'P\1', k): v for k, v in text.items()}

def q_to_c(text):
    """Replace 'Q' with 'C' in dictionary keys."""
    return {re.sub(r'Q(\d+)', r'C\1', k): v for k, v in text.items()}

def main(args):
    """Process video QA data and generate relevance scores."""
    input_path = args.inputPath
    output_folder = args.output_folder
    gpt_client = GPT(model=args.GPT_model, temperature=0.8)
    total_vid_num = len(os.listdir(input_path))

    setup_logging(output_folder)

    for vid_num, video_qa_file in enumerate(tqdm(os.listdir(input_path), desc="Processing Videos")):
        vid_name = os.path.splitext(video_qa_file)[0]
        json_path = os.path.join(output_folder, f"{vid_name}.json")
        try:
            with open(os.path.join(input_path, video_qa_file), 'r', encoding='utf-8') as jsonfile:
                video_data = json.load(jsonfile)
                history = {}

                link_raw = json.load(open(json_path, 'r', encoding='utf-8')) if os.path.exists(json_path) else {}
                if not link_raw:
                    check_coreference(video_data, link_raw)

                for seg_num, seg_data in enumerate(video_data):
                    segment_id = int(seg_data['segment_id'])
                    qa_pairs = seg_data['QA_pairs']
                    output = integrate_data(qa_pairs)
                    history[f"{segment_id}"] = output
                    keys = ["L1", "L3"]

                    # Process L3
                    if not link_raw.get(f'segment {segment_id}', {}).get('L3'):
                        for i in range(1, len(history) + 1):
                            pre_qa = {key: history[str(i)][key] for key in history[str(i)].keys() if key in keys}
                            for l in ['L1', 'L3']:
                                prompt = scoring_prompt(current=q_to_c(output.get('L3', {})), history=q_to_p(pre_qa[l]))
                                while True:
                                    t_start = time.time()
                                    answer = gpt_client.chat(prompt, system=system_prompt)
                                    if answer and (scores := QA_format(answer)):
                                        print(f"GPT response time: {time.time() - t_start:.2f}s")
                                        link_raw_tempt = copy.deepcopy(link_raw)
                                        try:
                                            process_scores(clevel='L3', plevel=l, scores=scores,
                                                          cqa_seg_id=segment_id, pqa_seg_id=i, link_raw=link_raw)
                                            break
                                        except Exception as e:
                                            print(f"Error processing scores: {e}\n{scores}")
                                            link_raw = link_raw_tempt
                                    break

                    # Process L4
                    if not link_raw.get(f'segment {segment_id}', {}).get('L4') and 'L4' in output:
                        for i in range(1, len(history) + 1):
                            pre_qa = {key: history[str(i)][key] for key in history[str(i)].keys() if key in keys}
                            for l in ['L1', 'L3']:
                                prompt = scoring_prompt(current=q_to_c(output['L4']), history=q_to_p(pre_qa[l]))
                                while True:
                                    t_start = time.time()
                                    answer = gpt_client.chat(prompt, system=system_prompt)
                                    if answer and (scores := QA_format(answer)):
                                        print(f"GPT response time: {time.time() - t_start:.2f}s")
                                        link_raw_tempt = copy.deepcopy(link_raw)
                                        try:
                                            process_scores(clevel='L4', plevel=l, scores=scores,
                                                          cqa_seg_id=segment_id, pqa_seg_id=i, link_raw=link_raw)
                                            break
                                        except Exception as e:
                                            print(f"Error processing scores: {e}\n{scores}")
                                            link_raw = link_raw_tempt
                                    break

                    os.makedirs(output_folder, exist_ok=True)
                    with open(json_path, 'w', encoding='utf-8') as json_file:
                        json.dump(link_raw, json_file, ensure_ascii=False, indent=4)

        except Exception as e:
            logging.error(f"Error processing video: {str(e)}", extra={'video_name': vid_name})
            print(f"Error processing {vid_name}: {str(e)}. Skipping.")
            continue

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate QA Table of Inference")
    parser.add_argument("--inputPath", default=r"C:\Users\COG27\Desktop\code\code\2_test\QA_pairs_new", type=str,
                        help="Input directory for QA JSON files")
    parser.add_argument("--output_folder", default=r"C:\Users\COG27\Desktop\code\code\2_test\QA_ToI", type=str,
                        help="Output directory for processed JSON files")
    parser.add_argument("--GPT_model", default="gpt-4o-2024-11-20", type=str, help="GPT model to use")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)