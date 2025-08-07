import json
import re
import os
import glob
from collections import defaultdict
import functools
from tqdm import tqdm
import numpy as np

root = 'eval_data/CogReasoner_score'
score_files = [f'{root}/IA', f'{root}/DC', f'{root}/CA', f'{root}/TP', f'{root}/LC']
output_json = f'{root}/CogStream.json'
raw_datas = glob.glob('Test_dataset/VQA_Dataset/*.json')
N = -1

video_data_cache = {}
for rd in raw_datas:
    vid_name = os.path.basename(rd).replace('.json', '')
    with open(rd, 'r') as f:
        video_data_cache[vid_name] = json.load(f)

levels = {
    "Streaming/Reasoning": ["Streaming/Reasoning", "Streaming/Analysis", "Streaming/Causality", 
                            "Streaming/Causal Discovery", "Streaming/Causal discovery", 
                            "Streaming/Ingredients Analysis", "Streaming/Intention", "Streaming/Prediction"]
}
LEVELS = {
    "Basic": ["Basic/Attributes", "Basic/Items", "Basic/Co-reference", "Basic/Actions"],
    "Streaming": ["Streaming/Reasoning", "Streaming/Analysis", "Streaming/Causality", 
                  "Streaming/Causal Discovery", "Streaming/Causal discovery", "Streaming/Ingredients Analysis", 
                  "Streaming/Intention", "Streaming/Prediction", "Streaming/Sequence Perception", 
                  "Streaming/Dialogue Recalling", "Streaming/Dynamic Updating", "Streaming/Object Tracking"],
    "Global": ["Global/Overall Summary", "Global/Global Analysis"]
}
SCORE = {
    "Basic": ["IA", "DC", "CA", "TP", "LC"],
    "Streaming": ["IA", "DC", "CA", "TP", "LC"],
    "Global": ["IA", "DC", "CA", "TP", "LC"],
    "IA": ['Basic', 'Streaming', 'Global'],
    "DC": ['Basic', 'Streaming', 'Global'],
    "CA": ['Basic', 'Streaming', 'Global'],
    "TP": ['Basic', 'Streaming', 'Global'],
    "LC": ['Basic', 'Streaming', 'Global']
}
k = 5 # Decimal Places to Be Retained


@functools.lru_cache(maxsize=None)
def extracted_label(seq, qaid, vid_name):
    data = video_data_cache[vid_name][seq]['Data'][qaid]
    if data["ID"] != qaid:
        raise ValueError(f"ID not match: {data['ID']} vs {qaid}")
    label = data['label']
    segid = data['segment_path'].split('_')[-1].split('.')[0]
    for key, value in levels.items():
        if label in value:
            label = key
            break
    level = next((l for l, v in LEVELS.items() if label in v), None)
    if level is None:
        raise ValueError(f"Level not found for {label}")
    return label, segid, level

def average_adjacent_coherence(nums):
    def normalize_scores(scores):
        min_score, max_score = 0, 10
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    if len(nums) < 2:
        return 1
    nums_ = normalize_scores(nums)
    total_diff = sum(abs(nums_[i] - nums_[i+1]) for i in range(len(nums_) - 1))
    avg_diff = total_diff / (len(nums_) - 1)
    return round((1 - avg_diff) * 10, k)

final_scores = {}
qa_class = defaultdict(lambda: defaultdict(list))
qa_level = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
inter_conherence = defaultdict(list)
exter_conherence = defaultdict(list)
vids = set()
models = set()

for score_file in tqdm(score_files, desc='Loading data'):
    score_type = os.path.basename(score_file)
    model_files = os.listdir(score_file)
    for model_f in model_files:
        models.add(model_f)
        json_files = glob.glob(os.path.join(score_file, model_f, '*.json'))
        all_score = []
        for file in sorted(json_files[:N]):
            with open(file, 'r') as f:
                vid_data = json.load(f)
            for item in vid_data["score"]:
                index = next(iter(item.keys()))
                seq, qaid = map(int, re.findall(r'\d+', index))
                label, _, level = extracted_label(seq, qaid, vid_data["video_name"])
                if level not in SCORE[score_type]:
                    continue
                try:
                    score = int(next(iter(item.values())))
                    all_score.append(score)
                except Exception as e:
                    print(f"Error in {file}: {e}")
                vids.add(vid_data["video_name"])
        final_score = round(sum(all_score) / len(all_score), 2) * 10 if all_score else 0
        final_scores.setdefault(model_f, {}).update({score_type: final_score})

for model in tqdm(models, desc='Processing models'):
    for vid in vids:
        SKIP = False
        vid_scors = defaultdict(dict)
        for score_file in score_files:
            score_type = os.path.basename(score_file)
            json_file = os.path.join(score_file, model, f'{vid}.json')
            if not os.path.exists(json_file):
                print(f"Video {json_file} does not exist")
                SKIP = True
                break
            with open(json_file, 'r') as f:
                vid_data = json.load(f)
            for item in vid_data["score"]:
                index = next(iter(item.keys()))
                seq, qaid = map(int, re.findall(r'\d+', index))
                label, seg_id, level = extracted_label(seq, qaid, vid)
                if level not in SCORE[score_type]:
                    continue
                vid_scors[f"{seg_id},{qaid}"].update({
                    "label": label,
                    "level": level,
                    score_type: int(item[index]) if level in SCORE[score_type] else None
                })
        if SKIP:
            continue
        mean_scores = []
        last_seg = None
        inter, exter = [0], [0]
        for key, qa_data in vid_scors.items():
            seg_id, qa_id = map(int, key.split(','))
            valid_scores = [qa_data[st] for st in SCORE[qa_data["level"]] if qa_data.get(st) is not None]
            mean_score = round(sum(valid_scores) / len(valid_scores), k) if valid_scores else 0
            vid_scors[key]["Mean"] = mean_score
            mean_scores.append(mean_score)
            qa_class[model][qa_data["label"]].append(mean_score)
            for st in SCORE[qa_data["level"]]:
                if qa_data.get(st) is not None:
                    qa_level[model][qa_data["level"]][st].append(qa_data[st])
            if seg_id != last_seg and qa_id != 0:
                inter_conherence[model].append(average_adjacent_coherence(inter))
                exter.append(round(sum(inter) / len(inter), k))
                inter = []
            inter.append(mean_score)
            last_seg = seg_id
        exter_conherence[model].append(average_adjacent_coherence(exter))

def get_mean_score_(dict_score,MEAN=False):
    if isinstance(dict_score, dict,):
        all_score = []
        for key in dict_score:
            all_score.extend(dict_score[key])
            dict_score[key] = round(sum(dict_score[key]) / len(dict_score[key]), 2) * 10
        if MEAN:
            dict_score['mean'] = round(sum(all_score) / len(all_score), 2) * 10
    elif isinstance(dict_score, list):
        return round(sum(dict_score) / len(dict_score), 2) * 10 if dict_score else 0

for model, scores in final_scores.items():
    get_mean_score_(qa_class[model])
    for k in qa_level[model].keys():
        get_mean_score_(qa_level[model][k],MEAN=True)
    inter_con_score = get_mean_score_(inter_conherence[model])
    exter_con_score = get_mean_score_(exter_conherence[model])
    mean_score = round(sum(scores.values()) / len(scores), 2)
    final_scores[model].update({
        "QA Class": qa_class[model],
        "QA Level": qa_level[model],
        "Inter Coherence": inter_con_score,
        "Exter Coherence": exter_con_score,
        "Mean": mean_score
    })

with open(output_json, 'w') as f:
    json.dump(final_scores, f, indent=4)
