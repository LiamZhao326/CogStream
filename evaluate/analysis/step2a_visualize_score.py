
import os,json
flag = True
path = 'CogStream.json'
with open(path,'r')as f:
    datas = json.load(f)


levels = ["Basic", "Streaming", "Global"]
keys_order = ["IA", "DC", "CA", "TP", "LC"]
scores = []
LEVELS = {
    "Basic": ["Basic/Attributes", "Basic/Items", "Basic/Co-reference", "Basic/Actions"],
    "Streaming": ["Streaming/Reasoning", "Streaming/Sequence Perception", "Streaming/Dialogue Recalling", "Streaming/Dynamic Updating", "Streaming/Object Tracking"],
    "Global": ["Global/Overall Summary", "Global/Global Analysis"]
}
SCORE = {
    "Basic":["IA","DC","CA","TP","LC"],
    "Streaming":["IA","DC","CA","TP","LC"],
    "Global":["IA","DC","CA","TP","LC"],
    "IA":['Basic','Streaming','Global'],
    "DC":['Basic','Streaming','Global'],
    "CA":['Basic','Streaming','Global'],
    "TP":['Basic','Streaming','Global'],
    "LC":['Basic','Streaming','Global']
}


try:
    sorted_models_by_mean = sorted(
        datas.keys(),
        key=lambda model_name: datas[model_name].get("Mean", float('-inf')), 
        reverse=True 
    )
except TypeError:
    sorted_models_by_mean = list(datas.keys())
metrics_part1 = ["IA", "DC", "CA", "TP", "LC", "Mean"] if not flag else ["Basic", "Streaming", "Global", "Mean"]
print('metrics\t\t\t\t& ' + '\t& '.join([x[:3]+'.' for x in metrics_part1]) + ' \\\\')
for model in sorted_models_by_mean:
    data = datas[model]
    scores = []
    if flag: 
        for level in metrics_part1: 
            if level in data.get("QA Level", {}): 
                level_data = data["QA Level"][level]
                ordered_scores = []
                # for key in keys_order: 
                #     if key not in SCORE.get(level, {}): continue 
                #     score_val = level_data.get(key)
                #     if isinstance(score_val, (int, float)):
                #         ordered_scores.append(f"{score_val:.2f}")
                #     else:
                #         ordered_scores.append(str(score_val) if score_val is not None else "N/A")
                score_val = level_data.get('mean')
                if isinstance(score_val, (int, float)):
                    ordered_scores.append(f"{score_val:.2f}")
                else:
                    ordered_scores.append(str(score_val) if score_val is not None else "N/A")
                scores.extend(ordered_scores)
            else: 
                score_val = data.get(level)
                if isinstance(score_val, (int, float)):
                    scores.append(f"{score_val:.2f}")
                else:
                    scores.append(str(score_val) if score_val is not None else "N/A")
                # scores.extend(["N/A"] * len(keys_order_for_this_level))
                pass 
    else:
        for metric_key in metrics_part1:
            score_val = data.get(metric_key)
            if isinstance(score_val, (int, float)):
                scores.append(f"{score_val:.2f}") 
            else:
                scores.append(str(score_val) if score_val is not None else "N/A") 
                

    print(f"{model:<25}\t& " + '\t& '.join(scores) + ' \\\\')

print("=========================")



print("\n--- QA Class Scores (Sorted by Mean from Part 1) ---")
header_part2_elements = []
for level, keys_in_level in LEVELS.items():
    for key_in_level in keys_in_level:
        header_part2_elements.append(f"{level}_{key_in_level}") 
print('model_name\t\t& ' + '\t& '.join(header_part2_elements) + ' \\\\')


for model in sorted_models_by_mean: 
    data = datas[model]
    result_part2 = []
    qa_class_data = data.get("QA Class", {}) 

    for level_name, keys_in_level in LEVELS.items():
        # level_scores = [] 
        for key in keys_in_level:
            score_val = qa_class_data.get(key) 
            if isinstance(score_val, (int, float)):
                result_part2.append(f"{score_val:.1f}")
            else:
                result_part2.append(str(score_val) if score_val is not None else "N/A")
    
    print(f"{model:<25}\t& " + '\t& '.join(result_part2) + ' \\\\')
