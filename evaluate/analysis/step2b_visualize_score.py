'''
模型分数可视化——>输出结果符合latex格式, 可直接复制
'''
flag = True
import os,json
path = '/mnt/nvme0n1/liamz/StreamVICL-main/streamvicl/eval/eval_data/CogReasoner_5_14_exp1_v5_1_4/streamvicl.json'

path = '/mnt/nvme0n1/liamz/StreamVICL-main/streamvicl/eval/eval_data/CogReasoner_5_16_exp2_new/streamvicl.json'
path = '/mnt/nvme0n1/liamz/StreamVICL-main/streamvicl/eval/eval_data/CogReasoner_5_16_exp2/streamvicl.json'
# path = '/mnt/nvme0n1/liamz/StreamVICL-main/streamvicl/eval/eval_data/CogReasoner_5_14_gemini_new/streamvicl.json'
# path = '/mnt/nvme0n1/liamz/StreamVICL-main/streamvicl/eval/eval_data/Cog_backup/CogReasoner_5_13_backup(100video)/streamvicl.json'
# path = path.replace('.json','_aligned.json')

with open(path,'r')as f:
    datas = json.load(f)

print(path)
# 按照 L1, L2, L3, L4 的顺序提取分数，并确保顺序是 [IA, DC, CA, TP, LC]
levels = ["Basic", "Streaming", "Global"]
keys_order = ["IA", "DC", "CA", "TP", "LC"]
scores = []
# levels = {
#     "Streaming/Reasoning": ["Streaming/Reasoning", "Streaming/Analysis", "Streaming/Causality", "Streaming/Causal Discovery","Streaming/Ingredients Analysis", "Streaming/Intention", "Streaming/Prediction"]
# }
LEVELS = {
    "Basic": ["Basic/Attributes", "Basic/Items", "Basic/Co-reference", "Basic/Actions", "Basic/Environment"],
    "Streaming": ["Streaming/Reasoning", "Streaming/Sequence Perception", "Streaming/Dialogue Recalling", "Streaming/Temporal Perception", "Streaming/Dynamic Updating", "Streaming/Object Tracking"],
    "Global": ["Global/Overall Summary", "Global/Global Analysis"]
}
# LEVELS = {
#     "Basic": ["Basic/Attributes", "Basic/Items", "Basic/Co-reference", "Basic/Actions"],
#     "Streaming": ["Streaming/Reasoning", "Streaming/Sequence Perception", "Streaming/Dialogue Recalling","Streaming/Dynamic Updating"],
#     "Global": ["Global/Overall Summary", "Global/Global Analysis"]
# }
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

# --- 第一部分打印 (基于 Mean 排序) ---
# 1. 提取 Mean 分数并用于排序
#    我们假设每个模型都有 'Mean' 键，并且其值是数字
try:
    # 创建一个包含 (model_name, mean_score) 的元组列表
    # 如果 'Mean' 可能不存在或不是数字，需要更复杂的错误处理
    sorted_models_by_mean = sorted(
        datas.keys(),
        key=lambda model_name: datas[model_name].get("Mean", float('-inf')), # 使用 get 提供默认值以防 'Mean' 缺失，float('-inf') 使缺失的排在最后
        reverse=True # 降序排列
    )
    # sorted_models_by_mean.pop(sorted_models_by_mean.index('hallucination_2640'))
except TypeError:
    print("错误：'Mean' 值可能不是可比较的数字类型，或者模型数据结构不一致。请检查 datas 字典。")
    # 可以选择在这里退出，或者用原始顺序打印
    sorted_models_by_mean = list(datas.keys())
metrics_part1 = ["IA", "DC", "CA", "TP", "LC", "Mean"] if not flag else ["Basic", "Streaming", "Global", "Mean"]
print('metrics\t\t\t\t& ' + '\t& '.join([x[:3]+'.' for x in metrics_part1]) + ' \\\\') # 调整制表符以更好对齐
for model in sorted_models_by_mean:
    data = datas[model]
    scores = []
    # 你的 flag 逻辑保持不变
    if flag: # 假设 flag 和相关变量已定义
        for level in metrics_part1: # 假设 levels 已定义
            if level in data.get("QA Level", {}): # 使用 .get 以防 "QA Level" 缺失
                level_data = data["QA Level"][level]
                ordered_scores = []
                # for key in keys_order: # 假设 keys_order 已定义
                #     if key not in SCORE.get(level, {}): continue # 使用 .get 以防 SCORE[level] 缺失
                #     # 确保取出的值是数字，并且格式化为两位小数
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
                # 如果某个 level 缺失，你可能想填充占位符
                # 例如，如果 keys_order 对每个 level 都适用且长度固定
                # scores.extend(["N/A"] * len(keys_order_for_this_level))
                pass # 或者根据你的需求处理
    else:
        # 直接从顶层键获取，确保格式化
        # 并且按照 metrics_part1 的顺序获取，以防万一
        for metric_key in metrics_part1:
            score_val = data.get(metric_key)
            if isinstance(score_val, (int, float)):
                scores.append(f"{score_val:.2f}") # 格式化为两位小数
            else:
                scores.append(str(score_val) if score_val is not None else "N/A") # 处理 None 或其他类型
                
    # 调整模型名称的打印宽度，使其大致对齐
    print(f"{model:<25}\t& " + '\t& '.join(scores) + ' \\\\')

print("=========================")


# --- 第二部分打印 (如果也需要基于 Mean 排序，则使用相同的 sorted_models_by_mean) ---

print("\n--- QA Class Scores (Sorted by Mean from Part 1) ---")
# 确定第二部分表格的表头（如果需要）
header_part2_elements = []
for level, keys_in_level in LEVELS.items():
    for key_in_level in keys_in_level:
        header_part2_elements.append(f"{level}_{key_in_level}") # 示例表头元素
print('model_name\t\t& ' + '\t& '.join(header_part2_elements) + ' \\\\')


for model in sorted_models_by_mean: # 使用与第一部分相同的排序顺序
    data = datas[model]
    result_part2 = []
    qa_class_data = data.get("QA Class", {}) # 使用 .get 以防 "QA Class" 缺失

    # 确保 LEVELS 中的键存在于 qa_class_data 中
    # 并且按照 LEVELS 中定义的顺序和结构来提取
    for level_name, keys_in_level in LEVELS.items(): # 假设 LEVELS 是一个有序字典或其迭代顺序是你想要的
        # level_scores = [] # 如果你想按 level 分组打印，但这看起来是平铺的
        for key in keys_in_level:
            score_val = qa_class_data.get(key) # 从实际的 QA Class 数据中获取
            if isinstance(score_val, (int, float)):
                result_part2.append(f"{score_val:.1f}")
            else:
                result_part2.append(str(score_val) if score_val is not None else "N/A")
    
    print(f"{model:<25}\t& " + '\t& '.join(result_part2) + ' \\\\')