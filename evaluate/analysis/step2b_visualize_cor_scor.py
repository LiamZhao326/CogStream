import json
import os
os.chdir('/mnt/nvme0n1/liamz/StreamVICL-main/streamvicl/eval')
import json
import os

def visualize_coi_scores(json_path, sort_by="F1 Score", decimal_places=2):
    """
    将 JSON 文件 (类似 COI.json) 中的分数可视化为 LaTeX 表格格式。

    Args:
        json_path (str): JSON 文件的路径。
        sort_by (str): 用于对模型进行排序的指标 (例如, "F1 Score", "Accuracy")。
        decimal_places (int): 分数保留的小数位数。
    """
    try:
        # 以只读模式打开文件，使用 utf-8 编码（通常是 JSON 文件的良好实践）
        with open(json_path, 'r', encoding='utf-8') as f:
            datas = json.load(f) # 解析 JSON 数据
    except FileNotFoundError:
        print(f"错误：文件未找到于 {json_path}")
        return
    except json.JSONDecodeError as e:
        print(f"错误：解析 JSON 文件 {json_path} 失败: {e}")
        return
    except Exception as e:
        # 捕获其他可能的读取错误
        print(f"读取文件 {json_path} 时发生意外错误: {e}")
        return

    if not datas: # 检查解析后的数据是否为空
        print(f"文件 {json_path} 中没有数据。")
        return

    # 从第一个模型确定指标顺序 (假设所有模型都有相同的指标)
    # 或者，如果你希望固定顺序，可以定义一个列表
    first_model_key = next(iter(datas)) # 获取字典中的第一个键 (模型名)
    metrics_order = list(datas[first_model_key].keys()) # 获取该模型的所有指标作为列顺序

    # 如果你想要一个特定的指标顺序，取消注释并设置以下列表：
    # metrics_order = ["Accuracy", "Precision", "Recall", "F1 Score"]

    # 确保用于排序的指标 (sort_by) 是有效的
    if sort_by not in metrics_order:
        print(f"警告：排序指标 '{sort_by}' 未在数据中找到。将使用第一个指标 '{metrics_order[0]}' 进行排序。")
        sort_by = metrics_order[0]


    print(f"--- 来自 {os.path.basename(json_path)} 的分数 (按 {sort_by} 排序) ---")
    # LaTeX 表格的头部
    # 可以根据需要调整列宽，例如 'l' 代表左对齐, 'c' 居中, 'r' 右对齐
    # num_metrics = len(metrics_order)
    # print(f"\\begin{{tabular}}{{l{''.join(['c']*num_metrics)}}}") # LaTeX 表格环境示例
    # print("\\hline") # LaTeX 表格中的水平线

    # 构建并打印表头字符串，符合 LaTeX 格式
    header_string = 'Method\t\t& ' + '\t& '.join(metrics_order) + ' \\\\'
    print(header_string)
    print("\\hline") # LaTeX 中在表头后画一条横线

    # 根据指定的指标对模型进行排序
    try:
        # sorted_models 将是一个根据 sort_by 指标降序排列的模型名称列表
        # lambda 函数用于从每个模型的数据中提取排序键
        # .get(sort_by, float('-inf')) 用于安全地获取值，如果指标不存在或不是数字，则提供一个很小的值，使其排在最后
        sorted_models = sorted(
            datas.keys(), # 获取所有模型名称
            key=lambda model_name: (
                datas[model_name].get(sort_by, float('-inf')) # 获取排序指标的值
                if isinstance(datas[model_name].get(sort_by), (int, float)) # 确保是数字才比较
                else float('-inf') # 如果不是数字或不存在，则视为最小值
            ),
            reverse=True  # 降序排列 (假设分数越高越好)
        )
    except Exception as e:
        print(f"排序时发生错误: {e}。模型 '{sort_by}' 指标的数据可能不一致。")
        print("将以未排序或原始顺序继续。")
        sorted_models = list(datas.keys()) # 如果排序失败，则按原始顺序


    # 确定模型名称的最大长度，用于对齐打印（主要用于控制台输出美观）
    max_model_name_len = 0
    if sorted_models: # 确保列表不为空
        max_model_name_len = max(len(name) for name in sorted_models)
    # 也考虑表头 "模型名称" 的长度，以确保对齐
    max_model_name_len = max(max_model_name_len, len("模型名称"))


    # 打印每个模型的数据
    for model_name in sorted_models:
        model_data = datas[model_name] # 获取当前模型的所有数据
        scores_for_row = [] # 存储当前行要打印的所有分数
        for metric in metrics_order: # 按照预定的指标顺序提取分数
            score_val = model_data.get(metric) # 安全地获取指标值
            if isinstance(score_val, (int, float)): # 如果是数字
                # 格式化为指定的小数位数
                scores_for_row.append(f"{score_val:.{decimal_places}f}")
            else:
                # 处理非数字或缺失的值
                scores_for_row.append(str(score_val) if score_val is not None else "N/A")

        # 使用 f-string 的填充功能左对齐模型名称
        # '{model_name:<{max_model_name_len}}' 表示左对齐，宽度为 max_model_name_len
        print(f"{model_name:<{max_model_name_len}}\t& " + '\t& '.join(scores_for_row) + ' \\\\')

    # print("\\hline") # 表格末尾的横线 (可选)
    # print(f"\\end{{tabular}}") # LaTeX 表格环境结束 (可选)
    print("===================================================")



cor_json_path = '/mnt/nvme0n1/liamz/StreamVICL-main/streamvicl/eval/eval_data/CogReasoner_cor_score/CoR.json'
cor_json_path = '/mnt/nvme0n1/liamz/StreamVICL-main/streamvicl/eval/eval_data/CogReasoner1/CoR.json'
# Call the visualization function
print("\n--- Visualizing COI Scores (Sorted by F1 Score) ---")
visualize_coi_scores(cor_json_path, sort_by="F1 Score", decimal_places=2)

print("\n--- Visualizing COI Scores (Sorted by Accuracy, 3 decimal places) ---")
visualize_coi_scores(cor_json_path, sort_by="Accuracy", decimal_places=3)

print("\n--- Visualizing COI Scores (Attempting to sort by a non-existent metric) ---")
visualize_coi_scores(cor_json_path, sort_by="NonExistentMetric")
