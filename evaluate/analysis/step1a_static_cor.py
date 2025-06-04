import os,logging
import json
import glob
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
os.chdir('/mnt/nvme0n1/liamz/StreamVICL-main/streamvicl/eval/eval_data')

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式
    handlers=[
        logging.FileHandler("CogReasoner_cor_score/metrics_calculation.log"),  # 将日志保存到文件
    ]
)
logger = logging.getLogger(__name__)

def parse_cor(cor_str):
    """将字符串形式的 CoR 转换为列表"""
    if cor_str == "null" or cor_str == "[]":
        return []
    if isinstance(cor_str, list):
        return cor_str
    return json.loads(cor_str)

def calculate_metrics(all_true, all_pred):
    """计算 F1, Accuracy, Precision, Recall"""
    if len(all_true) == 0 or len(all_pred) == 0:
        return None  # 如果数据为空，返回 None

    f1 = f1_score(all_true, all_pred, average="binary")
    acc = accuracy_score(all_true, all_pred)
    pre = precision_score(all_true, all_pred, average="binary")
    rec = recall_score(all_true, all_pred, average="binary")

    return {  
        "Accuracy": round(acc,2),
        "Precision": round(pre,2),  
        "Recall": round(rec,2),
        "F1 Score": round(f1,2)
    }

if __name__ == "__main__":
    Local_Metrics = False
    # 输入路径
    # input_root = '/mnt/nvme0n1/liamz/StreamVICL-main/streamvicl/eval/eval_data/CogReasoner_cor_score'
    # output_path = '/mnt/nvme0n1/liamz/StreamVICL-main/streamvicl/eval/eval_data/CogReasoner_cor_score/CoR.json' 
    input_root = '/mnt/nvme0n1/liamz/StreamVICL-main/streamvicl/eval/eval_data/CogReasoner1'
    output_path = '/mnt/nvme0n1/liamz/StreamVICL-main/streamvicl/eval/eval_data/CogReasoner1/CoR.json'
    dataset_files = os.listdir(input_root)
    final_result = {}
    # error_files = ['InternVL2-8B_full','LongVA_full','MiniCPM-V-2_6_full','VideoLLaMA2-7B_full','VideoLLaMA3-7B_full']
    error_files = []
    for dataset_file in dataset_files:
        if 'full' not in dataset_file :continue # 只处理包含 'full' 的文件, 因为1.生成了cor; 2.cor_GT与QA对应
        print(f"\nProcessing dataset: {dataset_file}")  
        logging.info(f"\nProcessing dataset: {dataset_file}")  
        json_files = glob.glob(os.path.join(input_root, dataset_file, '*.json'))

        # 全局存储所有数据的 true 和 pred
        all_true_global = []
        all_pred_global = []

        # 遍历每个 JSON 文件
        for vid_id, file in enumerate(json_files):
            # logging.info(f"\nProcessing video {vid_id + 1}: {os.path.basename(file)}")
            with open(file, 'r') as f:
                vid_data = json.load(f)

            # 提取当前文件的 true 和 pred
            all_true_local = []
            all_pred_local = []
            for group in vid_data["Data"]:
                for item in group:
                    true_cor = parse_cor(item.get("coi", "[]"))  # 默认值为 "[]" key暂时仍为coi
                    pred_cor = parse_cor(item.get("predicted_coi", "[]"))  # 默认值为 "[]"
                    if dataset_file in error_files:
                        pred_cor = pred_cor[1:]+[0]
                    # 如果长度不一致，填充较短的列表
                    max_len = max(len(true_cor), len(pred_cor))
                    true_cor.extend([0] * (max_len - len(true_cor)))
                    pred_cor.extend([0] * (max_len - len(pred_cor)))

                    all_true_local.extend(true_cor)
                    all_pred_local.extend(pred_cor)

            # 将当前文件的数据合并到全局数据中
            all_true_global.extend(all_true_local)
            all_pred_global.extend(all_pred_local)
            if Local_Metrics:
                # 计算当前文件的指标
                metrics_local = calculate_metrics(all_true_local, all_pred_local)
                if metrics_local:
                    logging.info("Local Metrics:")
                    for key, value in metrics_local.items():
                        logging.info(f"{key}: {value:.4f}")
                else:
                    logging.info("No valid data to calculate local metrics.")

        # 计算全局指标
        metrics_global = calculate_metrics(all_true_global, all_pred_global)
        if metrics_global:
            logging.info("\nGlobal Metrics:")
            for key, value in metrics_global.items():
                logging.info(f"{key}: {value:.4f}")
            final_result.update({dataset_file:metrics_global})
        else:
            logging.info("\nNo valid data to calculate global metrics.")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=4)
        

        