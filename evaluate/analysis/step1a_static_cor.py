import os,logging
import json
import glob
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(levelname)s - %(message)s",  
    handlers=[
        logging.FileHandler("CogStream/metrics_calculation.log"), 
    ]
)
logger = logging.getLogger(__name__)

def parse_cor(cor_str):
    if cor_str == "null" or cor_str == "[]":
        return []
    if isinstance(cor_str, list):
        return cor_str
    return json.loads(cor_str)

def calculate_metrics(all_true, all_pred):
    if len(all_true) == 0 or len(all_pred) == 0:
        return None  

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
    input_root = 'CogStream'
    output_path = 'CogStream/RQAS.json'
    dataset_files = os.listdir(input_root)
    final_result = {}
    error_files = []
    for dataset_file in dataset_files:
        if 'full' not in dataset_file :continue 
        print(f"\nProcessing dataset: {dataset_file}")  
        logging.info(f"\nProcessing dataset: {dataset_file}")  
        json_files = glob.glob(os.path.join(input_root, dataset_file, '*.json'))

        all_true_global = []
        all_pred_global = []

        for vid_id, file in enumerate(json_files):
            # logging.info(f"\nProcessing video {vid_id + 1}: {os.path.basename(file)}")
            with open(file, 'r') as f:
                vid_data = json.load(f)

            all_true_local = []
            all_pred_local = []
            for group in vid_data["Data"]:
                for item in group:
                    true_cor = parse_cor(item.get("coi", "[]")) 
                    pred_cor = parse_cor(item.get("predicted_coi", "[]"))
                    if dataset_file in error_files:
                        pred_cor = pred_cor[1:]+[0]

                    max_len = max(len(true_cor), len(pred_cor))
                    true_cor.extend([0] * (max_len - len(true_cor)))
                    pred_cor.extend([0] * (max_len - len(pred_cor)))

                    all_true_local.extend(true_cor)
                    all_pred_local.extend(pred_cor)

            all_true_global.extend(all_true_local)
            all_pred_global.extend(all_pred_local)
            if Local_Metrics:
                metrics_local = calculate_metrics(all_true_local, all_pred_local)
                if metrics_local:
                    logging.info("Local Metrics:")
                    for key, value in metrics_local.items():
                        logging.info(f"{key}: {value:.4f}")
                else:
                    logging.info("No valid data to calculate local metrics.")

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
        

        
