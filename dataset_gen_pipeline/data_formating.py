import json
import os
from tqdm import tqdm
def write_json_segments(json_path, json_output):
    """
    将 JSON 数据写入文件。如果文件不存在，则创建新文件；如果文件存在，则追加数据。
    """
    if not os.path.exists(json_path):
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump([json_output], json_file, indent=4, ensure_ascii=False)
    else:
        with open(json_path, 'r+', encoding='utf-8') as json_file:
            json_file.seek(0)
            existing_data = json.load(json_file)

            if not isinstance(existing_data, list):
                raise ValueError("JSON 文件内容必须是一个列表！")

            existing_data.append(json_output)
            json_file.seek(0)
            json.dump(existing_data, json_file, indent=4, ensure_ascii=False)
            json_file.truncate()

def extract_and_transform(data, replace_seg_path = None):
    """
    从数据中提取 QA 对，并将其转换为所需的格式。
    """
    result = []
    for cur_qa_num, item in enumerate(data["Data"]):
        if replace_seg_path and isinstance(replace_seg_path,str):
            segment_path = item.get('segment_path',{}).replace('\\','/')
            # 查找 "segments" 的起始位置
            index = segment_path.find('/segments')
            if index != -1:  
            # 构建新的路径
                new_path = replace_seg_path + segment_path[index:]
                item['segment_path'] = new_path
            else:
                print(f"未找到 'segments' 在路径中: {segment_path}")
        qa_pairs = item.get("QA_pairs", {})
        timestamp = item.get("segment_timestamp", [])
        event_time = item.get("event_timestamp", [])
        coi = item.get("coi_qa_info", None)
        id = item.get("qa_info",None)
        is_visual = item.get("is_visual",True)
        coi_vector = json.loads(coi)
        coi_vector = [1 if i in coi_vector else 0 for i in range(cur_qa_num)]
        if qa_pairs and timestamp:
            for q_key, q_value in qa_pairs.items():
                if q_key.startswith("Q"):
                    a_key = "A" + q_key[1:]
                    a_value = qa_pairs.get(a_key, "")
                    result.append({
                        "Q": q_value,
                        "A": a_value,
                        "T": timestamp,
                        "info" : {
                            "is_visual": is_visual,
                            "Event_Time": event_time,
                            "ID":id,
                            "COI":coi,
                            "relevance": str(coi_vector)
                        }
                    })
            del item['qa_info']
            del item['coi_qa_info']
            item.update({
                "ID":id,
                "COI":coi,
                "relevance": str(coi_vector)
            })

    return result
def format_vqa_dataset(root_dir, replace_seg_path = None):
    """
    处理 VQA 数据集，读取输入目录中的 JSON 文件，提取和转换数据，并将结果写入输出目录。
    """
    input_dir = os.path.join(root_dir, 'COG_Dataset_raw')
    output_dir = os.path.join(root_dir, 'COG_Dataset_simply')
    output_dir3 = os.path.join(root_dir, 'COG_Dataset_detailed')

    os.makedirs(output_dir3, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    video_names = [os.path.splitext(x)[0] for x in os.listdir(input_dir)]

    for video_name in tqdm(video_names,desc='处理VQA数据集，生成QA数据集'):
        json_path1 = os.path.join(input_dir, video_name) + '.json'
        json_path2 = os.path.join(output_dir, video_name) + '.json'
        json_path3 = os.path.join(output_dir3, video_name) + '.json'
        if os.path.exists(json_path2):
            os.remove(json_path2)
        with open(json_path1, 'r') as f:
            datasets = json.load(f)
            for dataset in datasets:
                transformed_data = extract_and_transform(dataset,replace_seg_path)
                write_json_segments(json_path3, dataset)
                write_json_segments(json_path2, transformed_data)
    format_vqa_dataset_(root_dir)

def format_vqa_dataset_(root_dir):
    """
    处理 VQA 数据集，读取输入目录中的 JSON 文件，提取和转换数据，并将结果写入输出目录。
    """
    input_dir = os.path.join(root_dir, 'COG_Dataset_raw')
    output_dir = root_dir
    os.makedirs(output_dir, exist_ok=True)
    json_path2 = os.path.join(output_dir, 'COG_streamv_dataset') + '.json'

    video_names = [os.path.splitext(x)[0] for x in os.listdir(input_dir)]
    json_output = []
    for video_name in tqdm(video_names,desc='合并QA数据集'):
        json_path1 = os.path.join(input_dir, video_name) + '.json'
        transformed_data = []
        with open(json_path1, 'r') as f:
            datasets = json.load(f)
            for dataset in datasets:
                transformed_data.append(extract_and_transform(dataset))
                # write_json_segments(json_path2, transformed_data)
        json_output.append({
                'video_name': video_name,
                'data':transformed_data
            })
    with open(json_path2, 'w', encoding='utf-8') as json_file:
        json.dump(json_output, json_file, indent=4, ensure_ascii=True)

if __name__ == "__main__":
    format_vqa_dataset('')
