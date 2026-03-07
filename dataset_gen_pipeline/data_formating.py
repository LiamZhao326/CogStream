import json
import os
from tqdm import tqdm
def write_json_segments(json_path, json_output):
    """
    Write JSON data to file. If the file does not exist, create a new file; if the file exists, append data.
    """
    if not os.path.exists(json_path):
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump([json_output], json_file, indent=4, ensure_ascii=False)
    else:
        with open(json_path, 'r+', encoding='utf-8') as json_file:
            json_file.seek(0)
            existing_data = json.load(json_file)

            if not isinstance(existing_data, list):
                raise ValueError("The content of the JSON file must be a list!")

            existing_data.append(json_output)
            json_file.seek(0)
            json.dump(existing_data, json_file, indent=4, ensure_ascii=False)
            json_file.truncate()

def extract_and_transform(data, replace_seg_path = None):
    """
    Extract QA pairs from data and convert them to the required format.
    """
    result = []
    for cur_qa_num, item in enumerate(data["Data"]):
        if replace_seg_path and isinstance(replace_seg_path,str):
            segment_path = item.get('segment_path',{}).replace('\\','/')
            # Find the starting position of "segments"
            index = segment_path.find('/segments')
            if index != -1:  
            # Build a new path
                new_path = replace_seg_path + segment_path[index:]
                item['segment_path'] = new_path
            else:
                print(f"The 'segments' is not found in the path: {segment_path}")
        qa_pairs = item.get("QA_pairs", {})
        timestamp = item.get("segment_timestamp", [])
        event_time = item.get("event_timestamp", [])
        cor = item.get("cor_qa_info", None)
        id = item.get("qa_info",None)
        is_visual = item.get("is_visual",True)
        cor_vector = json.loads(cor)
        cor_vector = [1 if i in cor_vector else 0 for i in range(cur_qa_num)]
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
                            "COR":cor,
                            "relevance": str(cor_vector)
                        }
                    })
            del item['qa_info']
            del item['cor_qa_info']
            item.update({
                "ID":id,
                "COR":cor,
                "relevance": str(cor_vector)
            })

    return result
def format_vqa_dataset(root_dir, replace_seg_path = None):
    """
    Process VQA dataset, read JSON files in input directory, extract and transform data, and write results to output directory.
    Args:
        root_dir: The root directory of the dataset.
        replace_seg_path: The path to replace the segment_path.
    """
    input_dir = os.path.join(root_dir, 'CogStream_raw_dataset')
    qa_output_dir = os.path.join(root_dir, 'QA_Dataset')
    vqa_output_dir = os.path.join(root_dir, 'VQA_Dataset')

    os.makedirs(vqa_output_dir, exist_ok=True)
    os.makedirs(qa_output_dir, exist_ok=True)

    video_names = [os.path.splitext(x)[0] for x in os.listdir(input_dir)]

    for video_name in tqdm(video_names,desc='Process VQA dataset, generate QA dataset'):
        json_path_raw = os.path.join(input_dir, video_name) + '.json'
        json_path_qa = os.path.join(qa_output_dir, video_name) + '.json'
        json_path_vqa = os.path.join(vqa_output_dir, video_name) + '.json'
        if os.path.exists(json_path_qa):
            os.remove(json_path_qa)
        with open(json_path_raw, 'r') as f:
            datasets = json.load(f)
            for dataset in datasets:
                transformed_data = extract_and_transform(dataset,replace_seg_path)
                write_json_segments(json_path_vqa, dataset)
                write_json_segments(json_path_qa, transformed_data)
    format_vqa_dataset_(root_dir)

def format_vqa_dataset_(root_dir):
    """
    Process VQA dataset, read JSON files in input directory, extract and transform data, and write results to output directory.
    """
    input_dir = os.path.join(root_dir, 'CogStream_raw_dataset')
    output_dir = root_dir
    os.makedirs(output_dir, exist_ok=True)
    json_path2 = os.path.join(output_dir, 'COG_streamv_dataset') + '.json'

    video_names = [os.path.splitext(x)[0] for x in os.listdir(input_dir)]
    json_output = []
    for video_name in tqdm(video_names,desc='Merge QA dataset'):
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
