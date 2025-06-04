
import json,os,argparse,glob,re,sys
from tqdm import tqdm
sys.path.append('/mnt/nvme0n1/liamz/StreamVICL-main/streamvicl/eval')
os.chdir('/mnt/nvme0n1/liamz/StreamVICL-main/streamvicl/eval')
from utils.Deepseekv3 import Deepseek_sf,Deepseek_sf_
from utils.GPT import GPT_OpenAi

'''
基于LVLM进行评估(gpt4o/deepseekv3)

中文释义：
————————————
你是一个视频问答评估专家，专注于分析生成答案的视频细节完整性。
你的任务是在整体理解问题和视频内容的基础上，将预测答案与正确答案进行对比，
判断其是否严格基于视频内容，全面覆盖视频细节，同时避免引入无关或错误的常识推理。请严格遵循以下评估标准：

Scoring Criteria (10分制)：
- 0-1分：完全脱离视频，充斥无关或错误推论，完全无法匹配答案
- 2-3分：少量视频细节，内容不完整，较多无关或错误的推论
- 4-6分：包含部分视频要素，存在一定常识性推论，略微脱离视频内容
- 7-8分：视频要素基本全面，仅少量细节缺失，推论基本符合视频内容
- 9-10分：与答案完全匹配，涵盖所有必要视频细节，无无关推论

请遵循以下步骤进行评估：  
1. 充分理解：确保彻底理解问题，明确问题的背景和核心要求
2. 精确比较：将模型的预测答案与真实答案进行精确对比，关注信息的相关性和正确性。  
3. 客观评分：请仅基于问题-答案信息，来对预测答案评分，不要受到任何先验知识、或常识假设的影响
答的准确性。  

输出要求
"用Python字典格式返回且仅包含'score'键值对，值必须为整数类型。"
"禁止任何额外文字说明，示例格式：{'score': 5}"
————————————
'''
def extract_scores(s):

    result = {}
    try:
        s_dict=json.loads(s)
        for k,v in s_dict:
            v = v.strip('"')
            key_ = f"PredA{int(key[-1])}"
            result[key_] = int(float(v))
        return result
    except Exception as e:
        pass

    pattern = r'"?(?P<key>\w+)"?\s*:\s*["\']?(?P<value>-?\d+\.?\d*)["\']?'
    matches = re.finditer(pattern, s)
    
    result = {}
    for match in matches:
        key = match.group('key')
        value_str = match.group('value')
        # 转换数值类型
        try:
            value = int(float(value_str))
        except ValueError:
            value = value_str  # 非数字保留原始字符串
        key_ = f"PredA{int(key[-1])}"
        result[key_] = value
    return result

def annotate(seq_data,file,seq_id):
    """
    Evaluates question and answer pairs using GPT-4
    """
    result = {}
    questions,answers,preds = "","",""
    for qa_id, single_qa in enumerate(seq_data):
        questions += f"\nQ{qa_id}: {single_qa['question']}"
        answers+=f"\nA{qa_id}: {single_qa['answer']}"
        preds+=f"\nPredA{qa_id}: {single_qa['prediction']}"
        coi = json.loads(single_qa['coi'])
        if qa_id>3:
            break
        coi_qas = ''
        if qa_id:
            count = 0
            for idx, bi_label in enumerate(coi):
                if bi_label==1:
                    count += 1
                    coi_qas +=f"\nQA{count}: {seq_data[idx]['question']} {seq_data[idx]['answer']}"
        if (qa_id + 1) % 1 != 0 and qa_id != len(seq_data): continue

        prompt1 = '''
You are a video QA evaluation expert specializing in assessing the **Detail Completeness** of predicted answers.

### Task Description
 - Your task is to compare the predicted answer with the correct answer based on a holistic understanding of the question, correct answer and video content. Evaluate whether the predicted answer strictly adheres to the video content, fully covers video details, and avoids introducing irrelevant or incorrect commonsense reasoning. Strictly follow the evaluation criteria below:

### Scoring Criteria (10-point scale):
- 0-1: Completely detached from the video, filled with irrelevant or incorrect reasoning, and entirely mismatched with the correct answer. 
- 2-3: Contains few video details, incomplete content, and includes multiple irrelevant or incorrect inferences.  
- 4-6: Includes some video elements but contains certain commonsense reasoning, slightly deviating from the video content.  
- 7-8: Mostly comprehensive video elements with only minor detail omissions; inferences largely align with the video content.  
- 9-10: Fully matches the correct answer, covers all necessary video details, and includes no irrelevant inferences. '''
        prompt2 =(f'''
### Input:
- Video preceding context:{coi_qas}
- Question: {questions}
- Correct Answer: {answers}
Please evaluate the **Detail Completeness Score** for each predicted answers:
- Predicted Answers: {preds}
'''
'''
### Evaluation Requirements:
1. Thorough Comprehension: Ensure a thorough understanding of the question, clarifying the context and core requirements.
2. Unbiased Judgment: Scoring the predicted answer based solely on the question-answer information, without being influenced by any prior knowledge or common-sense assumptions.
3. Precise Comparison: Precisely compare the model's predicted answer with the true answer, focusing on the relevance and correctness of the information.

### Output Requirements:
- Return in Python dictionary format containing only the 'score' key-value pair, with the value being an integer type.
- Prohibit any additional textual explanations, example JSON format: 
{
    "PredA1":score,
}''')
        messages=[
                    {
                        "role": "system",
                        "content": prompt1
                    },
                    {
                        "role": "user",
                        "content": prompt2
                    }
                ]
        while True:
            try:
                if (response := client.chat(messages)) is not None:
                    output = extract_scores(response)
                    result.update(output)
                break
            except Exception as e:
                print(f"Error processing file '{file}'-seq{seq_id}-id{qa_id}: {e}")
        questions,answers,preds = "","",""
    return result
def main(args):
    # 处理所有视频对应的JSON文件
    json_files = glob.glob(os.path.join(args.input_root, '*.json'))
    for data_id,file in enumerate(json_files):
        video_name = os.path.basename(file).split('.')[0] # qa_data['video_name']
        # raw_data_path = os.path.join(args.raw_data_root,f"{video_name}.json")
        # if not os.path.exists(raw_data_path):
        #     print(f'Fail to find the raw data of{file}')
        #     continue
        # with open(raw_data_path, 'r') as f:
        #     raw_data = json.load(f)
        output_path = os.path.join(args.output_root,f"{video_name}.json")
        if os.path.exists(output_path):
            print(f"Video_data {os.path.basename(output_path)} already assessed")
            continue
        with open(file, 'r') as f:
            qa_data = json.load(f)
        seq_results = []
        # 遍历视频数据中的每个序列
        for seq_id, seq_data in tqdm(enumerate(qa_data['Data']),
                                     desc=f'Processing video {video_name} ({data_id}/{len(json_files)}'):
            seq_results.append(annotate(seq_data,file,seq_id))
                # 保存结果（根据实际需求调整保存逻辑）
        output_resluts = {
            "video_name": qa_data['video_name'],
            "score": seq_results
        }
        with open(output_path, 'w') as f:
            json.dump(output_resluts, f, indent=4)


def create_client(model_name):
    if 'gpt' not in model_name:
        return Deepseek_sf_(model = model_name)
    else:
        return GPT_OpenAi(model = model_name)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, required=False,default='/mnt/nvme0n1/liamz/StreamVICL-main/streamvicl/eval/Streamvicl/input', #  Streamvicl/qa_dataset
                        help="包含QA数据的JSON目录")
    parser.add_argument("--raw_data_root", type=str, required=False,default='/mnt/sda/Streamvicl/Test_dataset/VQA_Dataset', #  Streamvicl/qa_dataset
                        help="包含QA数据的JSON目录")
    parser.add_argument("--output_root", type=str, default="/mnt/nvme0n1/liamz/StreamVICL-main/streamvicl/eval/Streamvicl/DC",
                        help="输出结果目录")
    parser.add_argument("--model", type=str, default="Pro/deepseek-ai/DeepSeek-V3",
                        help="模型路径")
    return parser.parse_args()

    
if __name__ == "__main__":
    args = parse_args()
    client = create_client(args.model)
    # 创建输出目录
    os.makedirs(args.output_root, exist_ok=True)
    main(args)
