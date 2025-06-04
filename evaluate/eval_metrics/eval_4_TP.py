
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
你的任务是将预测的答案与正确的答案进行比较，并评估预测的答案是否正确地识别了问题所针对的核心事件，同时参考了视频上下文中精确对齐的信息。
请严格遵循以下评估标准：

Scoring Criteria (10分制)：
- 0-1分：与问题指向的事件毫无关联，引用错误上下文信息，时间描述完全错位
- 2-3分：与事件部分相关，但引用大量无关/错误信息，时间描述严重不准确
- 4-6分：抓住了一般事件，但未引用关键细节，有模糊的时间或时序表述
- 7-8分：基本准确的指向事件，与正确答案保持时序一致性，仅少量非核心时间信息缺失
- 9-10分：完全匹配事件，有精确的时间引用（时间戳或时序描述）

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
        if qa_id:
            count = 0
            coi_qas = ''
            for idx, bi_label in enumerate(coi):
                if bi_label==1:
                    count += 1
                    coi_qas +=f"\nQA{count}: {seq_data[idx]['question']} {seq_data[idx]['answer']}"
        if (qa_id + 1) % 1 != 0 and qa_id != len(seq_data): continue

        prompt1 = '''
        You are a video QA evaluation expert specializing in assessing the **Temporal Precision** of generated answers.
        
        ### Task Description
        - Your task is to compare the predicted answer with the correct answer and evaluate whether the predicted answer correctly identifies the core event the question is targeting, meanwhile references precisely aligned information from the video context. Strictly follow the evaluation criteria below:
        
        ### Scoring Criteria (10-point scale):
        - 0-1: Completely unrelated to the target event, citing incorrect context and showing severe temporal misalignment.  
        - 2-3: Somewhat related to the event but includes irrelevant or incorrect information, with serious inaccuracies in temporal descriptions.  
        - 4-6: Captures the main event but lacks key details, using vague temporal or sequential references.  
        - 7-8: Mostly precise in targeting the event, maintaining temporal consistency with only minor non-core omissions.  
        - 9-10: Fully aligns with the event, and includes precise temporal references (e.g., timestamps or sequential descriptions).'''
        
        prompt2 =f'''
### Input:
'''
        if qa_id and coi_qas:
            prompt2 += f'''- Video preceding context:{coi_qas}'''

        prompt2 += (f'''
Please evaluate the following video-based question-answer pair:
- Questions: {questions}
- Correct Answers: {answers}
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
    "PredA2":score,
    ...
}
''')
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
    parser.add_argument("--output_root", type=str, default="/mnt/nvme0n1/liamz/StreamVICL-main/streamvicl/eval/Streamvicl/TP",
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
