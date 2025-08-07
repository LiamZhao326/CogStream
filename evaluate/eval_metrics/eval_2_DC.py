
import json,os,argparse,glob,re,sys
from tqdm import tqdm
from utils.Deepseekv3 import Deepseek_sf,Deepseek_sf_
from utils.GPT import GPT_OpenAi

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
        try:
            value = int(float(value_str))
        except ValueError:
            value = value_str
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

        for seq_id, seq_data in tqdm(enumerate(qa_data['Data']),
                                     desc=f'Processing video {video_name} ({data_id}/{len(json_files)}'):
            seq_results.append(annotate(seq_data,file,seq_id))

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
    parser.add_argument("--input_root", type=str, required=False, default='', 
                        help="Directory containing QA data JSON files")
    parser.add_argument("--output_root", type=str, default="", 
                        help="Directory for output results")
    parser.add_argument("--model", type=str, default="",
                        help="Model path")
    return parser.parse_args()


    
if __name__ == "__main__":
    args = parse_args()
    client = create_client(args.model)
    os.makedirs(args.output_root, exist_ok=True)
    main(args)
