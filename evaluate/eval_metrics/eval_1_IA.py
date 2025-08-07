
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

        if (qa_id + 1) % 5 != 0 and qa_id != len(seq_data): continue

        prompt1 = '''
        You are a rigorous language evaluation expert, skilled in assessing the **Informational Accuracy** of generated answers in video question-answering tasks. 

        ### Task Description
        - Your task is to compare the predicted answer with the correct answer based on a holistic understanding of the question and answer, and assessing it's information consistency with the correct answer to effectively address the question. Strictly follow the evaluation criteria below:
        ### Scoring Criteria (10-point scale):
        - 0-1: Completely incorrect, contradictory to the correct answer; fails to answer.
        - 2-3: Some correct information but mixed with serious errors or fabrications; largely fails.
        - 4-6: Key information correct but with errors, omissions, or vagueness; partially answers.
        - 7-8: Mostly correct, covers main points but with minor errors or omissions; effectively answers.
        - 9-10: Fully matches correct answer, includes all key elements; correctly and sufficiently answers.'''
        prompt2 = (f'''
        ### Input:
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

