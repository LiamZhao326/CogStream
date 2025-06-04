# 分词器存储路径：'anaconda3/envs/clip_vid/share/nltk_data'
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score

'''
Basic metrics: BLEU-4, METEOR, ROUGE, CIDEr
'''

def evaluate_vqa(references, candidate):
    """
    评估VQA输出与参考答案的相似性
    输入格式：
    references : 参考答案列表，例如 ["gt_A1"]
    candidate  : 生成答案字符串，例如 "A1"
    """
    # 小写
    refs_lower = [ref.lower() for ref in references]
    candidate_lower = candidate.lower()

    # 构建数据
    ref_tokens = [word_tokenize(ref) for ref in refs_lower]
    cand_tokens = word_tokenize(candidate_lower)

    # BLEU-4计算
    bleu_score = sentence_bleu(
        ref_tokens, 
        cand_tokens, 
        weights=(0.50, 0.50)
    )
    # METEOR 计算
    meteor_score_val = meteor_score(
        ref_tokens, 
        cand_tokens, 
    )

    # 构建数据
    refs_formatted = {'test_case': references}
    cands_formatted = {'test_case': [candidate]}

    # ROUGE-L计算
    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(refs_formatted, cands_formatted)

    return {
        'BLEU-4': bleu_score,
        'METEOR': meteor_score_val,
        'ROUGE-L': rouge_score
    }

if __name__ == "__main__":
    Local_Metrics = False
    # 示例数据
    example_data = [
        {
            "Q": "What is the capital of France?",
            "A1": "The capital of France is Paris.",
            "A2": "Paris is the capital of France."
        },
        {
            "Q": "Who wrote 'Pride and Prejudice'?",
            "A1": "Jane Austen wrote 'Pride and Prejudice'.",
            "A2": "'Pride and Prejudice' was written by Jane Austen."
        },
        {
            "Q": "What is the color of the sky?",
            "A1": "The sky is blue.",
            "A2": "Blue is the color of the sky."
        },
        {
            "Q": "What is the color of the sky?",
            "A1": "The sky is blue.",
            "A2": "The sky is blue."
        }
    ]
    # CIDEr计算
    refs_formatted = {str(key):[val['A2']] for key, val in enumerate(example_data)}
    cands_formatted = {str(key):[val['A1'] ]for key, val in enumerate(example_data)}
    cider_scorer = Cider()
    mean_cider_score, cider_score = cider_scorer.compute_score(refs_formatted, cands_formatted)

    # 其余指标计算
    bleu_score,meteor_score_val,rouge_score = [],[],[]

    # 对每个样本进行评估
    for i, data in enumerate(example_data):
        scores = evaluate_vqa([data['A2']], data['A1'])
        bleu_score.append(scores['BLEU-4'])
        meteor_score_val.append(scores['METEOR'])
        rouge_score.append(scores['ROUGE-L'])

        #打印单样本评估结果
        if Local_Metrics:
            print(f"\nSample {i+1}:")
            print(f"Q: {data['Q']}")
            print(f"A1 (Model Output): {data['A1']}")
            print(f"A2 (Ground Truth): {data['A2']}")
            print("Evaluation Scores:")
            scores.update({
                "CIDEr":cider_score[i]/10
            })
            for metric, value in scores.items():
                print(f"{metric}: {value:.4f}")
        
    print("\nEvaluation Scores:")
    final_score = {
        'BLEU-4': sum(bleu_score)/len(bleu_score),
        'METEOR': sum(meteor_score_val)/len(meteor_score_val),
        'ROUGE-L': sum(rouge_score)/len(rouge_score),
        'CIDEr': mean_cider_score/10  
    }
    for metric, value in final_score.items():
        print(f"{metric}: {value:.4f}")

