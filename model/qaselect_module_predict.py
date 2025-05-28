import torch
import re
from transformers import AutoTokenizer, AutoModel, LogitsProcessor, LogitsProcessorList

def format_example(example, include_demo=True):
    """
    Convert data sample to formatted prompt with instructions to output a list of indices
    of historical QA pairs that are helpful for answering the current question, prefixed
    with yes (requires additional visual info) or no (fully answerable by historical QA).
    """
    system_prompt = """<|im_start|>system
You are a QA-pair filtering assistant. Your task is to identify which of the historical QA pairs are helpful for answering the current question and determine if the historical QA pairs alone are sufficient to answer it.

A QA pair is considered helpful if it provides:
- Relevant background information, context, or details
- Additional facts or insights that can be used to answer the current question
- Matching roles, scenarios, or domain knowledge that could support the answer

Output a single bracketed sequence:
- Start with 'yes' if the historical QA pairs are insufficient to fully answer the question (additional visual information may be needed).
- Start with 'no' if the current question can be fully answered using only the historical QA pairs (no additional visual information needed).
- Follow with the indices (starting from 0) of the helpful QA pairs, e.g., [yes,0,5] or [no,0,5].
- If no QA pairs are helpful, output [yes] or [no] based on the question's dependency.
- Do not add extra text or explanation — only output the bracketed sequence.
<|im_end|>"""

    demo_section = ""
    if include_demo:
        demo_section = """\nExample:
Current Question: What causes earthquakes?
Historical QA Pairs:
0. Q: How to measure earthquakes? A: Using the Richter scale
1. Q: What is tectonic plate? A: Massive rock slabs beneath crust
2. Q: What is the weather like today? A: Sunny and warm
→ Output: [no,1]
------------------------------
Example:
Current Question: What does an earthquake look like?
Historical QA Pairs:
0. Q: How to measure earthquakes? A: Using the Richter scale
1. Q: What is tectonic plate? A: Massive rock slabs beneath crust
2. Q: What is the weather like today? A: Sunny and warm
→ Output: [yes]
------------------------------"""

    user_content = f"""{demo_section}
Current Question: {example['current_Q']}

Historical QA Pairs (ordered by time):"""

    for i, (q, a) in enumerate(zip(example['hist_Qs'], example['hist_As'])):
        user_content += f"\n{i}. Q: {q}\n   A: {a}"

    user_content += "\nGenerate a bracketed sequence (e.g., [yes,0,5] or [no,0,5]) indicating the dependency (yes or no) and the indices of helpful QA pairs. Only output the bracketed sequence."

    full_text = (
        f"{system_prompt}"
        f"<|im_start|>user\n{user_content}<|im_end|>"
        f"<|im_start|>assistant\n"
    )
    return full_text

def select_qas(
    current_question: str,
    hist_Qs: list,
    hist_As: list,
    base_model,
    tokenizer = None,
    include_demo: bool = True,
):
    inference_example = {
        "current_Q": current_question,
        "hist_Qs": hist_Qs,
        "hist_As": hist_As,
    }
    prompt_text = format_example(inference_example, include_demo=include_demo)


    model = base_model
    if tokenizer is None:
        raise ValueError("If passing a model instance, please provide a tokenizer as well.")

    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k,v in inputs.items()}

    class StructuredLogitsProcessor(LogitsProcessor):
        def __init__(self, tokenizer):
            self.allowed_token_ids = self._get_allowed_token_ids(tokenizer)

        def _get_allowed_token_ids(self, tokenizer):
            special_tokens = [str(i) for i in range(10)] + ['[', ']', ',', '<|im_end|>', 'no', 'yes']
            allowed_ids = set()
            for t in special_tokens:
                token_ids = tokenizer.encode(t, add_special_tokens=False)
                for idx in token_ids:
                    if idx >= 0:
                        allowed_ids.add(idx)
            return sorted(list(allowed_ids))

        def __call__(self, input_ids, scores):
            mask = torch.full_like(scores, float('-inf'))
            mask[..., self.allowed_token_ids] = 0
            return scores + mask

    logits_processor = LogitsProcessorList([StructuredLogitsProcessor(tokenizer)])
    
    with torch.no_grad():
        generation_output = model.generate_language_module(
            **inputs,
            max_new_tokens=50,
            num_beams=1,
            logits_processor=logits_processor,
            do_sample=False,
            eos_token_id=151645
        )
    generated_ids = generation_output[0]
    prompt_len = inputs["input_ids"].shape[1]
    gen_tokens = generated_ids[prompt_len:]
    output_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    output_text = output_text.strip()
    if output_text == "":
        output_text = '[yes]'
    if not output_text.endswith(']'):
        output_text += ']'
    if not output_text.startswith('['):
        output_text = '[' + output_text
    return output_text


