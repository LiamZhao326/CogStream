
def scoring_prompt(current, history=None):
    prompt = ('''
## Key Notes for Input Data
- PQA and CQA are from the same video, with CQA occurring chronologically later.  
- PQA may provide foundational context or reasoning steps critical for answering CQA.  

## Scoring Criteria  
**0-1: Irrelevant/Minimal Dependence**  
PQA provides no relevant information or logical connection to CQA, offering no real help.  
Typical Scenarios:  
- Completely unrelated topics. 
- PQA background irrelevant to CQA.  

**2-3: Slightly Helpful**  
PQA provides weak or indirect connections but lacks substantial value for answering CQA.  
Typical Scenarios:  
- The information is tangentially related to CQA but insufficient to form a clear answer or narrow down possibilities.  

**4-5: Moderately Helpful**  
PQA provides necessary context (e.g., names, background events) or partial reasoning steps, but requires additional information to fully answer CQA.  
Typical Scenarios:  
- Indicates a temporal sequence without precise timing.  
- Logical related, offers a causal description with indirect logical relevance to CQA’s answer.  
- Helps eliminate irrelevant options or narrows possibilities but does not directly resolve CQA.  

**6-7: Highly Helpful**  
PQA directly provides key information or critical reasoning steps required for CQA, enabling part or all of CQA’s answer to be inferred.  
Typical Scenarios:  
- Explicit temporal sequence or prediction that directly resolves CQA.  
- Strongly logical related, complete causal explanation with evidence, directly supporting CQA.  
- Provides reasoning steps or information allowing CQA’s answer to be directly deduced.  

## Analysis Process  
1. Identify critical elements in CQA’s answer (e.g., specific terms, numbers, causal links).  
2. Check if these elements appear in PQA or require PQA’s logic to derive.  

## Requirements
- For each CQA, compare it with every PQA independently. Process each pair in this sequence: "for cqa in CQA: {for pqa in PQAs: {}}".  
- Evaluate each PQA in its entirety for its contribution to CQA.  
- Avoid subjective assumptions; base analysis solely on given content.  
- Do NOT include text outside the valid JSON output.
'''
                    '''
### Output Format:
Present the scores in JSON format:
```json
{
    "C1": {
        "P1": 7,
        ...},
    "C2": {...},
    ...
    "Cn": {...}
}```'''
                    f'''
        ### Inputs
- Current QA pairs: {current}

- Previous QA pairs: {history}''')
    return prompt

def polish(qa_pairs,Global=False, CR=False):
    # gpt_client0 = Deepseek()
    # 优化QA内容
    format = """{
  "L3": {
    "Q1": "[label]...",
    "A1": "",
    "Q2": "[label]...",
    "A2": "",
    ...
  }
}"""
    if Global:
        format = """{  
    "Q1": "[label]...",
    "A1": "",
    "Q2": "[label]...",
    "A2": "",
    ...
}"""
    elif CR:
        format = """{
"L4": {
    "Q1": "[label]...",
    "A1": "",
    "Q2": "[label]...",
    "A2": ""
  }
}"""
    prompt1 = f'''
    ### **Task Description**: Polish the following Q&A pairs to meet strict quality standards.
    **Requirements**:
    1. **Question Formulation**:
       - Must not contain hints, clues, or information that could reveal the answer.
       - Preserve all labels within [ ] exactly.

    2. **Answer Formulation**:
       - Replace media-specific terms (e.g., "frame/image/segment/clip") with neutral alternatives.

    3. Do not modify the original text unless it is necessary.

    **Examples**:
    - Original Q: "[Reasoning] Why are these firefighters at the scene of a wildfire?"  
      → Polished Q: "[Reasoning] Why are these firemen here?"  
      Reason: The original question includes excessive background information (wildfire), allowing the answer to be deduced directly without understanding the context.
    - Original Q: "[Analysis] Given that the eggs are at room temperature and flour is added gradually, how do these factors affect the cake's texture?"  
      → Polished Q: "[Analysis] How do eggs and flour affect the cake's texture?"  
      Reason: The original question contained specific conditions (room-temperature eggs, gradual addition of flour), which guided towards the answer.
    - Original A: "That's probably because he wanted to drink water in clip four."  
      → Polished A: "That's probably because he wanted to drink water."  

    **Output Format (JSON)**: {format}
    **Input Q&A Pairs**: {qa_pairs}'''
    return prompt1

def creat_prompt(history=None, qa_pairs=None, summarize=False, Global=None, time=None,first=False):
    if first:
        format = '''
        {
          "title": "",
          "QA_pairs": {
            "L1": {
              "Q1": "[label]...",
              "A1": "",
              ...
            },
            "L2": {
              "Q1": "[Co-reference](QA_id)...",
              "A1": "",
              ...
            }
          }
        }
                '''
        prompt = f'''
# Task Description
You are an expert in video comprehension, specializing in visual and contextual analysis to generate accurate and meaningful question-answer (Q&A) pairs. Follow these steps to complete the task:

## Step 1: Generate a Clip Title
- Analyze the current video frames as a continuous clip, combined with the previous titles and context, to understand the full content.
- Create a concise, accurate title that reflects the current clip’s main focus or event.

##  Step 2: Generate Q&A Pairs
- Each question should be written in natural language, prefixed by a label in square brackets.

### Level 1: Video Understanding Q&A pairs(Only 4 Q&A pairs)
**Optional Labels**: 
- [Temporal Perception]: Sequential questions with “before/after"; Time-based questions,like "What happened at 18s?"; Determine the exact time point or time range when the event occurred.
- [Attributes]: Covers features of objects (color, size, position,environment， etc.).
- [Actions]: Refers to observable behaviors/movements. 
- [Items]: Relates to visible objects/items in the scene.

### Level 2: Co-reference Q&A pairs(Only 2 Q&A pairs)
**Features**:  
- Detailed questions about an object or person from Level 1, using pronouns (he/she/it/they) for reference.
- Each pair must reference to a specific Level 1 Q&A pair (e.g., QA1, QA2). Pronouns must be clear only when paired with the Level 1 Q&A.
- Answers must remain based on the current clip.
**Format:** Begin each question with the format: "[Co-reference](QA_id)...?" to indicate the linked pair.
**Example:** "[Co-reference](QA1) How did it break?", "[Co-reference](QA2) What's he going to do now?"

## Step 3: Q&A Refinement
- Ensure specificity and clarity. Remove ambiguous terms ("maybe/probably")
- Confirm all Q&A pairs are based on provided data, not fabricated details or common sense knowledge.
- Avoid overly formal, blunt or deliberate expressions.
- Eliminate media-specific terms ("frame/image/segment/clip")
- All Q&A pairs should avoid similarity to ensure comprehensive understanding of the entire video.

# Output Format
Provide the output in the following JSON format:  
```json
{format}
```
# Start Generating:  
- Timestamps for each video frame: {time}
- Current video frames: 
'''
    elif not summarize and not Global:
        format = '''
        {
            "L3": {
              "Q1": "[label]...",
              "A1": "",
              ...
            }
        }
                '''
    # TODO：'''在设计问答对时，你需要在充分理解完整视频内容的基础上，基于视频完整内容来设计问题，并确保其答案需依赖contex的文本信息；内容精炼明确，信息完整，不要冗长
    # 1. random.choice(range(len(segments)-1)), 选择1个幸运段落，用于生成对话回顾问题。将全部的QA送给GPT，要求其挑选其中有趣的、独立的问题，根据所给范式进行回顾性提问。放到dict最后，新增kv就行
    # 2. 长共指范式改一下；不用提问当前内容了，因此可以仅提供第一次出现时的相关视频帧。

        prompt = f'''
# Task Description
You are an expert in video comprehension, specializing in visual and contextual analysis to generate accurate and meaningful cross-time question-answer (Q&A) pairs. Follow these steps to complete the task:

## Task 1: Understanding the Current Video Clip. 
- Analyze the provided key frames of the current video clip alongside the previous titles and context. Treat the key frames as visual representations of the video footage.
- Each question should be written in natural language, prefixed by a label in square brackets.
- Focus on a holistic interpretation of the content to capture its essence.

## Task 2: Generating  Cross-temporal and Complex Q&A. 
**Definition**: Complex Q&A pair based on the entire video content, which relies on information from the "Previous Context" as a necessary premise or partial answer.
**Features**:  
    - Questions must be concise and self-contained. Must not include hints, clues, or information that could reveal the answers.
    - Answers must provide a correct, complete and clear solution to their question, avoiding negativity or evasiveness.
    - Q&A pairs must not rely on common-sense knowledge and must be based on the content provided (current and previous context).
**Quantity**: 5 Q&A pairs.  
**Optional labels**:
    - [Sequence Perception]: Builds on prior events, outlining a step-by-step progression of actions with emphasis on continuity. Example: "What did the characters do in sequence?"
    - [Causal discovery]: Explain the dirct cause or consequence of events in the video, not common-sense. Example: "Why is the flight delayed? ", "What causes the fire?".
    - [Intention]: Motivations behind behavior, using observable traits. Example: "Why is the woman in red shouting?"
    - [Prediction]: Predict future outcomes based on previous information. Examples: "Which regions are likely to face droughts?"
    - [Reasoning]: Drawing logical conclusions or making inferences based on evidence and facts. Examples: "What conclusion can be drawn from her reaction?","What was the reason why he finally gave up?"
    - [Analysis]: Examine the causes, structure, or significance of key events. Examples: "What is the turning point of the plot?", "How has marriage changed the host's life?"

## Task 3: Q&A Refinement
- Ensure specificity and clarity. Remove ambiguous terms ("maybe/probably")
- Avoid overly formal, blunt or deliberate expressions.
- Eliminate media-specific terms ("frame/image/segment/clip")
- All Q&A pairs should avoid similarity to ensure comprehensive understanding of the entire video.

# Output Format
Provide the output in the following JSON format:  
```json
{format}
```
# Start Generating:  
- Previous Context: {history} 
- Current video frames: 
'''
        return prompt
    # 生成QA后，TaskOne:进行段内信息总结，用于最后片段生成Global Q&A；TaskTwo:进行整体信息总结，后者为下一片段Q&A生成的上下文信息来源
    elif not Global:
        format1 = '{"L1_L2":"","L3":"",...}'
        format2 = '{"Task_One":{"L1_L2":"","L3":"",...},"Task_Two":{"L1_L2":"","L3":"",...}}'
        if not history:
            prompt = f'''
             Task description:
            You're an expert at extracting and summarizing information. Below is a Q&A dataset based on a certain video. Please review these Q&A pairs, and extract and summarize the contained information into a concise description. 

            - Requirements:
            1. The Q&A dataset consists of three levels: # L1&L2 (Direct Information), L3(Causal or Reasoning Information). First, summarize the contents of L1 and L2, and then summarize the contents of L3.
            2. The Q&A dataset contains some redundancies. Do not directly copy the text from the Q&A dataset. Instead, rephrase and distill the key information.
            3. Ensure no loss of information while aiming for brevity 

            - Output Format in JSON:
            ```json{format1}```

            - Here is the Q&A dataset:{qa_pairs}

            Please provide the final summarized result as per these requirements.
            '''
        else:
            prompt = f'''
            * Task One *
            You're an expert at extracting and summarizing information. Below is a Q&A dataset based on a certain video. Please review these Q&A pairs, and extract and summarize the contained information into a concise description. 

            - Requirements:
            1. The Q&A dataset consists of three levels: # L1&L2 (Direct Information), L3(Causal or Reasoning Information). First, summarize the contents of L1 and L2, and then summarize the contents of L3.
            2. The Q&A dataset contains some redundancies. Do not directly copy the text from the Q&A dataset. Instead, rephrase and distill the key information.
            3. Ensure no loss of information while aiming for brevity 

            - Here is the Q&A dataset:{qa_pairs}

            * Task Two *
            Please merge the newly summarized Q&A data with previously summarized Q&A data.

            - Requirements:
            1. Retain all unique information from both the new and previous summaries.
            2. Must add new key information from the Q&A dataset that is not already present in the previous summary.
            3. Ensure no loss of information while aiming for brevity.

            - Here is the previously information to be merged:{history}

            Output Format in JSON: 
            ```json{format2}```

            Now, Please provide the final summarized result as per these requirements.'''
    # 最后一段时，需要额外生成global QA, 会用到全部片段的QA总结（即Task One的内容）
    else:
        format = """
{
  "L4": {
    "Q1": "[label]...",
    "A1": "",
    "Q2": "[label]...",
    "A2": "",
    ...
  }
}"""
        prompt = f"""
# Task Description
You are an expert in video comprehension, specializing in visual and contextual analysis to generate accurate and meaningful cross-time question-answer (Q&A) pairs. Follow these steps to complete the task:

## Task 1: Understanding the Current Video Clip. 
- Analyze the provided key frames of the current video clip alongside the previous titles and context. Treat the key frames as visual representations of the video footage.
- Each question should be written in natural language, prefixed by a label in square brackets.
- Focus on a holistic interpretation of the content to capture its essence.

## Task 2: Generating Global and Complex Q&A Pairs. 
**Definition**: Global and Complex reasoning questions require multi-step analysis and comprehensive integration of all content from the video context to produce a coherent, well-supported answer.
**Features**:
    - These questions require a selective review based on video context and rely on an understanding of the video's complete **sequence of events or chain of reasoning**.
    - Questions are valuable and concise and do not contain any hints or partial answers.  
    - Answers must provide a concise, complete and clear solution to their question, avoiding negativity or evasiveness.  
    - Each QA pair must explicitly reference multiple clips.
**Quantity**: 2 Q&A pairs.  
**Labels**: 
    - [Global Analysis]: Questions requiring a detailed breakdown and multi-step analysis of specific topics, events, or processes across the video, integrating current and previous clips to reveal their structure, function, or significance. Example: "From all the chemical reactions in the video, please tell what specific reaction it is?"  
    - [Overall Summary]: Questions requiring a concise summary of key events, steps, or plot points across the entire video, based on a selective review of current and previous clips and an understanding of the full event sequence or reasoning chain.  Example: "What are the key clues for solving the case across the video?"
    
## Task 3: Q&A Refinement
- Ensure specificity and clarity. Remove ambiguous terms: "maybe", "probably", etc. Remove media-specific terms "segment" or "clip".
- Confirm all Q&A pairs are based on provided data, not fabricated details or common sense knowledge.
- Avoid overly formal, blunt or deliberate expressions.
- Ensure comprehensive understanding of the entire video and avoid similarity between Q&A pairs.

# Output Format
Provide the output in the following JSON format:  
```json
{format}
```
# Start Generating:  
Previous Context: {Global} 
Current video frames: 
"""
    return prompt


def recall(QA_pairs):
    prompt = '''
    **Task**:  
You will be given a set of question-answer (QA) pairs from a dialogue. Your task is to select one QA pair and create a new "Dialogue Recalling" QA pair based on it, as described below.

**"Dialogue Recalling" Definition**:  
- **Question**: Ask the model to recall a specific previous QA pair by referencing a key theme or phrase.  
- **Answer**: Accurately restate the original question and answer from that QA pair.

**Instructions**:  
1. Pick an "easy-to-ask" QA pair from the input, where the question has a clear theme and the answer is concise and memorable.  
2. Identify a key theme or phrase from the chosen question (e.g., "preparing cooking tools" from "Do I need to prepare tools like a sifter, beater, and pans?").  
3. Write a natural recalling question using the theme, for example:  
  - "Do you remember what I asked about [theme]?"  
  - "I asked about [theme] at 18 seconds,  how you responded?"  
  - "I forgot our discussion about [theme]. Can you repeat it?"  
  - "What did I ask about [theme], and how did you answer?"  

4. Write the recalling answer by restating the original QA pair, for example:  
  - "Yes, I remember. You asked, '[original question]'."  
  - "Well, I answered, '[original answer].'"  
  - "Sure. You asked, '[original question]', and I answered, '[original answer].'"

**Input Example**:  
{
...
"Q3": "What ingredients do I need for a basic omelette?"
"A3": "Eggs, salt, pepper, and a bit of butter."  
...
}

**Output Example**:  
{
  "Original_QA_ID": "3",
  "QA_pairs": {
    "Q1": "Do you remember what I asked about making an omelette?",
    "A1": "Yes, I remember. You asked, 'What ingredients do I need for a basic omelette?', and the answer is 'Eggs, salt, pepper, and a bit of butter.'"
  }
}

**Output Format**:  
Return the result in JSON format, including both the "original_QA_ID" and "dialogue_recalling_QA" sections.''' +    f'''

**Input**:  {QA_pairs}'''
    return prompt



def cr_prompt1(cr_object, history):
    format = '''{
  "L1": {
      "Q1": "[Object Tracking]...",
      "A1": ""
  }
}'''
    prompt = f'''
### **Input**:
1. Historical titles and historical context for each previous video clip.
2. The current video frames represent the content of this video clip.

### **Task Description**:
You are an expert in video comprehension with advanced skills in visual and contextual analysis.
Your task is to analyze the current video clip and generate basic question-answer (Q&A) pairs about a specific object/person.

### ** Generate Q&A Pairs**:
- Generate visual understanding Q&A pairs(about actions, attributes, intention, etc.) based solely on the current video clip.
- Q&A Label: [Object Tracking]
- Quantity: 1 Q&A pairs

### **Requirements**:
- Use the given object's name in the question directly and completely!
- Ensure specificity and clarity. Remove ambiguous terms: "maybe", "probably", etc. Remove media-specific terms "segment" or "clip".
- Do not contain any hints or partial answers.  
- Make the questions and answers more direct and natural. Avoid overly formal expressions. 

#### **Output Format in JSON**:{format}

### **Start Generating**:
- Object's name: {cr_object},
- Historical context: {history}
- Current video frames:'''
    return prompt

def cr_prompt2(cr_object, cr_qa, p_context):
    # {time2}-{title2}-{summaries2}
    format = '''
{
    "Q1": "",
    "A1": ""
    "Q2": "",
    "A2": ""
}
    '''
    prompt = f'''### **Input**:
### **Task Description**:
You are an expert in video comprehension with advanced skills in visual and contextual analysis.
Your role is to analyze and understand the current video clip, and then generate "Object Tracking" Question-Answer (Q&A) pairs for a specific object or person.

### **Steps to Follow** :
1. Treat the video content as a continuous flow of events. Understand and Analyze the Object's Context Across Appearances.
2. Generate Two Object Tracking Q&A Pair:
The Q&A pair should continue tracking and referencing the same specific object or person explicitly mentioned previously, regardless of visual changes.
To demonstrate continuous tracking and memory of the object, create two Q&A Pair followed:
(1) Directly ask about the object's status or actions during its previous appearance. Examples: "What color was the driver wearing initially?" "What was the cute dog doing the last time it appeared?"
(2) Directly ask about when the object first appeared. Examples: "When was the last time this male reporter interviewed a passerby?"

### **Requirements**:
- Use the given object's name in the question directly and completely!
- Ensure Q&A Pairs are concise and direct. Ensure specificity and clarity. Remove ambiguous terms ("maybe/probably")
- Use natural phrases like "previously shown" "in the earlier scenes" or "right now" to maintain continuity, rather than media-specific terms like "in previous frame/image/segment/clip".
- Do not contain any hints or partial answers.  
- Make the questions and answers more direct and natural. Avoid overly formal expressions. 

#### **Output Format in JSON**:{format}

### *Input**:
- Object's name: {cr_object}
- Previous Q&A:{cr_qa}
- Context of First Appearance:{p_context}
- Previous video Clip: '''
    return prompt

