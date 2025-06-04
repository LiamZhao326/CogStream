import os
import json
from tqdm import tqdm
import argparse
import numpy as np
import random
import re

# 假设 COG_data_format 是一个自定义模块，包含 format_vqa_dataset 函数
from data_formating import format_vqa_dataset

# 定义常量
SPECIAL_CLASSES = ['Temporal Perception', 'Dialogue Recalling', 'Object Tracking', 'Dynamic Updating']
BASIC_CLASSES = ['L1', 'L3', 'L2', 'L4']
ALL_CLASSES = SPECIAL_CLASSES + BASIC_CLASSES
SCORE_THRESHOLD = 8
L1_SELECTION_RATIO = 0.4
CHAIN_BOOST_FACTOR = 0.1

def extract_label(text):
    """从文本中提取标签、问题和时间戳（当前时间戳未使用）。"""
    time = None  # 时间戳功能当前未启用
    text = re.sub(r'\([^)]+\)', '', text)  # 移除括号内容
    match = re.match(r"^\[(.*?)\]\s*(.*)", text)
    if match:
        label = match.group(1)
        question = match.group(2)
        return label, question, time
    return None

class Sequence:
    """生成视频问答序列的核心类。"""
    def __init__(self, video_data, list_raw, R=4, tau=2, K=1, N=3):
        self.list_raw = list_raw  # 评分数据
        self.video_data = video_data  # QA 对数据
        self.tau = tau  # Softmax 温度参数
        self.K = K  # 每段引入的高级 QA 数量
        self.N = N  # 生成的序列数量
        self.R = R  # 分数阈值

        self.qa_slq2id = {}  # (segment, level, qa) 到 ID 的映射
        self.qa_id2cot = {}  # ID 到 CoT（推理链）的映射
        self.chain_lengths = {}  # 推理链长度
        self.qa_num = 0  # QA 对总数
        self.qa_list = []  # 当前序列中的 QA ID
        self._initial_params()

    def id2cot(self, id0):
        """根据 ID 获取推理链（CoT）。"""
        coi = self.qa_id2cot.get(id0, [])
        if coi:
            idx = self.qa_list.index(id0)
            coi = [(self.id2slq(id1), score) for id1, score in coi if id1 in self.qa_list[:idx]]
            coi = sorted(coi, key=lambda x: x[1], reverse=True)
            return [x[0] for x in coi]
        return []

    def slq2id(self, slq0, level_idx=None):
        """将 (segment, level, qa) 三元组转换为 ID。"""
        for id_, slq in self.qa_slq2id.items():
            if slq0 == slq:
                return int(id_)
        if level_idx != 2:  # 仅在特定情况下打印警告
            print(f'Cannot find {slq0}')
        return None

    def id2slq(self, id0):
        """将 ID 转换为 (segment, level, qa) 三元组。"""
        for id_, slq in self.qa_slq2id.items():
            if id0 == int(id_):
                return slq
        print(f'Cannot find {id0}')
        return None

    def _initial_params(self):
        """初始化参数和评分矩阵。"""
        Object = {}
        Dynamic = []
        Dialogue = [False, None, None]  # [是否有前驱，前驱 slq，前驱 id]

        for seg_info in self.video_data:
            seg_idx = seg_info['segment_id']
            qa_pairs = seg_info['QA_pairs']
            for level_key, QAs in qa_pairs.items():
                if level_key in BASIC_CLASSES:
                    level_idx = int(level_key.replace('L', ''))
                    qa_indices = {int(key[1:]) for key in QAs if key.startswith('Q') and f"A{key[1:]}" in QAs}
                    for qa_idx in sorted(qa_indices):
                        slq = (seg_idx, level_idx, qa_idx)
                        self.qa_slq2id[self.qa_num] = slq
                        self.qa_id2cot[self.qa_num] = []
                        self.qa_num += 1
                elif level_key in SPECIAL_CLASSES:
                    if level_key == 'Dynamic Updating':
                        for i, _ in enumerate(QAs):
                            slq = (seg_idx, level_key, i)
                            coi = Dynamic.copy()
                            self.qa_slq2id[self.qa_num] = slq
                            self.qa_id2cot[self.qa_num] = coi
                            Dynamic.append((self.qa_num, SCORE_THRESHOLD))
                            self.qa_num += 1
                        continue
                    elif level_key == 'Temporal Perception':
                        slq = (seg_idx, level_key, 1)
                        coi = []
                        ori_qa_id = int(next(iter(QAs['QA_pairs'].keys()))[-1])
                        ori_seg_id = int(QAs['Original_seg_ID'])
                        q, _, s = Dialogue
                        if s is not None and ori_qa_id == s[-1] and ori_seg_id == s[0]:
                            print('====Find missing Dialogue Recalling===')
                            Dialogue = [True, Dialogue[1], [(self.qa_num, SCORE_THRESHOLD)]]
                    elif level_key == 'Dialogue Recalling':
                        ori_seg = int(QAs['Original_seg_ID']) + 1
                        ori_qaid = int(str(QAs['Original_QA_ID'])[-1])
                        slq = (seg_idx, level_key, 1)
                        coi_id = self.slq2id((ori_seg, 1, ori_qaid))
                        if coi_id is None:
                            print(f'{level_key} !')
                            Dialogue = [False, slq, (ori_seg - 1, 1, ori_qaid)]
                            continue
                        else:
                            coi = [(coi_id, SCORE_THRESHOLD)]
                    elif level_key == 'Object Tracking':
                        for qa_idx, qa_value in QAs.items():
                            if 'L1' in qa_value:
                                slq = (seg_idx, level_key, (qa_idx, -1))
                                Object[qa_idx] = self.qa_num
                                coi = []
                            else:
                                slq = (seg_idx, level_key, (qa_idx, random.randint(0, 1)))
                                coi = [(Object[qa_idx], SCORE_THRESHOLD)]
                    self.qa_slq2id[self.qa_num] = slq
                    self.qa_id2cot[self.qa_num] = coi
                    self.qa_num += 1

        S0 = [self._initial_scores(list_raw) for list_raw in self.list_raw]
        S = np.mean(S0, axis=0) if len(S0) > 1 else S0[0]
        if len(S0) > 1:
            Diff = np.abs(S0[0] - S0[1])
            mask = Diff >= self.R
            S[mask] = np.maximum(S0[0][mask], S0[1][mask])

        self.S = np.zeros_like(S)
        for id1 in range(self.S.shape[0]):
            for id2 in range(self.S.shape[1]):
                score = S[id1, id2]
                if score >= self.R and id1 != id2:
                    self.qa_id2cot[id1].append((id2, score))
                    self.S[id1, id2] = score

    def _initial_scores(self, list_raw):
        """根据评分数据初始化评分矩阵。"""
        S = np.zeros((self.qa_num, self.qa_num), dtype=float)
        for seg_key, seg_info in list_raw.items():
            seg_idx = int(seg_key.replace('segment ', ''))
            for level_key, QAs in seg_info.items():
                if level_key not in BASIC_CLASSES:
                    continue
                level_idx = int(level_key[-1])
                for cqa_key, pqa_list in QAs.items():
                    cqa_idx = int(cqa_key[-1])
                    id1 = self.slq2id((seg_idx, level_idx, cqa_idx))
                    if id1 is None:
                        continue
                    for pseg_key, pseg_info in pqa_list.items():
                        pseg_idx = int(pseg_key.replace('segment ', ''))
                        for plevel_key, pQAs in pseg_info.items():
                            plevel_idx = int(plevel_key[-1])
                            for pqa_key, pqa_scores in pQAs:
                                pqa_idx = int(pqa_key[-1])
                                id2 = self.slq2id((pseg_idx, plevel_idx, pqa_idx))
                                if id2 is None:
                                    continue
                                score = int(pqa_scores['score'] if isinstance(pqa_scores, dict) else pqa_scores)
                                S[id1, id2] = score
        return S

    def build_sequence(self):
        """生成多个 QA 序列。"""
        qa_sequences = []
        segments = sorted({s[0] for s in self.qa_slq2id.values()})
        for _ in range(self.N):
            self.chain_lengths = {}
            self.qa_list = []
            sequence = []
            for seg_idx in segments:
                self._select_dy_qa(seg_idx)
                self._select_basic_qa(seg_idx)
                self._select_advanced_qa(seg_idx)
                self._select_special_qa(seg_idx)
            for _id in self.qa_list:
                sequence.append({'CQA': self.id2slq(int(_id)), 'COI': self.id2cot(int(_id))})
            qa_sequences.append(sequence)
        return qa_sequences

    def _select_dy_qa(self, seg_idx):
        """选择 Dynamic Updating 类 QA。"""
        candidates = [qa_id for qa_id, (s_idx, level, _) in self.qa_slq2id.items()
                      if s_idx == seg_idx and level == 'Dynamic Updating']
        self.qa_list.extend(candidates)

    def _select_special_qa(self, seg_idx):
        """选择特殊类 QA（除 Dynamic Updating 和 Dialogue Recalling）。"""
        candidates = [qa_id for qa_id, (s_idx, level, _) in self.qa_slq2id.items()
                      if s_idx == seg_idx and level in SPECIAL_CLASSES and level not in ['Dynamic Updating', 'Dialogue Recalling']]
        dialogue_candidates = [qa_id for qa_id, (s_idx, level, _) in self.qa_slq2id.items()
                               if s_idx == seg_idx and level == 'Dialogue Recalling']
        if dialogue_candidates:
            for qa_id in dialogue_candidates:
                p_id = [p_id for p_id, _ in self.qa_id2cot[qa_id] if p_id in self.qa_list]
                if p_id:
                    candidates.extend(dialogue_candidates)
                else:
                    print("===NO DIA===")
        self.qa_list.extend(candidates)

    def _select_basic_qa(self, seg_idx):
        """选择基础类 QA（L1 和可能的 L2）。"""
        l1_candidates = [qa_id for qa_id, (s_idx, level, _) in self.qa_slq2id.items()
                         if s_idx == seg_idx and level == 1]
        l2_candidates = [qa_id for qa_id, (s_idx, level, _) in self.qa_slq2id.items()
                         if s_idx == seg_idx and level == 2]
        select_num = max(1, int(len(l1_candidates) * L1_SELECTION_RATIO))
        selected_ids = random.sample(l1_candidates, select_num)
        if random.randint(0, 1):
            for qa_id in l2_candidates:
                p_id = [p_id for p_id, _ in self.qa_id2cot[qa_id] if p_id in selected_ids]
                if not p_id:
                    continue
                index = selected_ids.index(p_id[0])
                selected_ids.insert(index + 1, qa_id)
                for _id in selected_ids:
                    self._update_chain_length(_id)
                self.qa_list.extend(selected_ids)
                return
        for _id in selected_ids:
            self._update_chain_length(_id)
        self.qa_list.extend(selected_ids)

    def _select_advanced_qa(self, seg_idx):
        """选择高级类 QA（L3 和 L4）。"""
        for l in [3, 4]:
            l2_candidates = [qa_id for qa_id, (s_idx, level, _) in self.qa_slq2id.items()
                             if s_idx == seg_idx and level == l]
            random.shuffle(l2_candidates)
            for _ in range(self.K):
                valid_l2 = []
                for qa_id in l2_candidates:
                    if qa_id in self.qa_list:
                        continue
                    predecessors = [p_id for p_id, _ in self.qa_id2cot[qa_id] if p_id in self.qa_list]
                    if not predecessors:
                        valid_l2.append((qa_id, 1))
                        continue
                    max_score = max(self.S[qa_id][p_id] for p_id in predecessors)
                    if max_score == SCORE_THRESHOLD:
                        self._update_chain_length(qa_id)
                        self.qa_list.append(qa_id)
                        continue
                    chain_boost = CHAIN_BOOST_FACTOR * max(self.chain_lengths.get(p, 0) + 1 for p in predecessors)
                    valid_l2.append((qa_id, max_score + chain_boost))
                if not valid_l2:
                    break
                scores = [s for _, s in valid_l2]
                probs = self._softmax(scores)
                selected_id = np.random.choice([qa_id for qa_id, _ in valid_l2], size=1, p=probs)
                for qa_id in selected_id:
                    self._update_chain_length(qa_id)
                self.qa_list.extend(selected_id)

    def _update_chain_length(self, qa_id):
        """更新 QA 的推理链长度。"""
        predecessors = [p_id for p_id, _ in self.qa_id2cot.get(qa_id, [])]
        self.chain_lengths[qa_id] = max(self.chain_lengths.get(p, 0) for p in predecessors) + 1 if predecessors else 1

    def _softmax(self, scores):
        """带温度参数的 Softmax 归一化。"""
        exp_scores = np.exp(np.array(scores) / self.tau)
        return exp_scores / exp_scores.sum()

def main(args):
    """主函数，处理视频数据并生成 VQA 数据集。"""
    R = args.R
    K = args.K
    N = args.N
    tau = args.tau
    root_dir = args.root_dir

    qa_tot_path = os.path.join(root_dir, 'QA_ToI')
    qa_tot_path1 = os.path.join(root_dir, 'QA_ToI_gpt')
    qa_pairs_path = os.path.join(root_dir, 'QA_pairs_new2')
    output_folder = os.path.join(root_dir, 'COG_Dataset_raw')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_vid_num = len(os.listdir(qa_tot_path))
    for vid_num, video_folder in tqdm(enumerate(os.listdir(qa_tot_path)), total=total_vid_num,
                                      desc='Generating VQA Dataset'):
        video_name = os.path.splitext(video_folder)[0]
        json_path = os.path.join(output_folder, f'{video_name}.json')
        if os.path.exists(json_path):
            continue

        list_raw_path = os.path.join(qa_tot_path, video_folder)
        list_raw_path1 = os.path.join(qa_tot_path1, video_folder)
        video_data_path = os.path.join(qa_pairs_path, video_folder)
        all_list_raw = []

        try:
            with open(video_data_path, 'r', encoding='utf-8') as jsonfile:
                video_data = json.load(jsonfile)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f'Error loading video data for {video_name}: {e}')
            continue

        if os.path.exists(list_raw_path1):
            try:
                with open(list_raw_path1, 'r') as jsonfile:
                    all_list_raw.append(json.load(jsonfile))
            except json.JSONDecodeError as e:
                print(f'Error loading list_raw_path1 for {video_name}: {e}')
        if os.path.exists(list_raw_path):
            try:
                with open(list_raw_path, 'r') as jsonfile:
                    all_list_raw.append(json.load(jsonfile))
            except json.JSONDecodeError as e:
                print(f'Error loading list_raw_path for {video_name}: {e}')
        if not all_list_raw:
            print(f'No list_raw files found for {video_name}')
            continue

        qa_generater = Sequence(video_data, all_list_raw, R=R, tau=tau, K=K, N=N)
        qa_sequences = qa_generater.build_sequence()
        cqa_ids = []
        for seq in qa_sequences:
            cqa_id = {f'{s}{l}{q}': num for num, c in enumerate(seq) for s, l, q in [c['CQA']]}
            cqa_ids.append(cqa_id)

        output_data_list = []
        for num, qa_seq in enumerate(qa_sequences):
            one_seq_data = []
            cqa_id = cqa_ids[num]
            for qa in qa_seq:
                cqa, coi = qa["CQA"], qa["COI"]
                segment_id, level_id, qa_id = cqa
                IS_VISUAL = True

                seg_data = video_data[segment_id - 1]
                segment_timestamp = seg_data.get("segment_timestamp", None)

                if level_id in SPECIAL_CLASSES:
                    qa_pairs = seg_data['QA_pairs'][level_id]
                    label = 'Streaming/' + level_id
                    t = None
                    if level_id == 'Dialogue Recalling':
                        qa_pair = {k: qa_pairs['QA_pairs'][k] for k in list(qa_pairs['QA_pairs'])[-2:]}
                        IS_VISUAL = False
                    elif level_id == 'Temporal Perception':
                        qa_pair = qa_pairs['QA_pairs']
                    elif level_id == 'Object Tracking':
                        if qa_id[-1] == -1:
                            qa_pair = qa_pairs[qa_id[0]]['L1']
                            qa_pair['Q1'] = re.sub(r'\[.*?\]', '', qa_pair['Q1'])
                        else:
                            qaid = qa_id[-1] + 1
                            q_idx, a_idx = f'Q{qaid}', f'A{qaid}'
                            qa_pair = {q_idx: qa_pairs[qa_id[0]][q_idx], a_idx: qa_pairs[qa_id[0]][a_idx]}
                    elif level_id == 'Dynamic Updating':
                        t = qa_pairs[qa_id]['time']
                        qa_pair = qa_pairs[qa_id].copy()
                        qa_pair.pop('time')
                else:
                    qa_pairs = seg_data['QA_pairs'].get(f'L{level_id}', {})
                    qa_pair = {}
                    q_idx, a_idx = f'Q{qa_id}', f'A{qa_id}'
                    label, q_data, t = extract_label(qa_pairs.get(q_idx, ''))
                    if label:
                        label_prefix = 'Global/' if int(level_id) == 4 else 'Streaming/' if int(level_id) == 3 else 'Basic/'
                        label = label_prefix + label
                        if a_idx in qa_pairs:
                            qa_pair = {q_idx: q_data, a_idx: qa_pairs[a_idx]}

                cot_info = [cqa_id[f'{s}{l}{q}'] for s, l, q in coi] if coi else []
                one_seq_data.append({
                    "segment_path": seg_data['segment_path'],
                    "segment_timestamp": segment_timestamp[-1] if segment_timestamp else None,
                    "event_timestamp": t if t is not None else (segment_timestamp[-1] if segment_timestamp else None),
                    "qa_info": cqa_id[f'{segment_id}{level_id}{qa_id}'],
                    "label": label,
                    "is_visual": IS_VISUAL,
                    "QA_pairs": qa_pair,
                    "coi_qa_info": str(cot_info),
                })
            output_data_list.append({
                "video_name": video_name,
                "seq_info": f"{num + 1}/{len(qa_sequences)}",
                "Data": one_seq_data
            })

        with open(json_path, 'w') as output_file:
            json.dump(output_data_list, output_file, indent=4)
        format_vqa_dataset(root_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate VQA Dataset")
    parser.add_argument("--root_dir", default=r"C:\Users\COG27\Desktop\code\code\2_test", type=str)
    parser.add_argument("--K", default=2, type=int, help="Number of CQAs introduced per video segment")
    parser.add_argument("--N", default=5, type=int, help="Number of generated QA sequences")
    parser.add_argument("--R", default=4, type=int, help="Score threshold")
    parser.add_argument("--tau", default=1, type=int, help="Softmax temperature coefficient")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)