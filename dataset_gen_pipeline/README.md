# CogStream Dataset

## Introduction
### The CogStream Task
The CogStream task simulates a real-world scenario where users watch an ongoing video stream and continuously interact with a model by asking questions about the content presented so far. As illustrated in Figure 1, we categorize all QA pairs into three distinct types based on the temporal coverage of historical information required for answering:
- Basic QA: Questions answerable from the most recent video segment alone.
- Streaming QA: Questions that require information from a limited historical window of recent segments.
- Global QA: Questions that necessitate understanding the entire video history up to the current point..

## Dataset Overview
To build this dataset, we collected 6,361 unannotated videos from diverse public sources: [MovieChat](https://github.com/rese1f/MovieChat) (40.2%), [MECD](https://github.com/tychen-SJTU/MECD-Benchmark) (16.8%), [QVHighlights](https://github.com/jayleicn/moment_detr) (9.8%), [VideoMME](https://github.com/MME-Benchmarks/Video-MME) (6.5%), [COIN](https://github.com/coin-dataset/annotations) (18.0%), and [YouCook2](http://youcook2.eecs.umich.edu/) (8.6%). After selection for quality, the dataset comprises 1,088 videos and 59,032 QA pairs.the final dataset comprises 1,088 videos and 59,032 QA pairs, abd the data is formally split into a training set (852 videos) and a testing set (236 videos) with no overlap.

An illustration of the CogStream task and the dataset's composition is provided below.
<div align="center">
  <img src="../images/dataset_EN.png" alt="CogStream Task and Dataset Overview" width="800"/>
</div>

## Download and Setup

You can download the dataset from [Zenodo](https://zenodo.org/records/15870909?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImZiYmFlODkxLWUwZTUtNGUyMi04NzU0LWVhYTUxZWM3MzNhMiIsImRhdGEiOnt9LCJyYW5kb20iOiI3MWI5MzIwZDU3MDA1OWExMTBlZWQ5NTE2OTUzZTA5NSJ9.ITwHqhbk7cUcd4f9qHpIx972Jgfdis5qiMJwNXyo5vT7-Ltd-dvGQLD7yItrKmJPJI8oUCLb8ItODsrm7_t6NA). For seamless integration with training scripts, place this `Dataset` folder directly in the project's root directory (e.g., `./Dataset/Train_dataset/`).

After downloading and unzipping, place the Dataset folder in the root of your project directory. The structure is designed for seamless integration with the provided scripts (e.g., ./Dataset/Train_dataset/).

## Dataset Structure
The dataset is organized into the following directory structure:
```
./Dataset/
├── Train_dataset/
│   ├── event_segments/
│   │   └── {video_name}/
│   │       └── {video_name}_segment_*.mp4
│   ├── QA_Dataset/
│   │   └── {video_name}.json
│   └── VQA_Dataset/
│       └── {video_name}.json
└── Test_dataset/
    └── (...same structure as Train_dataset)
```

- **`event_segments/`**: Stores the non-overlapping video segments (.mp4 files) for each video, organized in subdirectories. These are the primary inputs for model processing.
- **`QA_Dataset/`**: Stores JSON files with QA pairs focused on dialogue retrieval task. Each JSON includes arrays of questions, answers, and basic metadata.
- **`VQA_Dataset/`**: Stores JSON files for Streaming VQA tasks, including detailed metadata such as labels for Basic/Streaming/Global types.

# Dataset Generation 
## Pipeline

We propose a semi-automatic pipeline to construct this dataset from unlabeled videos, as shown in Figure 2. The pipeline consists of four key steps:
<div align="center">
  <img src="../images/pipeline.png" alt="CogStream Task and Dataset Overview" width="800"/>
</div>
The pipeline consists of four main stages:

1. **Video Segmentation:** The raw video is divided into non-overlapping segments based on event boundaries, utilizing the [SceneTiling](https://github.com/bigai-nlco/VideoLLaMB?tab=readme-ov-file) algorithm, followed by manual review and refinement to ensure quality.

2. **QA Pairs Generation:** A Multimodal Large Language Model (MLLM), such as GPT-4o, is used to generate a diverse set of QA pairs for each video segment based on its visual content.

3. **Relevance QA Set Construction**  
   A relevance scoring mechanism is applied to measure interdependencies among QA pairs across different video segments, enabling the identification of interconnected question-answer sets.

4. **Dialogue Stream Generation**  
   Finally, coherent dialogue streams are constructed by linking QA pairs with high relevance scores, simulating a natural, interactive conversation flow.

## Generating Your Own Dataset

You can use our pipeline to create a custom CogStream-style dataset from your own unlabeled videos. All scripts are located in the `./dataset_gen_pipeline/` directory.

### 1. Video Segmentation
First, prepare a JSON file that maps video names to segmentation timestamps:
```json
  [
      {
          "name": "video_1",
          "time_stamps": [
              [0, 10.1],
              [10.1, 20.2]
          ]
      }
  ]
  ```
Place this file and your raw videos in a single input directory. Then, run the script:
```bash
python ./dataset_gen_pipeline/split_videos.py --input_file raw_videos --output_file Dataset --num_frames 8
```
- `--input_file`: Directory containing raw videos and the JSON file.
- `--output_file`: Root directory for the output dataset (default: `Dataset`). Segmented videos go to `/segments/`, and uniformly sampled keyframes to `/keyframes/`.
- `--num_frames`: Number of frames to sample uniformly per segment.

This produces segmented video clips (for inference) and sampled frames (for QA generation).

### 2. QA Pairs Generation
Generate QA pairs using an MLLM:
```bash
python ./dataset_gen_pipeline/generation.py --root_dir Dataset
```
- `--root_dir`: Root directory of the dataset.
- `--GPT_model`: Defines the model to use for generation (e.g., gpt-4o).

After generation, reformat the output:
```bash
python ./dataset_gen_pipeline/reformating.py
```

### 3. QA Pairs Relevance Scoring
Score the relevance between all generated QA pairs:
```bash
python ./dataset_gen_pipeline/scoring.py --root_dir Dataset
```
- `--root_dir`: Specifies the root directory of the dataset. Other paths will be inferred automatically
- `--GPT_model`: Defines the model to use for scoring.

### 4. Dialogue Stream Generation
Finally, construct the dialogue streams:
```bash
python ./dataset_gen_pipeline/dialogue_path_construction.py --root_dir Dataset
```
- `--K`: Number of CQAs introduced per video segment (default: 2).
- `--N`: Number of generated QA sequences (default: 5).
- `--R`: Score threshold for relevance (default: 4).
- `--tau`: Softmax temperature coefficient (default: 1).
