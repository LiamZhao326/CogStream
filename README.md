# CogStream: Context-guided Streaming Video Question Answering

This repository is the official implementation of [CogStream: Context-guided Streaming Video Question Answering]. 

A diagram illustrating the architecture of the CogReasoner model is provided below.
![Model Architecture](./images/model_diagram.png)

## Requirements

**Note**: Run all commands from the repository root directory to ensure correct path resolution.

We follow VideoLLaMA3. To install requirements:

```setup
conda env create -f environment.yml
```

Download only the VideoLLaMA3 model weights (.safetensors files) from here and place them in the ./model folder in this repository:

- [VideoLLaMA3](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA3-7B)

## Training

To train the Historic Dialogue Retrieval module in the paper (First Stage), run this command:

```train
torchrun --nproc_per_node=<number of processes> train/language_model_training.py --model_path <path to the base model directory> --QA_path <path to the dataset QA directory>
```

- `--nproc_per_node=<number of processes>`: Specifies the number of processes to run per node, typically set to the number of available GPUs (e.g., 8 for 8 GPUs).

To train the Video-text Interleave Reasoning module in the paper (Second Stage), you need to configure `accelerate` before training. Load the provided `accelerate` configuration file by running:

```bash
accelerate config --load_config accelerate_config.yaml
```

Then, run the training command:

```train
accelerate launch train/second_stage_training.py --model_path <path to the base model directory> --video_dir <directory containing train video files> --query_dir <directory containing train query (QA) files> --num_epochs <training epochs number>
```

## Evaluation

To evaluate our model on CogStream, first run the following command to generate answers on our dataset:

```eval
torchrun --nproc_per_node=<number of processes> evaluate/answer_generate.py --model_path <path to the base model directory> --lora_adapter_1_path <path to the first stage LoRA adapter> --lora_adapter_2_path <path to the second stage LoRA adapter> --video_dir <directory containing test video files> --query_dir <directory containing test query (QA) files> --save_dir <directory to save the result>
```


## Pre-trained Models

You can find pretrained lora weights in the directory:

- [First Stage](./pre_trainend_lora_weights/stage1)
- [Second Stage](./pre_trainend_lora_weights/stage2)

## Results

A visualization of an example result demonstrating the model's performance is shown below.
![Example Visualization](./images/example_result.png)

Performance metrics of different models in 11 **CogStream** capabilities. Prm. denotes the number of model parameters, Frm. denotes the number of sampled frames.

| **Method** | **Prm.** | **Frm.** | **Basic** |       |         |      | **Streaming** |       |       |      |      | **Global** |       | **Avg.**$\uparrow$ |
|-----------------|----------|----------|-----------|-------|---------|------|---------------|-------|-------|------|------|------------|-------|-----------------|
|                 |          |          | Att.      | Obj.  | Co-ref. | Act. | Rea.          | Seq.  | Dial. | Dyn. | Obj. | Over.      | Glob. |                 |
| *Open-Source Models* |          |          |           |       |         |      |               |       |       |      |      |            |       |                 |
| InternVL2       | 7B       | 12/seg   | 52.3      | 59.0  | 36.6    | 36.3 | 52.6          | 41.9  | 39.2  | 39.1 | 43.9 | 52.4       | 59.8  | 48.66           |
| LongVA          | 7B       | 12/seg   | 63.6      | 55.0  | 42.0    | 33.6 | 53.1          | 40.9  | 55.4  | 25.3 | 36.8 | 42.4       | 53.3  | 48.76           |
| VideoLLaMA 2    | 7B       | 20/seg   | 60.0      | 61.7  | 47.8    | 46.4 | 47.5          | 47.4  | 54.1  | 30.2 | 56.8 | 54.3       | 54.8  | 50.72           |
| MiniCPM-o 2.6   | 8B       | 20/seg   | 77.3      | 76.4  | 63.6    | 60.6 | 65.9          | 61.0  | 47.1  | 50.9 | 44.7 | 57.4       | 62.8  | 64.08           |
| VideoLLaMA 3    | 7B       | 1fps     | 75.7      | 71.8  | 62.6    | 64.6 | 67.7          | 61.5  | 56.9  | 52.4 | 60.3 | 66.0       | 72.3  | 66.52           |
| MiniCPM-V-2.6   | 8B       | 20/seg   | **78.6** | 73.6  | 70.7    | 59.6 | **70.5** | 59.7  | 50.0  | 49.2 | **64.5** | 64.2       | 69.4  | 66.84           |
| **CogReasoner** | 7B       | 1fps     | 77.3      | **78.9**| **74.6**| **70.0** | 69.7     | **68.8**| **83.4**| **70.5** | 62.7     | **75.4** | **76.0**| **72.26** |
| *Proprietary Models* |          |          |           |       |         |      |               |       |       |      |      |            |       |                 |
| Gemini 1.5 Pro  | -        | 20/seg   | 75.5      | 73.4  | 66.4    | 62.5 | 66.2          | 61.1  | 64.1  | 42.0 | 36.2 | 69.4       | 74.4  | 66.04           |
| Qwen2-VL-Max    | -        | 50(max)  | 77.2      | **76.7**| **70.4**| **69.2** | 76.7          | 66.5  | 62.3  | **53.7** | 52.4 | 76.2       | 76.6  | 72.58           |
| GPT-4o          | -        | 20/seg   | **78.4** | 73.9  | 68.2    | 66.1 | **77.5** | **72.1**| **73.0**| 52.4 | **44.2** | **77.0** | **79.6**| **73.90** |


## License

This project is licensed under the [MIT License](LICENSE). All contributions to the code must be made under this license. See the [LICENSE](LICENSE) file for details.






