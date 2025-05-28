# CogStream: Context-guided Streaming Video Question Answering

This repository is the official implementation of [CogStream: Context-guided Streaming Video Question Answering]. 

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

To evaluate our model on CogStreaam, first run the following command to generate answers on our dataset:

```eval
torchrun --nproc_per_node=<number of processes> evaluate/answer_generate.py --model_path <path to the base model directory> --lora_adapter_1_path <path to the first stage LoRA adapter> --lora_adapter_2_path <path to the second stage LoRA adapter> --video_dir <directory containing test video files> --query_dir <directory containing test query (QA) files> --save_dir <directory to save the result>
```


## Pre-trained Models

You can find pretrained lora weights in the directory:

- [First Stage](./pre_trainend_lora_weights/stage1)
- [Second Stage](./pre_trainend_lora_weights/stage2)

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 






