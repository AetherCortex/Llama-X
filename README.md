
<p align="center" width="100%">
<a ><img src="src/imgs/pandallm.png" alt="Llama-X" style="width: 60%; min-width: 300px; display: block; margin: auto;"></a>
</p>


[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)

## Llama-X: Open Academic Research on Improving LLaMA to SOTA LLM


This is the repo for the Llama-X, which aims to:

- Progressively improve the performance of LLaMA to SOTA LLM with open-source community.
- Conduct Llama-X as an open academic research which is long-term, systematic and rigorous.
- Save the repetitive work of community and we work together to create more and faster increment.

The project will follow these principles:

- We will publish all the `code`, `model`, `data`, and `experiments` details.
- We will `continuously` improve the model version by version and open the `newest` method.
- We will summary the method of each main version as `academic papers`.
- We announce a complete [research plan](#research-areas). The contributors are wellcome to cooperate with 
each other to progressively improve Llama-X through 
iteration of the [target versions](#model).
- The check-in of the new model must achieve significant improvement with current version on [automatic evaluation](#evaluation).

&#x1F4E3; Please join <a href="https://discord.gg/2etwhe6GvU"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a> if you are interested in Llama-X.

## Contents
1. [News](#news)

2. [Ten main research areas](#research-areas)

3. [Llama-X Model Version](#model)

4. [Llama-X Evaluation](#evaluation)

5. [Llama-X Paper List](#paper)

6. [Usage](#usage)

7. [How to contribute](#contribute)


<h2 id="news">News</h2>

We have completed the training of our first version of model (Llama-X 3.0.1 7B). Please experience our model in the [demo page](https://cedfabe00f0fcbde.gradio.app), and the data, code and model weights of different scales will be updated in this repo later.

<h2 id="research-areas">Ten main research areas</h2>

[1]. Research on `Instruction Tuning`
- [ ] instruction-following tuning

[2]. Research on `RLHF & RLAIF`
- [ ] fundamental RLHF 
- [ ] AI learning from AI

[3]. Research on `Data Quality`
- [ ] high quality data for pre-training, fine-tuning, user feedbacks, multi-modality, etc

[4]. Research on `Long Context Transformer`
- [ ] enable efficient transformers for long sequence (>30k)

[5]. Research on `Multi-modal (text + image) Modeling`
- [ ] text + image in; text out

[6]. Research on `Multilingual`
- [ ] comparable multilingual performance with English

[7]. Research on `Efficient infrastructure and optimization`
- [ ] improve training and inference speed
- [ ] build deep learning stack which scales predictably

[8]. Research on `Evaluation`
- [ ] comprehensive evaluation of model capabilities

[9]. Research on `Interpretability`
- [ ] interpret the source of each capability of LLM

[10]. Research on `LLM on Actions`
- [ ] combine LLM with search, recommendation and other plugins


<h2 id="model">Llama-X Model Version</h2>



| Llama-X       | Baseline         | Performance                        |
|---------------|------------------|------------------------------------|
| 3.0.0 (LLaMA) | GPT-3            | Outperform |
| 3.1.0         | text-davinci-001 | Comparable                         |
| 3.2.0         | text-davinci-002 | Comparable                         |
| 3.3.0         | text-davinci-003 | Comparable                         |
| 3.5.0         | gpt-35-turbo     | Comparable                         |
| 3.6.0         | GPT-4            | 80% Avg.Gap                        |
| 3.7.0         | GPT-4            | 60% Avg.Gap                        |
| 3.8.0         | GPT-4            | 40% Avg.Gap                        |
| 3.9.0         | GPT-4            | 20% Avg.Gap                        |
| 4.0.0         | GPT-4            | Comparable                         |


We are focusing on the above research areas [1] & [3] now, and would public our first version of model (Llama-X 3.0.1) and paper.



<h2 id="evaluation">Llama-X Evaluation</h2>

Each new version of Llama-X model should significantly outperform (+>1%) the current version model on the automatic evaluation 
of all the following Type-A benchmarks. And the additional evaluation for Type-B benchmarks should be added in the 3.6.0+ versions:

| Type | Benchmarks          |                       
|------|---------------------|
| A    | MMLU                | 
| A    | HumanEval           |
| A    | GSM-8K              | 
| A    | NaturalQuestions    |
| A    | TruthfulQA          | 
| B    | Leetcode            | 
| B    | GRE                 | 
| B    | AP                  | 
| B    | MMLU-Multilingual   |
| B    | Visual Inputs (TBD) |


Results:

| Model                        | MMLU   | TruthfulQA | GSM-8K | NaturalQuestions |
|------------------------------|--------|------------|--------|------------------|
|InstructGPT davinci v2 (175B)^ | 0.57   | 0.62       |  0.35  | 0.389            |
|Llama-X 3.0.1 (7B)            | 0.4412 | 0.2032     |  0.1887| 0.2422           |
|Llama-i (7B)                  | 0.5121 | 0.2142     |  0.2259| 0.3499           |

^ The results of `InstructGPT davinci v2 (175B)` are copied from [Stanford CRFM Benchmark](https://crfm.stanford.edu/).

<h2 id="paper">Llama-X Paper List</h2>

1. [LLaMA: Open and Efficient Foundation Language Models.](https://arxiv.org/abs/2302.13971v1)


<h2 id="usage">Usage</h2>

- Setup. Install the conda environment:
```bash
conda create -n llamax python=3.10
conda activate llamax
git clone https://github.com/AetherCortex/Llama-X.git
cd Llama-X/src
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install transformers==4.31.0
cd ../..
pip install -r requirements.txt
```

- Training data example (e.g., [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)):
```bash
Llama-X/src/data/alpaca_data.json
```

- Convert LLaMA checkpoint to HuggingFace format:
```bash
cd Llama-X/src
python transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/llama-7B/ \
    --model_size 7B \
    --output_dir /path/to/llama-7B/hf
```

- Train LLaMA-7B on DeepSpeed Zero-3:
```bash
deepspeed train.py \
    --model_name_or_path /path/to/llama-7B/hf \
    --data_path /path/to/example_data.json \
    --output_dir /path/to/llama-7B/hf/ft \
    --num_train_epochs 3 \
    --model_max_length 512 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --warmup_steps 2 \
    --logging_steps 2 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/deepspeed_config.json \
    --fp16 True
```
- Train LLaMA-7B on DeepSpeed Zero-3 with Multi-nodes
```bash
deepspeed --num_gpus num_of_gpus_in_each_node \
    --num_nodes num_of_nodes \
    --master_addr ip_address_of_main_node \
    --master_port 34545 \
    --hostfile configs/hostfile \
    train.py \
    --model_name_or_path /path/to/llama-7B/hf \
    --data_path /path/to/example_data.json \
    --output_dir /path/to/llama-7B/hf/ft \
    --num_train_epochs 3 \
    --model_max_length 512 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --warmup_steps 2 \
    --logging_steps 2 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/deepspeed_config.json \
    --fp16 True
```

- The current code of Llama-X support:
    - Fully Finetune: Optimize full LLaMA checkpoint, instead of `Low-Rank Adaptation (LoRA)`.
    - High Efficiency: Training 7B model with `50k examples/epoch` & `batch_size=64` within `1 hour` on `8 x V100 GPUs`.

| LLaMA  | Batch Size | V100s  | Time (h)    |
|--------|------------|--------|-------------|
| 7 B    | 64         | 8      | 1.00        |
| 13 B   | 32         | 8      | 2.00        |


- Inference
```bash
# web demo inference
python generate.py

# batch inference
To Do
```


<h2 id="contribute">How to contribute</h2>

Developers can become Contributors by contributing helpful code, data, paper and computing resource, etc.

1. Code: Including algorithm implementation, training optimization, inference optimization, and model deployment.

2. Data: Every [research area](#research-areas) and [version iteration](#model) requires high-quality data, including instruction-answer, pre-training, multi-modal, multilingual, and user feedbacks data, etc.

3. Paper: We will maintain a [Llama-X Paper List](#paper), and use Llama-X as the base model for optimized, fully tested, and significantly improved academic papers. You can check in to the Llama X Paper List.

4. Computing resource: We hope to help accelerate model iteration speed by coordinating redundant computing power from some developers or non-profit sponsorship from universities/enterprises.

<h2 id="communication">How to communicate with us</h2>

1. Github Issues

2. Email: llama-x@mail.com

3. Discord: <a href="https://discord.gg/2etwhe6GvU"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>


## Thanks For

This project has been inspired by multiple open source projects:

[Meta AI LLaMA](https://arxiv.org/abs/2302.13971v1)

[Huggingface Transformers Llama](https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama)

[Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) and [Alpaca-LoRA](https://github.com/tloen/alpaca-lora)


## Disclaimer

The use of resources(e.g., code, data and model weights) related to this project is limited to academic research and is prohibited for commercial purposes. The content generated by any model of Llama-X is subject to factors such as randomness and uncontrollability, and this project cannot guarantee its accuracy. This project does not assume any legal responsibility for the content of the model output, nor does it assume any responsibility for any losses that may arise from the use of related resources and output results.




