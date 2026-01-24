# Medical Dialogue Model Tutorial for Roo

This tutorial demonstrates how to fine-tune a medical dialogue model using the Qwen2.5-3B-Instruct base model. The process involves data preparation, supervised fine-tuning (SFT), and group relative policy optimization (GRPO) to build a model that generates medical consultation responses.

## Overview

The tutorial is structured as a complete pipeline that takes you from raw data to a production-ready model. You'll start by downloading and processing medical dialogue datasets, then apply two training techniques: SFT for basic adaptation and GRPO for quality refinement. Throughout this process, you'll learn how modern language model training works in practice.

The project uses 4-bit quantization and LoRA (Low-Rank Adaptation) to make training feasible on consumer GPUs. These techniques allow you to fine-tune a 3-billion parameter model with just 2 GPUs, which would otherwise require significantly more resources.

## Getting Started

Before running any scripts, activate the conda environment that contains all necessary dependencies:

```bash
conda activate tutorial
cd /Data/czhaobo/metattt/ref/tutorial/src
```

The environment includes PyTorch, Transformers, TRL (Transformer Reinforcement Learning), and evaluation libraries like ROUGE and NLTK.

## Step 1: Download Datasets

The first step is obtaining medical dialogue data. We use two complementary datasets: a single-turn QA dataset for straightforward questions and answers, and a multi-turn dataset for complex consultations.

```bash
python 0_download_dataset.py
```

This script downloads samples from HuggingFace datasets. The single-turn dataset (`medical_meadow_medical_flashcards`) contains medical flashcard-style Q&A pairs, while the multi-turn dataset (`ChatDoctor-HealthCareMagic-100k`) includes realistic patient-doctor conversations. Both are saved locally in `../data/raw/` for processing.

If the download is slow, the script automatically uses a Hugging Face mirror (`hf-mirror.com`). It only downloads a subset of each dataset (1000 and 500 samples, respectively) to keep things manageable for tutorial purposes.

## Step 2: Process and Standardize Data

Raw medical data comes in various formats, so we need to standardize it into a unified structure that our model can understand.

```bash
python 1_process_dataset.py
```

This processing step converts all data into a consistent format with clear role labels (system, user, assistant). For single-turn data, it transforms simple input-output pairs into conversation format. For multi-turn data, it parses complex dialogue structures and ensures proper alternation between patient and doctor responses.

The script also splits your data into training (80%), validation (10%), and test (10%) sets. This is crucial for proper model evaluation - you never want to test on data the model has seen during training. The processed data is saved in `../data/processed/` with clear separation between datasets.

The processing script has been updated to also generate Parquet files (saved in `../data/verl_format/`). These files contain pre-formatted prompts and are specifically required for the advanced VERL training pipeline described in Step 6.

## Step 3: Test the Base Model

Before training, it's valuable to evaluate the base model's performance on medical questions. This gives you a baseline for comparison.

```bash
python 2_inference.py --model_path ../model/Qwen2.5-3B-Instruct \
                      --test_data ../data/processed/medical_meadow/test.json \
                      --num_samples 50
```

The inference script loads the model, runs predictions on your test set, and computes metrics such as BLEU and ROUGE. These metrics measure how similar the model's responses are to reference answers. Don't worry if the base model's scores are low - that's precisely why we need fine-tuning!

You can also use this script later to evaluate your trained models by changing the `--model_path` to point to your fine-tuned checkpoints.

## Step 4: Supervised Fine-Tuning (SFT)

Now comes the core training. SFT teaches the model to generate responses in the style and format of medical consultations.

```bash
python 3_train_sft.py --train_data ../data/processed/medical_meadow/train.json \
                      --epochs 3 \
                      --batch_size 4 \
                      --learning_rate 2e-4
```

During SFT, the model learns by comparing its predictions against the ground truth responses in your dataset. The training uses cross-entropy loss, which measures how surprised the model is by the correct answer. Over multiple epochs, the model adjusts its parameters to reduce this surprise and improve its ability to predict medical responses. Training will take some time, depending on your dataset size. You'll see a progress bar indicating that loss decreases over time. When it finishes, your model is saved in `../output/trained_model/` with a timestamp.

## Step 5: Advanced Distributed Training with VERL (Optional)

For users interested in large-scale distributed training (e.g., using FSDP, Megatron, or multi-node setups), we have integrated **Volcengine VERL**. This framework offers better scalability and configuration management via Hydra.

**A. Run SFT with VERL**
This uses the configuration defined in `config/sft.yaml` and executes the FSDP SFT trainer.

```bash
python 5_train_sft_verl.py
、、、

B. Run GRPO with VERL This uses config/grpo.yaml and our custom reward functions defined in src/reward_utils.py. Unlike the simple script in Step 5, this version is ready for scaling across multiple GPUs using Ray.

、、、bash
# Run with default settings (uses 8 GPUs per node by default in config)
python 6_train_grpo_verl.py

# Override config from command line (e.g., use only 2 GPUs)
python 6_train_grpo_verl.py trainer.n_gpus_per_node=2
、、、

Key Differences from TRL:

Data: Uses Parquet files instead of JSON.

Config: All hyperparameters are managed in config/*.yaml files instead of command-line arguments.

Architecture: Decouples the Actor, Rollout, and Reward computation for efficiency.

## Step 6: Group Relative Policy Optimization (GRPO)

After SFT provides your model with basic medical knowledge, GRPO refines its responses to improve quality and coherence. This is an advanced technique from reinforcement learning.

```bash
python 4_train_grpo.py --train_data ../data/processed/medical_meadow/train_small.json \
                       --epochs 1 \
                       --batch_size 2 \
                       --num_generation 2 \
                       --max_completion_length 128
```

GRPO works differently from SFT. For each medical question, it generates multiple candidate responses and scores them using a reward function. The model then learns to favor high-reward responses. The reward function considers factors like response length, coherence, similarity to reference answers, and linguistic quality.

This script is more computationally intensive because it generates multiple candidates per question. Start with smaller datasets (like `train_small.json`) to verify everything works. The `num_generation` parameter controls how many candidates to generate; fewer candidates are faster but may yield less refined results.

One important note: you may notice that the "loss" value remains at 0.0 during GRPO training. This is actually normal! GRPO uses relative advantages rather than absolute loss values. Check the "reward" and "grad_norm" metrics instead; they show the model is learning.

## Understanding the Configuration

The `config.py` file centralizes all settings. You can modify parameters there instead of passing command-line arguments every time. Key settings include:

The GPU configuration restricts training to 2 GPUs to avoid conflicts with others sharing the server. Model paths point to local copies to speed up loading. Training hyperparameters such as learning rate and batch size are tuned for the Qwen2.5-3B model, but can be adjusted based on your hardware.

## Evaluating Results

After training, compare your fine-tuned model against the base model:

```bash
# Test base model
python 2_inference.py --model_path ../model/Qwen2.5-3B-Instruct

# Test SFT model
python 2_inference.py --model_path ../output/trained_model/TIMESTAMP

# Test GRPO model  
python 2_inference.py --model_path ../output/trained_model_grpo/TIMESTAMP

# Test VERL SFT model (checkpoints are saved in the configured path)
python 2_inference.py --model_path ../checkpoints/sft/final_model

# Test VERL GRPO model
python 2_inference.py --model_path ../checkpoints/grpo/final_model
```

Look for improvements in ROUGE-L scores (measuring response relevance) and in qualitative factors such as response coherence and medical accuracy. The SFT model should show significant improvement over the base model, while GRPO provides more subtle refinements in response quality.

## Troubleshooting

If you encounter CUDA out-of-memory errors, try reducing the batch size or using a smaller `max_completion_length`. For GRPO specifically, decreasing `num_generation` from 4 to 2 significantly reduces memory usage.

The scripts restrict GPU usage to devices 0 and 1. If you need to use different GPUs, modify the `CUDA_VISIBLE_DEVICES` environment variable at the top of each training script.

## Next Steps

Once you have a trained model, you can deploy it for inference, integrate it into applications, or continue training on more specialized medical data. The model checkpoints are compatible with the standard HuggingFace Transformers library, making them easy to use in other projects.

Consider experimenting with different hyperparameters, trying larger datasets, or adjusting the GRPO reward function to emphasize different aspects of response quality. The field of medical AI is rapidly evolving, and this tutorial provides a foundation for more advanced work.


*This tutorial is designed for educational purposes*

