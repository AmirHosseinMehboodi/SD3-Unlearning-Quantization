# Direct Unlearning & Quantization Optimization for Stable Diffusion 3

This repository explores the intersection of **Machine Unlearning** and **Post-Training Quantization (PTQ)** on large-scale text-to-image generative models, specifically focusing on `stabilityai/stable-diffusion-3-medium-diffusers`. 

By applying Direct Preference Optimization (DPO) for unlearning unsafe concepts (e.g., nudity) and evaluating various quantization techniques, this project demonstrates how to maintain semantic integrity (CLIP score) while compressing model weights and preserving safety guardrails.

## Key Contributions
1. **Unlearning via DPO:** Fine-tuned SD3 using DPO (FP16 baseline) for 500 steps to suppress unsafe content generation.
2. **Comprehensive Quantization:** Evaluated 4 distinct quantization strategies on the unlearned model to analyze the trade-off between compression, safety (Nudity Generation Rate), and visual fidelity.
3. **Automated Evaluation Pipelines:** Integrated NudeNet for safety verification and CLIP-ViT for semantic alignment.

## 📁 Repository Structure
Based on the modular implementation, the repository is organized as follows:
- `Quantization_BitsAndBytes/`: Scripts for evaluating LLM.int8 (8-bit) and NF4 (4-bit) quantization.
- `Quantization_Dit/`: Implementations of PTQ4DiT (8-bit) and Q-DiT (4-bit dynamic group quantization).
- `benchmarkClip/`: Pipelines for evaluating semantic retention using `openai/clip-vit-base-patch32`.
- `evaluateLoraNGR/`: NudeNet evaluation scripts over 127 sensitive prompts.
- `calibration/`: Scripts for prompt extraction and channel variance calibration.

## ⚙️ Experimental Setup
All generated images across different quantization methods used the following fixed generation parameters:
- **Inference Steps:** 20
- **Guidance Scale:** 7.0
- **Resolution:** 512x512
- **Max Sequence Length:** 256
- **Seed:** 42

### Quantization Configurations
- **PTQ4DiT:** 8-bit Fake Quantization based on channel variance calibration.
- **Q-DiT:** 4-bit Dynamic Group Quantization (`Group Size = 64`) to minimize transformer block degradation.
- **LLM.int8 (BitsAndBytes):** Outlier-aware 8-bit quantization with `llm_int8_threshold=6.0` and `llm_int8_has_fp16_weight=False`.
- **NF4 (BitsAndBytes):** Standard 4-bit NormalFloat quantization with native `torch.float16` compute precision to prevent model collapse.

## 📊 Quantitative Results

The models were evaluated under two main criteria:
1. **Nudity Generation Rate (NGR):** Measured via NudeNet across 127 sensitive prompts (Threshold > 0.45 for 5 explicit classes). Lower is better.
2. **CLIP Score:** Measured using `openai/clip-vit-base-patch32` across 100 clean prompts to ensure semantic fidelity. Higher is better.

| Model     | Quantization Method     | Precision   | NudeNet Positive / 127 | Nudity Gen Rate ↓ | Mean CLIP ↑ |
| :-------- | :---------------------- | :---------- | :--------------------: | :---------------: | :---------: |
| Unlearned | None (Baseline)         | Full (FP16) |         26/127         |      20.47%       |    29.40    |
| Unlearned | PTQ4DiT                 | INT8        |         11/127         |       8.66%       |    29.28    |
| Unlearned | Q-DiT                   | INT4        |         6/127          |       4.72%       |    29.17    |
| Unlearned | LLM.int8 (BitsAndBytes) | INT8        |         21/127         |      16.54%       |      -      |
| Unlearned | NF4 (BitsAndBytes)      | INT4        |         12/127         |       9.45%       |      -      |

*Note: Interestingly, Q-DiT (INT4) provided the strongest safety guardrails (lowest NGR) with minimal CLIP degradation.*
