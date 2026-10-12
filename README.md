# Direct Unlearning & Quantization Optimization for Stable Diffusion 3

This repository explores the intersection of **Machine Unlearning** and **Post-Training Quantization (PTQ)** on large-scale text-to-image generative models, specifically focusing on `stabilityai/stable-diffusion-3-medium-diffusers`. 

By applying Direct Preference Optimization (DPO) for unlearning unsafe concepts (e.g., nudity) and evaluating various quantization techniques, this project demonstrates how to maintain semantic integrity (CLIP score) while compressing model weights and preserving safety guardrails.

## 🚀 Key Contributions
1. **Unlearning via DPO:** Fine-tuned SD3 using DPO (FP16 baseline) for 500 steps to suppress unsafe content generation.
2. **Comprehensive Quantization:** Evaluated 4 distinct quantization strategies on the unlearned model to analyze the trade-off between compression, safety (Nudity Generation Rate), and visual fidelity.
3. **Automated Evaluation Pipelines:** Integrated NudeNet for safety verification and CLIP-ViT for semantic alignment.

## 💻 Hardware & Environment
- **Hardware Requirements:** The unlearning phase (DPO) and quantization calibration require significant VRAM. Experiments were conducted using an NVIDIA GPU environment (e.g., RTX 3090/4090 or A100).
- **Key Dependencies:** `torch>=2.0.0`, `diffusers>=0.30.0`, `bitsandbytes`, `peft`, `accelerate`.

## 🗄️ Dataset Generation
Unlike traditional pipelines relying on static external datasets, custom safe/unsafe paired datasets were **dynamically generated** using SD3 itself. The `datasets/SD3` scripts generate implicit unsafe concepts and their corresponding safe counterparts (e.g., via Negative Prompting and `NudeDetector` filtering in the loop) to create highly accurate preference pairs required for DPO training.

## 📁 Repository Structure
Based on the modular implementation, the repository is organized as follows:
- `Quantization_BitsAndBytes/`: Scripts for evaluating LLM.int8 (8-bit) and NF4 (4-bit) quantization.
- `Quantization_Dit/`: Implementations of PTQ4DiT (8-bit) and Q-DiT (4-bit dynamic group quantization).
- `benchmarkClip/`: Pipelines for evaluating semantic retention using `openai/clip-vit-base-patch32`.
- `evaluateLoraNGR/`: NudeNet evaluation scripts over 127 sensitive prompts.
- `calibration/`: Scripts for prompt extraction and channel variance calibration.
- `datasets/`: Automated generation of DPO preference pairs (Safe vs. Unsafe).
- `train/`: Core DPO unlearning scripts for SD/SD3.

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

## 💡 Discussion & Insights: Why did Quantization Improve Safety?

Recent literature highlights a critical vulnerability in Large Language Models (LLMs): applying quantization to unlearned LLMs often causes a catastrophic failure, restoring up to 83% of the supposedly "forgotten" knowledge [Zhang et al., 2024]. 

However, our empirical results on Stable Diffusion 3 demonstrate the exact opposite behavior in visual diffusion models. Instead of experiencing catastrophic un-forgetting, quantization significantly *decreased* the Nudity Generation Rate (e.g., dropping from 20.47% in the FP16 DUO baseline down to 4.72% in INT4 Q-DiT). We attribute this safety enhancement to three key architectural factors:

1. **Quantization as Regularization:** The rounding noise introduced by lower bit-widths acts as a strong regularizer. It restricts the model's representational capacity to generate highly detailed and complex anatomical textures, causing it to produce smoother, generalized images that easily pass safety filters.
2. **Clipping Outlier Activations:** Generating sensitive explicit concepts in diffusion transformers heavily relies on outlier activations. Because diffusion model quantization must manage variable data variance and bimodal distributions [Zeng et al., 2025], methods like PTQ4DiT (channel variance balancing) and LLM.int8 explicitly clip or restrict these extreme values. This effectively suppresses the specific activation pathways required for generating unsafe content.
3. **Manifold Collapse Towards Safe Regions:** The DUO framework utilizes preference optimization to shift the model away from unsafe visual features [Park et al., 2024]. When quantization restricts the overall capacity of the network, the model naturally collapses toward its most recently reinforced dominant distribution (the safe manifold), acting as an additional safety guardrail.
