import torch
import pandas as pd
import os
import gc
from diffusers import StableDiffusion3Pipeline

CLEAN_CSV_PATH = "clean.csv"
NUDE_CSV_PATH = "nudity_benchmark_127_prompts.csv"
STATS_PATH = "calibration_stats.pt"

ACTIVE_MODE = "INT4"

@torch.no_grad()
def fake_quantize(weight, bits, group_size=None):
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1
    
    if group_size:
        original_shape = weight.shape
        if weight.numel() % group_size != 0:
            group_size = weight.shape[-1]
            
        weight = weight.reshape(-1, group_size)
        scale = (weight.abs().max(dim=-1, keepdim=True)[0] / qmax).clamp(min=1e-5).to(weight.dtype)
        weight_q = torch.round(weight / scale).clamp(qmin, qmax)
        return (weight_q * scale).reshape(original_shape)
    else:
        scale = (weight.abs().max(dim=-1, keepdim=True)[0] / qmax).clamp(min=1e-5).to(weight.dtype)
        weight_q = torch.round(weight / scale).clamp(qmin, qmax)
        return weight_q * scale

@torch.no_grad()
def apply_ptq4dit_int8(pipe, stats):
    print("Applying PTQ4DiT Channel-wise Salience Balancing (INT8)...")
    for name, module in pipe.transformer.named_modules():
        if isinstance(module, torch.nn.Linear) and name in stats:
            act_max = stats[name]['channel_max'].to(module.weight.device, dtype=module.weight.dtype)
            w_max = module.weight.abs().max(dim=1)[0]
            
            scale = ((act_max * w_max).sqrt() / (act_max + 1e-5)).unsqueeze(1).to(module.weight.dtype)
            
            module.weight.data = module.weight.data * scale
            module.weight.data = fake_quantize(module.weight.data, bits=8)
            module.weight.data = module.weight.data / scale
            
            gc.collect()

@torch.no_grad()
def apply_qdit_int4(pipe, stats):
    print("Applying Q-DiT Dynamic Group Quantization (INT4)...")
    for name, module in pipe.transformer.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data = fake_quantize(module.weight.data, bits=4, group_size=64)
            gc.collect()

if not os.path.exists(STATS_PATH):
    raise FileNotFoundError(f"Calibration stats not found at {STATS_PATH}. Run Phase 1 first.")

if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"STARTING PROCESS FOR {ACTIVE_MODE}")
    print(f"{'='*50}")
    
    out_dir_clean = f"images_clean_{ACTIVE_MODE}"
    out_dir_nude = f"images_nude_{ACTIVE_MODE}"
    os.makedirs(out_dir_clean, exist_ok=True)
    os.makedirs(out_dir_nude, exist_ok=True)
    
    print(f"Loading fresh SD3 + LoRA for {ACTIVE_MODE}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16
    )
    pipe.load_lora_weights("../outputs/unlearn/SD3-train/dpo/500/Nudity")
    
    stats = torch.load(STATS_PATH, map_location="cpu")
    
    if ACTIVE_MODE == "INT8":
        apply_ptq4dit_int8(pipe, stats)
    elif ACTIVE_MODE == "INT4":
        apply_qdit_int4(pipe, stats)
        
    del stats
    gc.collect()
        
    pipe.enable_model_cpu_offload()
    
    df_clean = pd.read_csv(CLEAN_CSV_PATH)
    df_nude = pd.read_csv(NUDE_CSV_PATH)
    
    print(f"\n---> Generating CLEAN images under {ACTIVE_MODE}...")
    for index, row in df_clean.iterrows():
        prompt = str(row['prompt'])
        img_path = os.path.join(out_dir_clean, f"clean_{index}.png")
        
        if os.path.exists(img_path):
            print(f"[{ACTIVE_MODE} CLEAN - {index+1}/{len(df_clean)}] SKIPPED.")
            continue
            
        with torch.no_grad():
            generator = torch.Generator(device="cpu").manual_seed(42)
            image = pipe(
                prompt=prompt, num_inference_steps=20, guidance_scale=7.0, 
                height=512, width=512, generator=generator
            ).images[0]
            image.save(img_path)
        print(f"[{ACTIVE_MODE} CLEAN - {index+1}/{len(df_clean)}] Generated.")
        gc.collect()
        torch.cuda.empty_cache()
        
    print(f"\n---> Generating NUDE images under {ACTIVE_MODE}...")
    for index, row in df_nude.iterrows():
        prompt = str(row['prompt'])
        img_path = os.path.join(out_dir_nude, f"nude_{index}.png")
        
        if os.path.exists(img_path):
            print(f"[{ACTIVE_MODE} NUDE - {index+1}/{len(df_nude)}] SKIPPED.")
            continue
            
        with torch.no_grad():
            generator = torch.Generator(device="cpu").manual_seed(42)
            image = pipe(
                prompt=prompt, num_inference_steps=20, guidance_scale=7.0, 
                height=512, width=512, generator=generator
            ).images[0]
            image.save(img_path)
        print(f"[{ACTIVE_MODE} NUDE - {index+1}/{len(df_nude)}] Generated.")
        gc.collect()
        torch.cuda.empty_cache()
        
    print(f"\nProcess finished for {ACTIVE_MODE}. You can now change ACTIVE_MODE and run again.")