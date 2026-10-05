import torch
import json
import gc
from diffusers import StableDiffusion3Pipeline

CALIBRATION_PROMPTS_PATH = "calibration_prompts.json"
OUTPUT_STATS_PATH = "calibration_stats.pt"

with open(CALIBRATION_PROMPTS_PATH, "r", encoding="utf-8") as f:
    prompts = json.load(f)

activation_stats = {}

def get_activation_hook(name):
    def hook(model, input, output):
        if isinstance(output, torch.Tensor):
            out = output.detach().float().cpu()
            
            channel_max = out.abs().max(dim=0)[0].max(dim=0)[0]
            token_var = out.var(dim=-1).mean(dim=0)
            
            if name not in activation_stats:
                activation_stats[name] = {'channel_max': channel_max, 'token_var': token_var, 'count': 1}
            else:
                activation_stats[name]['channel_max'] = torch.max(activation_stats[name]['channel_max'], channel_max)
                activation_stats[name]['token_var'] += token_var
                activation_stats[name]['count'] += 1
    return hook

print("Loading SD3 + LoRA for Calibration Phase...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16
)
pipe.load_lora_weights("../outputs/unlearn/SD3-train/dpo/500/Nudity")
pipe.enable_model_cpu_offload()

print("Attaching monitoring hooks to Transformer Linear layers...")
hooks = []
for name, module in pipe.transformer.named_modules():
    if isinstance(module, torch.nn.Linear):
        hooks.append(module.register_forward_hook(get_activation_hook(name)))

print(f"Starting forward passes for {len(prompts)} prompts...")
with torch.no_grad():
    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] Calibrating on prompt: {prompt[:30]}...")
        _ = pipe(
            prompt=prompt,
            num_inference_steps=20,
            guidance_scale=7.0,
            height=512,
            width=512,
            output_type="latent" 
        )
        gc.collect()
        torch.cuda.empty_cache()

for name in activation_stats:
    activation_stats[name]['token_var'] /= activation_stats[name]['count']

for h in hooks:
    h.remove()

torch.save(activation_stats, OUTPUT_STATS_PATH)
print(f"\nSuccess! Calibration statistics saved to {OUTPUT_STATS_PATH}")