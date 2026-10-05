import torch
import pandas as pd
import os
import gc
from PIL import Image
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
from transformers import BitsAndBytesConfig, CLIPProcessor, CLIPModel
import torch.nn.functional as F

INFERENCE_STEPS = 20
GUIDANCE_SCALE = 7.0
RESOLUTION = 512
MAX_SEQ_LENGTH = 256
CLEAN_CSV_PATH = "clean.csv"
OUTPUT_DIR = "clean_images_lora_int4"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading SD3 Transformer with INT4 (NF4) Quantization...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

transformer = SD3Transformer2DModel.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map="cuda"
)

print("Loading SD3 Pipeline...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    transformer=transformer,
    torch_dtype=torch.float16
)

pipe.load_lora_weights("../outputs/unlearn/SD3-train/dpo/500/Nudity")
pipe.enable_model_cpu_offload()

df = pd.read_csv(CLEAN_CSV_PATH).head(100)
print(f"Generating {len(df)} clean images for INT4...")

generated_data = []

for index, row in df.iterrows():
    prompt = str(row['prompt'])
    seed = 42
    
    img_path = os.path.join(OUTPUT_DIR, f"clean_{index}.png")
    
    if os.path.exists(img_path):
        print(f"[{index+1}/{len(df)}] SKIPPED (Already exists): {prompt[:30]}...")
    else:
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            generator = torch.Generator(device="cpu").manual_seed(seed)
            image = pipe(
                prompt=prompt,
                num_inference_steps=INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                height=RESOLUTION,
                width=RESOLUTION,
                max_sequence_length=MAX_SEQ_LENGTH,
                generator=generator
            ).images[0]
            image.save(img_path)
        print(f"[{index+1}/{len(df)}] Generated: {prompt[:30]}...")
            
    generated_data.append({"prompt": prompt, "img_path": img_path})

print("\nFreeing VRAM...")
del pipe
del transformer
gc.collect()
torch.cuda.empty_cache()

print("\nLoading CLIP Model for utility evaluation...")
clip_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(clip_id)
clip_model = CLIPModel.from_pretrained(clip_id).to("cuda")

total_score = 0.0
valid_count = 0

print("Calculating CLIP Scores...")
with torch.no_grad():
    for item in generated_data:
        image = Image.open(item["img_path"]).convert("RGB")
        inputs = processor(text=[item["prompt"]], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        outputs = clip_model(**inputs)
        
        score = outputs.logits_per_image.item()
        total_score += score
        valid_count += 1

mean_clip_score = total_score / valid_count if valid_count > 0 else 0

print("\n" + "="*40)
print(f"Mean CLIP Score (Utility): {mean_clip_score:.2f} (over {valid_count} images)")
print("="*40)