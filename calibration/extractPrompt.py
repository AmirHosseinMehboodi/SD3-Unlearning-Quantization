import pandas as pd
import json
import os

CLEAN_CSV_PATH = "clean.csv"
CALIBRATION_OUTPUT_PATH = "calibration_prompts.json"
NUM_CALIBRATION_SAMPLES = 32
SEED = 42

def prepare_calibration_data():
    print(f"Loading prompts from {CLEAN_CSV_PATH}...")
    df = pd.read_csv(CLEAN_CSV_PATH)
    
    df = df.dropna(subset=['prompt'])
    
    calibration_df = df.sample(n=NUM_CALIBRATION_SAMPLES, random_state=SEED)
    
    calibration_prompts = calibration_df['prompt'].tolist()
    
    with open(CALIBRATION_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(calibration_prompts, f, indent=4)
        
    print(f"Successfully saved {len(calibration_prompts)} prompts to {CALIBRATION_OUTPUT_PATH}")
    print("Sample prompts:")
    for i, prompt in enumerate(calibration_prompts[:3]):
        print(f"  {i+1}. {prompt[:50]}...")

if __name__ == "__main__":
    if os.path.exists(CLEAN_CSV_PATH):
        prepare_calibration_data()
    else:
        print(f"Error: {CLEAN_CSV_PATH} not found in the current directory.")