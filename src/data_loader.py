from datasets import load_dataset
import pandas as pd
import os
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

def load_ruslanmv_meddialog(n=100, output_path="data/medical_data.csv"):
    """
    Downloads the ruslanmv/ai-medical-chatbot dataset (MedDialog mirror).
    """
    print("Attempting to download ruslanmv/ai-medical-chatbot...")
    
    # Authenticate
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("Logging in to Hugging Face...")
        login(token=hf_token)
    else:
        print("Warning: HF_TOKEN not found in .env. Download might fail if dataset is gated.")

    try:
        # Load ruslanmv/ai-medical-chatbot
        print("Loading 'ruslanmv/ai-medical-chatbot'...")
        dataset = load_dataset("ruslanmv/ai-medical-chatbot", split="train")
        
        # Convert to pandas
        df = pd.DataFrame(dataset)
        print(f"Dataset loaded. Columns: {df.columns}")
        
        # Preprocessing
        normalized_df = pd.DataFrame()
        normalized_df['id'] = range(1, len(df) + 1)
        normalized_df['source'] = 'ruslanmv_ai_medical_chatbot'
        
        # Handle columns
        cols_lower = {c.lower(): c for c in df.columns}
        
        # Description
        if 'description' in cols_lower:
            normalized_df['description'] = df[cols_lower['description']]
        else:
            normalized_df['description'] = "No description"

        # Text (Patient + Doctor)
        if 'patient' in cols_lower and 'doctor' in cols_lower:
            pat_col = cols_lower['patient']
            doc_col = cols_lower['doctor']
            normalized_df['text'] = "Patient: " + df[pat_col] + "\n\nDoctor: " + df[doc_col]
        elif 'dialogue' in cols_lower:
             normalized_df['text'] = df[cols_lower['dialogue']]
        else:
            normalized_df['text'] = df.astype(str).agg('\n'.join, axis=1)

        # Sample
        sample_df = normalized_df.head(n)
        
        # Save
        os.makedirs("data", exist_ok=True)
        sample_df.to_csv(output_path, index=False)
        print(f"Saved {len(sample_df)} samples to {output_path}")
        print(f"Columns: {sample_df.columns.tolist()}")
        return output_path
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

if __name__ == "__main__":
    load_ruslanmv_meddialog()
