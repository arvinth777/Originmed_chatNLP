import pandas as pd
import json
import os
import time
from tqdm import tqdm
from src.pipeline import ClinicalPipeline
from src.data_loader import load_ruslanmv_meddialog
from dotenv import load_dotenv

load_dotenv()

def process_batch(input_csv="data/medical_data.csv", output_file="data/batch_results.json", num_samples=10):
    print(f"🚀 Starting AI Batch Processing on {num_samples} records...")
    
    # 1. Load Data
    if not os.path.exists(input_csv):
        print(f"⚠️ Input file {input_csv} not found. Attempting to download...")
        load_ruslanmv_meddialog()
        
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"❌ Error: '{input_csv}' not found even after download attempt.")
        return

    # Initialize Pipeline
    pipeline = ClinicalPipeline()
    results = []
    
    # 2. Loop with Rate Limiting
    # We use .head(num_samples)
    for index, row in tqdm(df.head(num_samples).iterrows(), total=num_samples):
        
        transcript = row['text']
        
        try:
            # --- THE PIPELINE CALL ---
            pipeline_output = pipeline.run(transcript)
            
            record = {
                "id": row.get('id', index),
                "original_source": row.get('source', 'unknown'),
                "ai_output": pipeline_output
            }
            results.append(record)
            
            # --- CRITICAL FIX: RATE LIMIT SLEEP ---
            # 4 agents per record. 30 RPM limit with gemini-2.0-flash-lite.
            # We need to wait ~10 seconds to be safe.
            print("💤 Sleeping 10s to respect Gemini Free Tier limits...")
            time.sleep(10) 
            
        except Exception as e:
            print(f"⚠️ Error processing ID {index}: {e}")
            # If we hit a rate limit error, wait longer!
            if "429" in str(e) or "Quota exceeded" in str(e):
                print("🛑 Rate limit hit! Cooling down for 60 seconds...")
                time.sleep(60)

    # 3. Calculate Metrics
    from src.evaluation import calculate_rouge
    
    rouge_scores = []
    for record in results:
        ai_output = record.get('ai_output', {})
        summary = ai_output.get('summary', '')
        original_text = ai_output.get('anonymized_text', '')
        
        if summary and original_text:
            try:
                scores = calculate_rouge(original_text, summary)
                rouge_scores.append(scores)
            except Exception as e:
                print(f"⚠️ ROUGE calculation failed for record {record['id']}: {e}")
    
    # Calculate average ROUGE scores
    if rouge_scores:
        avg_rouge = {
            'rouge1': sum(s['rouge1'] for s in rouge_scores) / len(rouge_scores),
            'rouge2': sum(s['rouge2'] for s in rouge_scores) / len(rouge_scores),
            'rougeL': sum(s['rougeL'] for s in rouge_scores) / len(rouge_scores)
        }
        print(f"\n📊 Average ROUGE Scores:")
        print(f"   ROUGE-1: {avg_rouge['rouge1']:.3f}")
        print(f"   ROUGE-2: {avg_rouge['rouge2']:.3f}")
        print(f"   ROUGE-L: {avg_rouge['rougeL']:.3f}")
        
        # Add metrics to output
        metrics_summary = {
            "total_records": len(results),
            "successful_records": len([r for r in results if 'ai_output' in r]),
            "average_rouge_scores": avg_rouge
        }
        
        # Save results with metrics
        output_data = {
            "metrics": metrics_summary,
            "results": results
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
    else:
        # Save results without metrics if ROUGE failed
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
    print(f"✅ Finished! Results saved to {output_file}")

if __name__ == "__main__":
    # Check for API Key
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in .env file.")
    else:
        # Reduce this to 5 or 10 for testing
        process_batch(num_samples=5)
