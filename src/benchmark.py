"""
Benchmark comparison script for the clinical summarization pipeline.
Compares our 4-agent pipeline against baseline approaches.
"""

import json
import pandas as pd
from src.evaluation import calculate_rouge

def load_batch_results(filepath="data/batch_results.json"):
    """Load processed batch results."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'results' in data:
        return data['results'], data.get('metrics', {})
    return data, {}

def baseline_extractive_summary(text, max_sentences=3):
    """
    Baseline 1: Simple extractive summarization.
    Takes first N sentences from the conversation.
    """
    sentences = text.split('.')
    summary = '. '.join(sentences[:max_sentences]) + '.'
    return summary

def baseline_template_summary(text):
    """
    Baseline 2: Template-based summary.
    Simple rule-based approach without AI.
    """
    return f"Patient presented with medical concerns. Conversation documented. Further review recommended."

def run_benchmark():
    """Compare our pipeline against baselines."""
    print("=" * 60)
    print("BENCHMARK COMPARISON: Clinical Summarization")
    print("=" * 60)
    
    # Load our results
    results, metrics = load_batch_results()
    
    if not results:
        print("Error: No batch results found. Run batch_processor first.")
        return
    
    # Calculate metrics for each approach
    our_rouge_scores = []
    baseline1_rouge_scores = []
    baseline2_rouge_scores = []
    
    for record in results:
        source = record.get('ai_output', {}).get('anonymized_text', '')
        our_summary = record.get('ai_output', {}).get('summary', '')
        
        if not source or not our_summary:
            continue
        
        # Our pipeline
        our_scores = calculate_rouge(source, our_summary)
        our_rouge_scores.append(our_scores)
        
        # Baseline 1: Extractive
        baseline1_summary = baseline_extractive_summary(source)
        baseline1_scores = calculate_rouge(source, baseline1_summary)
        baseline1_rouge_scores.append(baseline1_scores)
        
        # Baseline 2: Template
        baseline2_summary = baseline_template_summary(source)
        baseline2_scores = calculate_rouge(source, baseline2_summary)
        baseline2_rouge_scores.append(baseline2_scores)
    
    # Average scores
    def avg_scores(scores_list):
        if not scores_list:
            return {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
        return {
            'rouge1': sum(s['rouge1'] for s in scores_list) / len(scores_list),
            'rouge2': sum(s['rouge2'] for s in scores_list) / len(scores_list),
            'rougeL': sum(s['rougeL'] for s in scores_list) / len(scores_list)
        }
    
    our_avg = avg_scores(our_rouge_scores)
    baseline1_avg = avg_scores(baseline1_rouge_scores)
    baseline2_avg = avg_scores(baseline2_rouge_scores)
    
    # Display results
    print(f"\nTested on {len(results)} medical conversations\n")
    
    print("ROUGE-1 Scores:")
    print(f"  Our 4-Agent Pipeline:     {our_avg['rouge1']:.3f}")
    print(f"  Baseline (Extractive):    {baseline1_avg['rouge1']:.3f}")
    print(f"  Baseline (Template):      {baseline2_avg['rouge1']:.3f}")
    
    print("\nROUGE-2 Scores:")
    print(f"  Our 4-Agent Pipeline:     {our_avg['rouge2']:.3f}")
    print(f"  Baseline (Extractive):    {baseline1_avg['rouge2']:.3f}")
    print(f"  Baseline (Template):      {baseline2_avg['rouge2']:.3f}")
    
    print("\nROUGE-L Scores:")
    print(f"  Our 4-Agent Pipeline:     {our_avg['rougeL']:.3f}")
    print(f"  Baseline (Extractive):    {baseline1_avg['rougeL']:.3f}")
    print(f"  Baseline (Template):      {baseline2_avg['rougeL']:.3f}")
    
    # Improvement calculation
    print("\n" + "=" * 60)
    print("IMPROVEMENT OVER BASELINES")
    print("=" * 60)
    
    if baseline1_avg['rouge1'] > 0:
        improvement1 = ((our_avg['rouge1'] - baseline1_avg['rouge1']) / baseline1_avg['rouge1']) * 100
        print(f"vs Extractive Baseline: {improvement1:+.1f}% (ROUGE-1)")
    
    if baseline2_avg['rouge1'] > 0:
        improvement2 = ((our_avg['rouge1'] - baseline2_avg['rouge1']) / baseline2_avg['rouge1']) * 100
        print(f"vs Template Baseline:   {improvement2:+.1f}% (ROUGE-1)")
    
    # Save benchmark results
    benchmark_data = {
        "our_pipeline": our_avg,
        "baseline_extractive": baseline1_avg,
        "baseline_template": baseline2_avg,
        "num_samples": len(results)
    }
    
    with open("data/benchmark_results.json", "w") as f:
        json.dump(benchmark_data, f, indent=2)
    
    print(f"\n✅ Benchmark results saved to data/benchmark_results.json")
    print("=" * 60)
    
    return benchmark_data

if __name__ == "__main__":
    run_benchmark()
