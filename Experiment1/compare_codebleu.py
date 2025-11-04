import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import nltk
import re
from fractions import Fraction
from datetime import datetime

# Download required NLTK data
nltk.download('punkt')

def tokenize_for_bleu(code):
    # Tokenize code into tokens
    tokens = []
    # Split by whitespace and special characters
    for token in re.findall(r'\w+|[^\w\s]', code):
        tokens.append(token)
    return tokens

def calculate_codebleu(reference, candidate):
    # Tokenize both codes
    reference_tokens = tokenize_for_bleu(reference)
    candidate_tokens = tokenize_for_bleu(candidate)
    
    # Calculate BLEU score with modified smoothing
    smoothing = SmoothingFunction().method1
    
    # Calculate n-gram matches manually
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def get_precision(reference, candidate, n):
        ref_ngrams = get_ngrams(reference, n)
        cand_ngrams = get_ngrams(candidate, n)
        
        if not cand_ngrams:
            return 0.0
            
        matches = 0
        for ngram in cand_ngrams:
            if ngram in ref_ngrams:
                matches += 1
                ref_ngrams.remove(ngram)
                
        return matches / len(cand_ngrams)
    
    # Calculate precision for n-grams up to 4
    precisions = []
    for n in range(1, 5):
        precisions.append(get_precision(reference_tokens, candidate_tokens, n))
    
    # Calculate brevity penalty
    bp = 1.0
    if len(candidate_tokens) < len(reference_tokens):
        bp = np.exp(1 - len(reference_tokens) / len(candidate_tokens))
    
    # Calculate final score
    score = bp * np.exp(np.mean(np.log(precisions)))
    return score

def load_data_from_file(file_path):
    """Load data from JSON file and return dictionary with subdirectory as key"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return {item['subdirectory']: item['code'] for item in data}
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found. Using empty dictionary.")
        return {}
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def write_report_to_file(results, exact_matches, output_file):
    """Write the complete report to a text file"""
    with open(output_file, 'w') as f:
        # Write header
        f.write("="*120 + "\n")
        f.write("CodeBLEU Comparison Analysis Report\n")
        f.write("="*120 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*120 + "\n\n")
        
        # Write results table
        f.write("CodeBLEU Comparison Results (Excluding Exact Matches)\n")
        f.write("="*120 + "\n")
        f.write(f"{'Subdirectory':<25} {'S1(Cft,Corig)':<12} {'S2(Cft,Cbase)':<12} {'S3(Cbase,Corig)':<12} {'S4(Cft,Ctool)':<12}\n")
        f.write("-"*120 + "\n")
        
        for result in results:
            f.write(f"{result['subdirectory']:<25} "
                   f"{result['S1_codebleu_score']:<12.4f} "
                   f"{result['S2_codebleu_score']:<12.4f} "
                   f"{result['S3_codebleu_score']:<12.4f} "
                   f"{result['S4_codebleu_score']:<12.4f}\n")
        
        # Write statistics
        f.write("\n" + "="*120 + "\n")
        f.write("Statistics (Excluding Exact Matches)\n")
        f.write("="*120 + "\n")
        f.write(f"Number of Exact Matches Skipped: {len(exact_matches)}\n")
        f.write(f"Number of Non-Exact Matches Analyzed: {len(results)}\n")
        
        if results:
            # S1 Statistics (Cft vs Corig)
            s1_scores = [r['S1_codebleu_score'] for r in results]
            f.write(f"\nS1 (Cft vs Corig) - Fine-tuned output vs Original code:\n")
            f.write(f"  Average: {np.mean(s1_scores):.4f}\n")
            f.write(f"  Median:  {np.median(s1_scores):.4f}\n")
            f.write(f"  Min:     {min(s1_scores):.4f}\n")
            f.write(f"  Max:     {max(s1_scores):.4f}\n")
            f.write(f"  Std Dev: {np.std(s1_scores):.4f}\n")
            
            # S2 Statistics (Cft vs Cbase)
            s2_scores = [r['S2_codebleu_score'] for r in results]
            f.write(f"\nS2 (Cft vs Cbase) - Fine-tuned output vs Non-fine-tuned output:\n")
            f.write(f"  Average: {np.mean(s2_scores):.4f}\n")
            f.write(f"  Median:  {np.median(s2_scores):.4f}\n")
            f.write(f"  Min:     {min(s2_scores):.4f}\n")
            f.write(f"  Max:     {max(s2_scores):.4f}\n")
            f.write(f"  Std Dev: {np.std(s2_scores):.4f}\n")
            
            # S3 Statistics (Cbase vs Corig)
            s3_scores = [r['S3_codebleu_score'] for r in results]
            f.write(f"\nS3 (Cbase vs Corig) - Non-fine-tuned output vs Original code:\n")
            f.write(f"  Average: {np.mean(s3_scores):.4f}\n")
            f.write(f"  Median:  {np.median(s3_scores):.4f}\n")
            f.write(f"  Min:     {min(s3_scores):.4f}\n")
            f.write(f"  Max:     {max(s3_scores):.4f}\n")
            f.write(f"  Std Dev: {np.std(s3_scores):.4f}\n")
            
            # S4 Statistics (Cft vs Ctool)
            s4_scores = [r['S4_codebleu_score'] for r in results]
            f.write(f"\nS4 (Cft vs Ctool) - Fine-tuned output vs Tool-generated reference:\n")
            f.write(f"  Average: {np.mean(s4_scores):.4f}\n")
            f.write(f"  Median:  {np.median(s4_scores):.4f}\n")
            f.write(f"  Min:     {min(s4_scores):.4f}\n")
            f.write(f"  Max:     {max(s4_scores):.4f}\n")
            f.write(f"  Std Dev: {np.std(s4_scores):.4f}\n")
            
            # Summary of what each score measures
            f.write(f"\n" + "="*120 + "\n")
            f.write("Score Interpretation:\n")
            f.write("="*120 + "\n")
            f.write("S1 (Cft vs Corig): Measures how well fine-tuned model preserves original code structure\n")
            f.write("S2 (Cft vs Cbase): Measures improvement of fine-tuned over non-fine-tuned model\n")
            f.write("S3 (Cbase vs Corig): Measures baseline performance of non-fine-tuned model\n")
            f.write("S4 (Cft vs Ctool): Measures alignment with ground-truth transformations\n")
            f.write("Higher scores indicate greater syntactic similarity.\n")
        
        # Write exact matches if any
        if exact_matches:
            f.write(f"\nExact Matches (Skipped):\n")
            for match in exact_matches:
                f.write(f"- {match}\n")
        
        # Write detailed results for each subdirectory
        f.write(f"\n" + "="*120 + "\n")
        f.write("Detailed Results by Subdirectory\n")
        f.write("="*120 + "\n")
        for result in results:
            f.write(f"\nSubdirectory: {result['subdirectory']}\n")
            f.write(f"S1 (Cft vs Corig): {result['S1_codebleu_score']:.4f}\n")
            f.write(f"S2 (Cft vs Cbase): {result['S2_codebleu_score']:.4f}\n")
            f.write(f"S3 (Cbase vs Corig): {result['S3_codebleu_score']:.4f}\n")
            f.write(f"S4 (Cft vs Ctool): {result['S4_codebleu_score']:.4f}\n")
            f.write(f"Token counts - Corig: {result['token_counts']['corig_tokens']}, "
                   f"Cft: {result['token_counts']['cft_tokens']}, "
                   f"Cbase: {result['token_counts']['cbase_tokens']}, "
                   f"Ctool: {result['token_counts']['ctool_tokens']}\n")
            f.write("-" * 80 + "\n")

def main():
    # Define file paths for the four data sources
    # Corig: Original code
    corig_file = 'Obfuscated_code _generation/Evaluation_dataset/least_300_tokens.json'
    
    # Cft: Fine-tuned model output (using varRen as example)
    cft_file = '/Users/lionel.nikiema/Desktop/contrastive_results/gpt-3.5/deobfuscated/gpt-3.5-mini-VarRen_Deobfuscator_contrastive_strategy_cleaned.json'
    
    # Cbase: Non-fine-tuned model output
    cbase_file = '/Users/lionel.nikiema/Desktop/contrastive_results/gpt-3.5/obfuscated/gpt-3.5_obfuscated_VarRen_contrastive.json'
    
    # Ctool: Tool-generated reference obfuscation
    ctool_file = '/Users/lionel.nikiema/Documents/Thesis/Project2_FineTuning/Reversibility_Hypothesis/Reversibity_Hypothesis/Obfuscated_code _generation/Evaluation_dataset/least_300_VarRen_Obfuscator.jar.json'
    
    # Load all four data sources
    print("Loading data sources...")
    corig_dict = load_data_from_file(corig_file)
    cft_dict = load_data_from_file(cft_file)
    cbase_dict = load_data_from_file(cbase_file)
    ctool_dict = load_data_from_file(ctool_file)
    
    # Find common subdirectories across all four data sources
    all_subdirs = set(corig_dict.keys()) & set(cft_dict.keys()) & set(cbase_dict.keys()) & set(ctool_dict.keys())
    
    if not all_subdirs:
        print("No common subdirectories found across all four data sources.")
        return
    
    print(f"Found {len(all_subdirs)} common subdirectories across all data sources.")
    
    # Calculate and store results
    results = []
    exact_matches = []
    
    for subdir in all_subdirs:
        corig_code = corig_dict[subdir]
        cft_code = cft_dict[subdir]
        cbase_code = cbase_dict[subdir]
        ctool_code = ctool_dict[subdir]
        
        # Skip if any code is found exactly in another (exact match)
        if (corig_code in cft_code or corig_code in cbase_code or corig_code in ctool_code or
            cft_code in corig_code or cft_code in cbase_code or cft_code in ctool_code or
            cbase_code in corig_code or cbase_code in cft_code or cbase_code in ctool_code or
            ctool_code in corig_code or ctool_code in cft_code or ctool_code in cbase_code):
            exact_matches.append(subdir)
            continue
        
        # Calculate the four CodeBLEU scores
        # S1 = CodeBLEU(Cft, Corig) - Fine-tuned output vs Original code
        s1_score = calculate_codebleu(corig_code, cft_code)
        
        # S2 = CodeBLEU(Cft, Cbase) - Fine-tuned output vs Non-fine-tuned output
        s2_score = calculate_codebleu(cbase_code, cft_code)
        
        # S3 = CodeBLEU(Cbase, Corig) - Non-fine-tuned output vs Original code
        s3_score = calculate_codebleu(corig_code, cbase_code)
        
        # S4 = CodeBLEU(Cft, Ctool) - Fine-tuned output vs Tool-generated reference
        s4_score = calculate_codebleu(ctool_code, cft_code)
        
        results.append({
            'subdirectory': subdir,
            'S1_codebleu_score': s1_score,  # Cft vs Corig
            'S2_codebleu_score': s2_score,  # Cft vs Cbase
            'S3_codebleu_score': s3_score,  # Cbase vs Corig
            'S4_codebleu_score': s4_score,  # Cft vs Ctool
            'token_counts': {
                'corig_tokens': len(tokenize_for_bleu(corig_code)),
                'cft_tokens': len(tokenize_for_bleu(cft_code)),
                'cbase_tokens': len(tokenize_for_bleu(cbase_code)),
                'ctool_tokens': len(tokenize_for_bleu(ctool_code))
            }
        })
    
    # Sort results by S1 score (primary comparison)
    results.sort(key=lambda x: x['S1_codebleu_score'], reverse=True)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"codebleu_comparison_report_{timestamp}.txt"
    
    # Write report to file
    write_report_to_file(results, exact_matches, output_file)
    
    # Print results to console as well
    print("\n" + "="*120)
    print("CodeBLEU Comparison Results (Excluding Exact Matches)")
    print("="*120)
    print(f"{'Subdirectory':<25} {'S1(Cft,Corig)':<12} {'S2(Cft,Cbase)':<12} {'S3(Cbase,Corig)':<12} {'S4(Cft,Ctool)':<12}")
    print("-"*120)
    
    for result in results:
        print(f"{result['subdirectory']:<25} "
              f"{result['S1_codebleu_score']:<12.4f} "
              f"{result['S2_codebleu_score']:<12.4f} "
              f"{result['S3_codebleu_score']:<12.4f} "
              f"{result['S4_codebleu_score']:<12.4f}")
    
    # Calculate and print statistics for each score
    print("\n" + "="*120)
    print("Statistics (Excluding Exact Matches)")
    print("="*120)
    print(f"Number of Exact Matches Skipped: {len(exact_matches)}")
    print(f"Number of Non-Exact Matches Analyzed: {len(results)}")
    
    if results:
        # S1 Statistics (Cft vs Corig)
        s1_scores = [r['S1_codebleu_score'] for r in results]
        print(f"\nS1 (Cft vs Corig) - Fine-tuned output vs Original code:")
        print(f"  Average: {np.mean(s1_scores):.4f}")
        print(f"  Median:  {np.median(s1_scores):.4f}")
        print(f"  Min:     {min(s1_scores):.4f}")
        print(f"  Max:     {max(s1_scores):.4f}")
        print(f"  Std Dev: {np.std(s1_scores):.4f}")
        
        # S2 Statistics (Cft vs Cbase)
        s2_scores = [r['S2_codebleu_score'] for r in results]
        print(f"\nS2 (Cft vs Cbase) - Fine-tuned output vs Non-fine-tuned output:")
        print(f"  Average: {np.mean(s2_scores):.4f}")
        print(f"  Median:  {np.median(s2_scores):.4f}")
        print(f"  Min:     {min(s2_scores):.4f}")
        print(f"  Max:     {max(s2_scores):.4f}")
        print(f"  Std Dev: {np.std(s2_scores):.4f}")
        
        # S3 Statistics (Cbase vs Corig)
        s3_scores = [r['S3_codebleu_score'] for r in results]
        print(f"\nS3 (Cbase vs Corig) - Non-fine-tuned output vs Original code:")
        print(f"  Average: {np.mean(s3_scores):.4f}")
        print(f"  Median:  {np.median(s3_scores):.4f}")
        print(f"  Min:     {min(s3_scores):.4f}")
        print(f"  Max:     {max(s3_scores):.4f}")
        print(f"  Std Dev: {np.std(s3_scores):.4f}")
        
        # S4 Statistics (Cft vs Ctool)
        s4_scores = [r['S4_codebleu_score'] for r in results]
        print(f"\nS4 (Cft vs Ctool) - Fine-tuned output vs Tool-generated reference:")
        print(f"  Average: {np.mean(s4_scores):.4f}")
        print(f"  Median:  {np.median(s4_scores):.4f}")
        print(f"  Min:     {min(s4_scores):.4f}")
        print(f"  Max:     {max(s4_scores):.4f}")
        print(f"  Std Dev: {np.std(s4_scores):.4f}")
        
        # Summary of what each score measures
        print(f"\n" + "="*120)
        print("Score Interpretation:")
        print("="*120)
        print("S1 (Cft vs Corig): Measures how well fine-tuned model preserves original code structure")
        print("S2 (Cft vs Cbase): Measures improvement of fine-tuned over non-fine-tuned model")
        print("S3 (Cbase vs Corig): Measures baseline performance of non-fine-tuned model")
        print("S4 (Cft vs Ctool): Measures alignment with ground-truth transformations")
        print("Higher scores indicate greater syntactic similarity.")
    
    # Print list of exact matches if any
    if exact_matches:
        print(f"\nExact Matches (Skipped):")
        for match in exact_matches[:10]:  # Show first 10
            print(f"- {match}")
        if len(exact_matches) > 10:
            print(f"... and {len(exact_matches) - 10} more")
    
    print(f"\nReport saved to: {output_file}")

if __name__ == "__main__":
    main() 