import json
import numpy as np
import re
import subprocess
import tempfile
import os
from datetime import datetime
import pandas as pd

def write_code_to_temp_file(code, prefix="code"):
    """Write code to a temporary Java file and return the file path"""
    # Create a temporary file with .java extension
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.java', prefix=prefix, delete=False)
    temp_file.write(code)
    temp_file.close()
    return temp_file.name

def run_readability_tool(file_path):
    """Run the readability tool on a Java file and return the score"""
    try:
        # Run the readability tool
        result = subprocess.run(
            ['java', '-jar', 'rsm.jar', file_path],
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        
        if result.returncode != 0:
            print(f"Warning: Readability tool failed for {file_path}: {result.stderr}")
            return None
        
        # Parse the output to extract readability score
        output_lines = result.stdout.strip().split('\n')
        
        # Look for the readability score in the output
        # The tool typically outputs something like "Readability: X.XXX"
        for line in output_lines:
            if 'readability' in line.lower() or 'score' in line.lower():
                # Try to extract numeric value
                match = re.search(r'(\d+\.?\d*)', line)
                if match:
                    return float(match.group(1))
        
        # If no specific readability line found, try to parse the last line as a number
        if output_lines:
            try:
                return float(output_lines[-1].strip())
            except ValueError:
                pass
        
        print(f"Warning: Could not parse readability score from output: {result.stdout}")
        return None
        
    except subprocess.TimeoutExpired:
        print(f"Warning: Readability tool timed out for {file_path}")
        return None
    except Exception as e:
        print(f"Error running readability tool on {file_path}: {e}")
        return None

def clean_code_from_strategy(code_str):
    """Clean code from strategy output, removing markdown and extra text"""
    # Handle None values
    if code_str is None:
        return ""
    
    # Remove markdown code blocks
    code_str = re.sub(r'```java\s*', '', code_str)
    code_str = re.sub(r'```\s*$', '', code_str)
    
    # Remove common prefixes
    prefixes_to_remove = [
        'reverse engineer code:"',
        'Step 1 - Analyze Program Purpose:',
        'Step 2 - Identify Variable Roles:',
        'Step 3 - Apply Meaningful Names:',
        'Step 4 - Final Deobfuscated Code:',
        'Final code with variables renamed:'
    ]
    
    for prefix in prefixes_to_remove:
        if code_str.startswith(prefix):
            code_str = code_str[len(prefix):].strip()
    
    # Remove reasoning process text
    if 'reasoning_process' in code_str.lower():
        # Try to extract only the code part
        code_match = re.search(r'```java\s*(.*?)\s*```', code_str, re.DOTALL)
        if code_match:
            code_str = code_match.group(1)
    
    return code_str.strip()

def load_reference_data(file_path):
    """Load reference data from JSON file"""
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

def load_strategies_data(file_path):
    """Load strategies data from JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found.")
        return []
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def write_report_to_file(results, output_file):
    """Write the complete report to a text file"""
    with open(output_file, 'w') as f:
        # Write header
        f.write("="*120 + "\n")
        f.write("Readability Comparison: Reference vs Prompting Strategies\n")
        f.write("="*120 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*120 + "\n\n")
        
        # Write results table
        f.write("Readability Comparison Results\n")
        f.write("="*120 + "\n")
        f.write(f"{'Subdirectory':<25} {'Default':<12} {'Few_Shot':<12} {'CoT':<12} {'Aug_Reasoning':<12}\n")
        f.write("-"*120 + "\n")
        
        for result in results:
            f.write(f"{result['subdirectory']:<25} "
                   f"{result.get('default_score', 'N/A'):<12} "
                   f"{result.get('few_shot_score', 'N/A'):<12} "
                   f"{result.get('cot_score', 'N/A'):<12} "
                   f"{result.get('augmented_reasoning_score', 'N/A'):<12}\n")
        
        # Calculate and write summary statistics table
        f.write("\n" + "="*120 + "\n")
        f.write("Summary Statistics Table\n")
        f.write("="*120 + "\n")
        
        if results:
            strategies = ['default', 'few_shot', 'cot', 'augmented_reasoning']
            strategy_names = ['Default', 'Few_Shot', 'CoT', 'Augmented_Reasoning']
            
            # Create header for statistics table
            f.write(f"{'Strategy':<20} {'Average':<12} {'Median':<12} {'Min':<12} {'Max':<12} {'Std Dev':<12} {'Count':<8}\n")
            f.write("-"*120 + "\n")
            
            for i, strategy in enumerate(strategies):
                scores = [r[f'{strategy}_score'] for r in results if f'{strategy}_score' in r]
                if scores:
                    avg = np.mean(scores)
                    median = np.median(scores)
                    min_val = min(scores)
                    max_val = max(scores)
                    std_dev = np.std(scores)
                    count = len(scores)
                    
                    f.write(f"{strategy_names[i]:<20} "
                           f"{avg:<12.4f} "
                           f"{median:<12.4f} "
                           f"{min_val:<12.4f} "
                           f"{max_val:<12.4f} "
                           f"{std_dev:<12.4f} "
                           f"{count:<8}\n")
            
            # Summary of what each strategy represents
            f.write(f"\n" + "="*120 + "\n")
            f.write("Strategy Descriptions:\n")
            f.write("="*120 + "\n")
            f.write("Default: Basic prompting without special techniques\n")
            f.write("Few_Shot: Prompting with examples\n")
            f.write("CoT: Chain-of-Thought reasoning approach\n")
            f.write("Augmented_Reasoning: Enhanced reasoning with additional context\n")
            f.write("Higher scores indicate better readability.\n")

def create_dataframe(results):
    """Create a pandas DataFrame for easy analysis"""
    df_data = []
    for result in results:
        row = {
            'subdirectory': result['subdirectory']
        }
        # Only add scores that exist
        for strategy in ['default', 'few_shot', 'cot', 'augmented_reasoning']:
            if f'{strategy}_score' in result:
                row[f'{strategy}_score'] = result[f'{strategy}_score']
        
        df_data.append(row)
    
    return pd.DataFrame(df_data)

def main():
    # Define file paths
    reference_file = '/Users/lionel.nikiema/Documents/Thesis/Project2_FineTuning/Reversibility_Hypothesis/Reversibity_Hypothesis/Obfuscated_code _generation/Evaluation_dataset/least_300_VarRen_Obfuscator_Test.json'
    strategies_file = 'prompting_strategy.json'
    
    # Check if rsm.jar exists
    if not os.path.exists('rsm.jar'):
        print("Error: rsm.jar not found in current directory.")
        print("Please ensure rsm.jar is in the same directory as this script.")
        return
    
    # Load data
    print("Loading reference data...")
    reference_dict = load_reference_data(reference_file)
    
    print("Loading strategies data...")
    strategies_data = load_strategies_data(strategies_file)
    
    if not reference_dict:
        print("No reference data found. Please check the file path.")
        return
    
    if not strategies_data:
        print("No strategies data found. Please check the file path.")
        return
    
    # Find common subdirectories
    reference_subdirs = set(reference_dict.keys())
    strategy_subdirs = set(item['subdirectory'] for item in strategies_data)
    common_subdirs = reference_subdirs & strategy_subdirs
    
    if not common_subdirs:
        print("No common subdirectories found between reference and strategies data.")
        return
    
    print(f"Found {len(common_subdirs)} common subdirectories.")
    
    # Calculate and store results
    results = []
    skipped_count = 0
    temp_files = []  # Keep track of temp files to clean up
    
    try:
        for subdir in common_subdirs:
            reference_code = reference_dict[subdir]
            
            # Find corresponding strategy data
            strategy_item = next((item for item in strategies_data if item['subdirectory'] == subdir), None)
            
            if not strategy_item:
                continue
            
            result = {'subdirectory': subdir}
            
            # Calculate readability for each strategy
            strategies = ['default', 'few_shot', 'cot', 'augmented_reasoning']
            available_strategies = []
            
            for strategy in strategies:
                if strategy in strategy_item['strategies']:
                    strategy_code = strategy_item['strategies'][strategy]['deobfuscated_code']
                    if strategy_code is None:
                        # Skip this strategy if the code is None
                        continue
                    cleaned_code = clean_code_from_strategy(strategy_code)
                    
                    if cleaned_code:
                        # Write code to temporary file
                        temp_file = write_code_to_temp_file(cleaned_code, f"{subdir}_{strategy}")
                        temp_files.append(temp_file)
                        
                        # Calculate readability score
                        score = run_readability_tool(temp_file)
                        
                        if score is not None:
                            result[f'{strategy}_score'] = score
                            available_strategies.append(strategy)
                        else:
                            # Skip this strategy if score couldn't be computed
                            continue
                    else:
                        # Skip this strategy if cleaned code is empty
                        continue
                else:
                    # Skip this strategy if it doesn't exist in the data
                    continue
            
            # Only add result if at least one strategy was computed
            if any(f'{s}_score' in result for s in strategies):
                results.append(result)
                print(f"  {subdir}: {len(available_strategies)} strategies available ({', '.join(available_strategies)})")
            else:
                skipped_count += 1
                print(f"  {subdir}: No valid strategies found")
    
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
    
    # Sort results by average score
    def avg_score(result):
        scores = [result.get(f'{s}_score', 0) for s in strategies if f'{s}_score' in result]
        return np.mean(scores) if scores else 0
    
    results.sort(key=avg_score, reverse=True)
    
    print(f"Processed {len(results)} subdirectories with valid strategies.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} subdirectories with no valid strategies.")
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"strategies_readability_comparison_{timestamp}.txt"
    
    # Write report to file
    write_report_to_file(results, output_file)
    
    # Create and save DataFrame
    df = create_dataframe(results)
    csv_file = f"strategies_readability_comparison_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    
    # Print results to console
    print("\n" + "="*120)
    print("Readability Comparison Results")
    print("="*120)
    print(f"{'Subdirectory':<25} {'Default':<12} {'Few_Shot':<12} {'CoT':<12} {'Aug_Reasoning':<12}")
    print("-"*120)
    
    for result in results:
        print(f"{result['subdirectory']:<25} "
              f"{result.get('default_score', 'N/A'):<12} "
              f"{result.get('few_shot_score', 'N/A'):<12} "
              f"{result.get('cot_score', 'N/A'):<12} "
              f"{result.get('augmented_reasoning_score', 'N/A'):<12}")
    
    # Calculate and print statistics
    print("\n" + "="*120)
    print("Summary Statistics Table")
    print("="*120)
    
    if results:
        strategies = ['default', 'few_shot', 'cot', 'augmented_reasoning']
        strategy_names = ['Default', 'Few_Shot', 'CoT', 'Augmented_Reasoning']
        
        # Create header for statistics table
        print(f"{'Strategy':<20} {'Average':<12} {'Median':<12} {'Min':<12} {'Max':<12} {'Std Dev':<12} {'Count':<8}")
        print("-"*120)
        
        for i, strategy in enumerate(strategies):
            scores = [r[f'{strategy}_score'] for r in results if f'{strategy}_score' in r]
            if scores:
                avg = np.mean(scores)
                median = np.median(scores)
                min_val = min(scores)
                max_val = max(scores)
                std_dev = np.std(scores)
                count = len(scores)
                
                print(f"{strategy_names[i]:<20} "
                      f"{avg:<12.4f} "
                      f"{median:<12.4f} "
                      f"{min_val:<12.4f} "
                      f"{max_val:<12.4f} "
                      f"{std_dev:<12.4f} "
                      f"{count:<8}")
        
        # Summary
        print(f"\n" + "="*120)
        print("Summary:")
        print("="*120)
        print("Higher scores indicate better readability.")
        print("Each strategy represents a different prompting approach:")
        print("- Default: Basic prompting without special techniques")
        print("- Few_Shot: Prompting with examples")
        print("- CoT: Chain-of-Thought reasoning approach")
        print("- Augmented_Reasoning: Enhanced reasoning with additional context")
    
    print(f"\nReport saved to: {output_file}")
    print(f"CSV data saved to: {csv_file}")

if __name__ == "__main__":
    main() 