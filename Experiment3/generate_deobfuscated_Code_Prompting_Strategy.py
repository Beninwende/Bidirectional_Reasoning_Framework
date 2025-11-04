import os
import json
from openai import OpenAI
from dotenv import load_dotenv 
import lizard
from transformers import AutoTokenizer
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def load_evaluation_dataset(file_path):
    """
    Load the evaluation dataset from JSON file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in dataset: {e}")

def calculate_complexity(code):
    """
    Calculate code complexity using Lizard
    """
    try:
        # Create a temporary file to analyze the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=True) as temp_file:
            temp_file.write(code)
            temp_file.flush()
            
            # Analyze the code using Lizard
            analysis = lizard.analyze_file(temp_file.name)
            
            # Get the complexity of the main function/class
            complexity = analysis.function_list[0].cyclomatic_complexity if analysis.function_list else 1
            
            return complexity
    except Exception as e:
        logger.error(f"Error calculating complexity: {str(e)}")
        return 1  # Return default complexity if calculation fails

# Initialize tokenizer once at module level
_tokenizer = None

def get_tokenizer():
    """Get or initialize the tokenizer"""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return _tokenizer

def count_tokens(code):
    """
    Count tokens using the sentence transformer tokenizer
    """
    try:
        tokenizer = get_tokenizer()
        tokens = tokenizer.encode(code)
        return len(tokens)
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
        return 0

def get_prompt_strategy(prompting_strategy, code):
    """
    Get the appropriate prompt based on the selected strategy - CORRECTED VERSION
    """
    if prompting_strategy == "few_shot":
        return [
            {"role": "system", "content": "You rename obfuscated Java identifiers to meaningful names while preserving exact code functionality. ONLY change identifier names."},
            {"role": "user", "content": "Example of CORRECT identifier renaming:\n\nObfuscated:\n```java\nimport java.util.Scanner;\npublic class Main_3 {\n    public static void main(String[] stringArray) {\n        Scanner llllllIIIIllIl = new Scanner(System.in);\n        long IlllllIIIIllIl = llllllIIIIllIl.nextLong();\n        long lIllllIIIIllIl = IlllllIIIIllIl / 10L;\n        if (IlllllIIIIllIl < 0L & IlllllIIIIllIl % 10L != 0L) {\n            --lIllllIIIIllIl;\n        }\n        System.out.println(lIllllIIIIllIl);\n    }\n}\n```"},
            {"role": "assistant", "content": "Renamed identifiers only:\n```java\nimport java.util.Scanner;\npublic class Main_3 {\n    public static void main(String[] args) {\n        Scanner scanner = new Scanner(System.in);\n        long inputNumber = scanner.nextLong();\n        long result = inputNumber / 10L;\n        if (inputNumber < 0L & inputNumber % 10L != 0L) {\n            --result;\n        }\n        System.out.println(result);\n    }\n}\n```\n\nNotice: ONLY variable names changed, all statements and logic remain exactly the same."},
            {"role": "user", "content": f"Rename ONLY the obfuscated identifiers in this code:\n\n```java\n{code}\n```"}
        ]
    
    elif prompting_strategy == "cot":
        return [
            {"role": "system", "content": "You are a Java identifier deobfuscator. Your ONLY task is to rename obfuscated identifiers to meaningful names while preserving EXACT code behavior.\n\nCRITICAL CONSTRAINTS:\n- ONLY rename identifiers (variables, methods, classes)\n- Do NOT add any new statements or lines\n- Do NOT remove any existing statements or lines\n- Do NOT modify any logic, operations, or control flow\n- Preserve EXACT semantic behavior"},
            {"role": "user", "content": f"Rename ONLY the obfuscated identifiers in this Java code. Follow these steps:\n\nStep 1: Identify each obfuscated identifier and determine its purpose\nStep 2: Choose a meaningful name for each identifier\nStep 3: Replace ONLY the identifier names, changing nothing else\n\nIMPORTANT: The output code must be functionally identical to the input code.\n\nCode:\n```java\n{code}\n```\n\nProvide your analysis, then the code with ONLY renamed identifiers:"}
        ]
    
    elif prompting_strategy == "augmented_reasoning":
        return [
            {"role": "system", "content": "You are a Java identifier deobfuscator. Study the example below to understand how to rename obfuscated identifiers while preserving exact code behavior."},
            {"role": "user", "content": f"Here's how to deobfuscate by ONLY renaming identifiers:\n\n**Example Process:**\n\nObfuscated:\n```java\nScanner llIlIl = new Scanner(System.in);\nlong IlIlIl = llIlIl.nextLong();\nlong lIlIlI = IlIlIl / 10L;\nSystem.out.println(lIlIlI);\n```\n\nAnalysis:\n- llIlIl: Scanner for input → scanner\n- IlIlIl: input number → inputNumber\n- lIlIlI: division result → result\n\nDeobfuscated (ONLY names changed):\n```java\nScanner scanner = new Scanner(System.in);\nlong inputNumber = scanner.nextLong();\nlong result = inputNumber / 10L;\nSystem.out.println(result);\n```\n\n**CRITICAL: Notice that ONLY the identifier names changed. No statements were added, removed, or modified.**\n\nNow apply the same approach to this code:\n\n```java\n{code}\n```"}
        ]
    
    else:  # Default strategy
        return [
            {"role": "system", "content": "You are a Java identifier deobfuscator. Your ONLY task is to rename obfuscated variable, method, and class names to meaningful names while preserving EXACT code behavior.\n\nCRITICAL CONSTRAINTS:\n- ONLY rename identifiers (variables, methods, classes)\n- Do NOT add any new statements, lines, or code\n- Do NOT remove any existing statements, lines, or code\n- Do NOT modify any logic, operations, or control flow\n- Preserve EXACT semantic behavior"},
            {"role": "user", "content": f"Rename ONLY the obfuscated identifiers in this Java code to meaningful names.\n\nIMPORTANT: The output code must be functionally identical to the input code. Only identifier names should change.\n\nCode:\n```java\n{code}\n```\n\nProvide the code with ONLY renamed identifiers:"}
        ]

def extract_code_and_reasoning(response_text, prompting_strategy):
    """
    Extract the final code and reasoning process from the response - IMPROVED VERSION
    """
    import re
    
    if not response_text or not response_text.strip():
        return None, ""
    
    response_text = response_text.strip()
    
    # Try to extract from code blocks first
    java_pattern = r'```java\s*\n(.*?)\n```'
    matches = re.findall(java_pattern, response_text, re.DOTALL)
    
    if matches:
        # Take the longest match (likely the most complete code)
        final_code = max(matches, key=len).strip()
        # Remove code blocks from reasoning
        reasoning_text = re.sub(r'```java.*?```', '', response_text, flags=re.DOTALL).strip()
        return final_code, reasoning_text
    
    # Try generic code blocks
    generic_pattern = r'```\s*\n(.*?)\n```'
    matches = re.findall(generic_pattern, response_text, re.DOTALL)
    
    if matches:
        # Filter for Java-like code
        java_matches = [match for match in matches if any(indicator in match for indicator in ['class ', 'public ', 'import ', '{', '}', ';'])]
        if java_matches:
            final_code = max(java_matches, key=len).strip()
            reasoning_text = re.sub(r'```.*?```', '', response_text, flags=re.DOTALL).strip()
            return final_code, reasoning_text
    
    # Fallback: look for code after certain keywords
    code_keywords = ['deobfuscated', 'final', 'result', 'renamed']
    for keyword in code_keywords:
        pattern = rf'{keyword}.*?code.*?:\s*\n(.*?)(?=\n\n|\Z)'
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1).strip()
            if len(code) > 50 and any(indicator in code for indicator in ['class ', 'public ', '{', '}', ';']):
                reasoning_text = response_text[:match.start()].strip()
                return code, reasoning_text
    
    # Last resort: return the whole response as code if it looks like Java
    if any(indicator in response_text for indicator in ['class ', 'public ', 'import ', 'System.out']):
        return response_text.strip(), ""
    
    return None, response_text

def generate_deobfuscated_code(client, code, model_name, prompting_strategy="default"):
    """
    Generate deobfuscated code using the fine-tuned model - IMPROVED ERROR HANDLING
    """
    from openai import RateLimitError, AuthenticationError, APIError
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            messages = get_prompt_strategy(prompting_strategy, code)
            
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3,  # Lower temperature for more consistent output
                max_tokens=2048
            )
            return response.choices[0].message.content.strip()
            
        except RateLimitError:
            wait_time = 2 ** attempt
            logger.warning(f"Rate limit hit, waiting {wait_time} seconds...")
            time.sleep(wait_time)
        except AuthenticationError:
            logger.error("Authentication failed - check API key")
            return None
        except APIError as e:
            logger.error(f"API error: {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(1)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None
    
    return None

def process_item_all_strategies(item, client, model_name, tokenizer):
    logger.info(f"Processing code from {item['subdirectory']} with all strategies...")
    
    # Define prompting strategies to test
    prompting_strategies = ["default", "few_shot", "cot", "augmented_reasoning"]
    
    result = {
        "subdirectory": item['subdirectory'],
        "original_code": item['code'],
        "strategies": {}
    }
    
    for strategy in prompting_strategies:
        logger.info(f"  Generating with {strategy} strategy...")
        response = generate_deobfuscated_code(client, item['code'], model_name, strategy)
        
        if response:
            # Extract code and reasoning
            final_code, reasoning_process = extract_code_and_reasoning(response, strategy)
            
            if final_code:
                strategy_result = {}
                
                # Add reasoning_process for CoT and augmented_reasoning strategies
                if strategy in ["cot", "augmented_reasoning"]:
                    strategy_result["reasoning_process"] = reasoning_process
                
                strategy_result["deobfuscated_code"] = final_code
                
                result["strategies"][strategy] = strategy_result
                logger.info(f"    Generated deobfuscated code successfully")
            else:
                result["strategies"][strategy] = {
                    "deobfuscated_code": None
                }
                logger.warning(f"    Failed to extract code with {strategy}")
        else:
            result["strategies"][strategy] = {
                "deobfuscated_code": None
            }
            logger.warning(f"    Failed to generate with {strategy}")
        
        # Reduced sleep to avoid hitting API rate limits
        time.sleep(0.2)
    
    logger.info(f"Completed processing for {item['subdirectory']}")
    return result

def save_results(results, output_file):
    """
    Save the results to a JSON file with error handling
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise

def main():
    # Initialize OpenAI client
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please ensure it is set in your .env file or environment.")
    
    client = OpenAI(api_key=api_key)
    
    # Configuration - make these configurable
    fine_tuned_model = os.getenv('MODEL_NAME', "ft:gpt-4.1-mini-2025-04-14:isge-bf::BwU1szpq")
    dataset_path = os.getenv('DATASET_PATH', '/Users/lionel.nikiema/Documents/Thesis/Project2_FineTuning/Reversibility_Hypothesis/Reversibity_Hypothesis/Obfuscated_code _generation/Evaluation_dataset/least_300_VarRen_Obfuscator.jar.json')
    output_file = os.getenv('OUTPUT_FILE', 'gpt-4.1-mini-VarRen_Deobfuscator_all_prompting_strategies.json')
    max_workers = int(os.getenv('MAX_WORKERS', '2'))
    
    # Load evaluation dataset
    try:
        evaluation_data = load_evaluation_dataset(dataset_path)
        logger.info(f"Loaded {len(evaluation_data)} items from dataset")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # Initialize tokenizer
    tokenizer = get_tokenizer()
    results = []
    
    # Process with thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_item_all_strategies, item, client, fine_tuned_model, tokenizer) 
                  for item in evaluation_data]
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error processing item: {e}")
            
            # Reduced sleep between items
            time.sleep(0.5)
    
    # Save all results
    try:
        save_results(results, output_file)
        logger.info(f"\nAll results saved to {output_file}")
        logger.info(f"Processed {len(results)} items successfully")
    except Exception as e:
        logger.error(f"Failed to save final results: {e}")

if __name__ == "__main__":
    main()

