import os
import json
import subprocess
import difflib
from tqdm import tqdm
import shutil
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from openai import OpenAI
from dotenv import load_dotenv
import tempfile
import time

# Load environment variables
load_dotenv()

# Directory names
WORK_DIR = "A_Z_testing/obfuscated_VarRen_GPT35_epoch-3"
TEST_DIR = "Test_Cases"
REPORT_FILE = "A_Z_testing/obfuscated_VarRen_GPT35_epoch-3.txt"
FINE_TUNED_MODEL = "ft:gpt-3.5-turbo-0125:isge-bf::BWk8uOtu"

# Map subdir suffix to test subsubdir
SUFFIX_MAP = {chr(ord('a') + i): chr(ord('A') + i) for i in range(26)}

# Add this utility function near the top, after imports
MAX_ERROR_CHARS = 2000
def truncate_error(text, max_chars=MAX_ERROR_CHARS):
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...\n[TRUNCATED]\n...\n" + text[-half:]

def prepare_work_dir():
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.makedirs(WORK_DIR)

def extract_public_class_name(java_code):
    match = re.search(r'public\s+class\s+([A-Za-z_][A-Za-z0-9_]*)', java_code)
    if match:
        return match.group(1)
    return None

def generate_obfuscated_code(client, code, attempt=1, prev_failed_code=None, prev_error=None):
    """Generate or correct obfuscated code using the fine-tuned model"""
    try:
        if attempt == 1:
            system_prompt = "You are a code obfuscator. Your task is to obfuscate the given code by Variable Renaming while preserving functionality."
            user_content = f"Obfuscate this code:\n{code}"
        else:
            safe_prev_error = truncate_error(prev_error)
            system_prompt = (
                "You are a code obfuscator. The previous obfuscated code did not work. "
                "Here is the code and the error. Please correct the code so it compiles and passes all tests."
            )
            user_content = (
                f"Original code (for reference):\n{code}\n\n"
                f"Previous obfuscated code:\n{prev_failed_code}\n\n"
                f"Error message:\n{safe_prev_error}\n\n"
                "Please correct the code."
            )
        
        response = client.chat.completions.create(
            model=FINE_TUNED_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3,
            max_tokens=2048
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating obfuscated code: {str(e)}")
        return None

def compile_java(java_file):
    compile_cmd = ["javac", java_file]
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr

def run_java(class_dir, class_name, input_file, timeout=30):
    run_cmd = ["java", "-cp", class_dir, class_name]
    with open(input_file, "r") as infile:
        result = subprocess.run(run_cmd, stdin=infile, capture_output=True, text=True, timeout=timeout)
    return result.returncode == 0, result.stdout.strip(), result.stderr

def compare_output(actual, expected_file, float_tol=1e-3):
    with open(expected_file, "r") as f:
        expected = f.read().strip()
    actual_lines = actual.strip().splitlines()
    expected_lines = expected.strip().splitlines()
    if len(actual_lines) != len(expected_lines):
        return False, expected
    for a, e in zip(actual_lines, expected_lines):
        try:
            a_f = float(a)
            e_f = float(e)
            if math.isclose(a_f, e_f, rel_tol=float_tol, abs_tol=float_tol):
                continue
            else:
                return False, expected
        except ValueError:
            if a.strip() != e.strip():
                return False, expected
    return True, expected

def process_java_file(args):
    subdir, original_code = args
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)
    report_lines = []
    error_summary = []
    num_passed = 0
    num_total = 0
    class_name = extract_public_class_name(original_code) or "Main"
    subdir_path = os.path.join(WORK_DIR, subdir)
    os.makedirs(subdir_path, exist_ok=True)
    java_file = os.path.join(subdir_path, f"{class_name}.java")

    prev_failed_code = None
    prev_error = None

    for attempt in range(1, 4):
        report_lines.append(f"\nAttempt {attempt} for {subdir}:\n")
        obfuscated_code = generate_obfuscated_code(
            client, original_code, attempt,
            prev_failed_code=prev_failed_code, prev_error=prev_error
        )
        if not obfuscated_code:
            report_lines.append("Failed to generate obfuscated code\n")
            error_summary.append("Failed to generate obfuscated code")
            prev_failed_code = prev_failed_code or ""
            prev_error = "Failed to generate obfuscated code"
            continue
        with open(java_file, "w") as jf:
            jf.write(obfuscated_code)
        compiled, compile_err = compile_java(java_file)
        if not compiled:
            report_lines.append(f"Compile Error:\n{compile_err}\n")
            error_summary.append(f"Compile Error: {compile_err}")
            prev_failed_code = obfuscated_code
            prev_error = compile_err
            continue
        parts = subdir.split("_")
        if len(parts) < 2:
            report_lines.append("Invalid subdir name\n")
            error_summary.append("Invalid subdir name")
            return (subdir, java_file, 0, 1, report_lines, error_summary)
        base = parts[0]
        suffix = parts[1]
        test_case_letter = SUFFIX_MAP.get(suffix)
        if not test_case_letter:
            report_lines.append(f"Unknown suffix '{suffix}'\n")
            error_summary.append(f"Unknown suffix '{suffix}'")
            return (subdir, java_file, 0, 1, report_lines, error_summary)
        test_path = os.path.join(TEST_DIR, base, test_case_letter)
        in_dir = os.path.join(test_path, "in")
        out_dir = os.path.join(test_path, "out")
        if not (os.path.isdir(in_dir) and os.path.isdir(out_dir)):
            report_lines.append("Test case directories not found\n")
            error_summary.append("Test case directories not found")
            return (subdir, java_file, 0, 1, report_lines, error_summary)
        test_files = [f for f in os.listdir(in_dir) if f.endswith(".txt")]
        all_tests_passed = True
        num_passed = 0
        num_total = 0
        for test_file in test_files:
            input_file = os.path.join(in_dir, test_file)
            output_file = os.path.join(out_dir, test_file)
            if not os.path.exists(output_file):
                report_lines.append(f"{test_file}: Output file missing\n")
                error_summary.append(f"{test_file}: Output file missing")
                num_total += 1
                all_tests_passed = False
                continue
            try:
                ran, actual_out, run_err = run_java(subdir_path, class_name, input_file)
                num_total += 1
                if not ran:
                    report_lines.append(f"{test_file}: Runtime Error:\n{run_err}\n")
                    error_summary.append(f"{test_file}: Runtime Error: {run_err}")
                    all_tests_passed = False
                    prev_failed_code = obfuscated_code
                    prev_error = run_err
                    continue
                correct, expected_out = compare_output(actual_out, output_file)
                if correct:
                    report_lines.append(f"{test_file}: Passed\n")
                    num_passed += 1
                else:
                    diff = "\n".join(difflib.unified_diff(
                        expected_out.splitlines(), actual_out.splitlines(),
                        fromfile='expected', tofile='actual', lineterm=''))
                    report_lines.append(f"{test_file}: Failed\nDiff:\n{diff}\n")
                    error_summary.append(f"{test_file}: Failed\nDiff:\n{diff}")
                    all_tests_passed = False
                    prev_failed_code = obfuscated_code
                    prev_error = diff
            except subprocess.TimeoutExpired:
                report_lines.append(f"{test_file}: Timeout\n")
                error_summary.append(f"{test_file}: Timeout")
                all_tests_passed = False
                prev_failed_code = obfuscated_code
                prev_error = "Timeout"
        if all_tests_passed and num_total > 0:
            report_lines.append(f"\nAll tests passed on attempt {attempt}\n")
            return (subdir, java_file, num_passed, num_total, report_lines, error_summary)
        if attempt == 3:
            report_lines.append("\nFailed after 3 attempts\n")
    return (subdir, java_file, num_passed, num_total, report_lines, error_summary)

def main():
    # Load original code
    with open('Gpt-3.5_Generation/Variable_Renaming/obfuscated_gpt35_varren_epoch_3_1K.json', 'r') as f:
        original_data = json.load(f)
    
    prepare_work_dir()
    
    total = len(original_data)
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    java_file_results = []
    all_report_lines = []
    failed_json = []
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_java_file, (item['subdirectory'], item['code'])) for item in original_data]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Java files"):
            try:
                subdir, java_file, num_passed, num_total, report_lines, error_summary = future.result()
                java_file_results.append((subdir, java_file, num_passed, num_total, error_summary))
                all_report_lines.extend(report_lines)
                total_tests += num_total
                passed_tests += num_passed
                failed_tests += (num_total - num_passed)
                
                if num_passed < num_total:
                    failed_json.append({
                        "subdirectory": subdir,
                        "errors": error_summary
                    })
            
            except Exception as e:
                print(f"Error processing file: {str(e)}")
    
    # Calculate statistics
    java_passed = sum(1 for _, _, p, t, _ in java_file_results if p == t and t > 0)
    java_failed = len(java_file_results) - java_passed
    
    # Write report
    summary_lines = [
        "+++++++++++++++++++++++++++++++++ TEST REPORT SUMMARY ++++++++++++++++++++++++++++++++++\n",
        f"Java files summary:\n",
        f"Total Java files: {len(java_file_results)}\n",
        f"Passed (all test cases): {java_passed}\n",
        f"Failed: {java_failed}\n"
    ]
    
    if total_tests > 0:
        pass_percent = (passed_tests / total_tests) * 100
        fail_percent = (failed_tests / total_tests) * 100
        summary_lines.extend([
            f"\nTotal test cases: {total_tests}\n",
            f"Passed: {passed_tests} ({pass_percent:.2f}%)\n",
            f"Failed: {failed_tests} ({fail_percent:.2f}%)\n"
        ])
    else:
        summary_lines.append("No tests were run.\n")
    
    summary_lines.append("+++++++++++++++++++++++++++++++++ END OF SUMMARY ++++++++++++++++++++++++++++++++++\n\n")
    
    with open(REPORT_FILE, "w") as f:
        f.writelines(summary_lines)
        f.writelines(line if line.endswith('\n') else line + '\n' for line in all_report_lines)
    
    with open("A_Z_testing/obfuscated_VarRen_GPT35_epoch-3-failed_code.json", "w") as f:
        json.dump(failed_json, f, indent=2)
    
    print("".join(summary_lines))
    print(f"Report written to {REPORT_FILE}")

if __name__ == "__main__":
    main() 