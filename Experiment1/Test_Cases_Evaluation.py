import os
import subprocess
import difflib
from tqdm import tqdm
import json
import shutil
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import math

# Directory names
JSON_FILE = "Gpt-3.5_Generation/int&string-encryption/obfuscated_integer_string_encryption_GPT35-epoch-3.json"
WORK_DIR = "Z_TEST/obfuscated_encryption_GPT35_epoch-3"
TEST_DIR = "Test_Cases"
REPORT_FILE = "Z_TEST/obfuscated_encryption_GPT35_epoch-3.txt"

# Map subdir suffix to test subsubdir
SUFFIX_MAP = {chr(ord('a') + i): chr(ord('A') + i) for i in range(26)}

def prepare_work_dir():
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.makedirs(WORK_DIR)

def extract_public_class_name(java_code):
    """
    Extracts the public class name from Java code.
    Returns the class name if found, else None.
    """
    match = re.search(r'public\s+class\s+([A-Za-z_][A-Za-z0-9_]*)', java_code)
    if match:
        return match.group(1)
    return None

def get_java_files_from_json():
    with open(JSON_FILE, "r") as f:
        data = json.load(f)
    for entry in data:
        subdir = entry["subdirectory"]
        code = entry["code"]
        # Extract the intended file name (default to Main)
        intended_file_name = "Main"
        class_name = extract_public_class_name(code)
        if class_name:
            intended_file_name = class_name
        # If class name and file name don't match, harmonize
        if not class_name or class_name != intended_file_name:
            # Replace the public class name in the code
            code = re.sub(
                r'(public\s+class\s+)([A-Za-z_][A-Za-z0-9_]*)',
                r'\1' + intended_file_name,
                code,
                count=1
            )
        subdir_path = os.path.join(WORK_DIR, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        java_file = os.path.join(subdir_path, f"{intended_file_name}.java")
        with open(java_file, "w") as jf:
            jf.write(code)
        yield subdir, java_file

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
    """
    Compare actual and expected output, allowing for minor float differences.
    Returns (True, expected) if similar enough, else (False, expected).
    """
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
    subdir, java_file = args
    report_lines = []
    class_name = os.path.splitext(os.path.basename(java_file))[0]
    parts = subdir.split("_")
    if len(parts) < 2:
        report_lines.append(f"{java_file}: Invalid subdir name\n")
        return (subdir, java_file, 0, 1, report_lines, ["Invalid subdir name"])
    base = parts[0]
    suffix = parts[1]
    test_subdir = base
    test_case_letter = SUFFIX_MAP.get(suffix, None)
    if not test_case_letter:
        report_lines.append(f"{java_file}: Unknown suffix '{suffix}'\n")
        return (subdir, java_file, 0, 1, report_lines, [f"Unknown suffix '{suffix}'"])
    test_path = os.path.join(TEST_DIR, test_subdir, test_case_letter)
    in_dir = os.path.join(test_path, "in")
    out_dir = os.path.join(test_path, "out")
    if not (os.path.isdir(in_dir) and os.path.isdir(out_dir)):
        report_lines.append(f"{java_file}: Test case directories not found\n")
        return (subdir, java_file, 0, 1, report_lines, ["Test case directories not found"])
    compiled, compile_err = compile_java(java_file)
    report_lines.append(f"Testing {java_file}:\n")
    if not compiled:
        report_lines.append(f"  Compile Error:\n{compile_err}\n")
        return (subdir, java_file, 0, 1, report_lines, [f"Compile Error: {compile_err}"])
    class_dir = os.path.dirname(java_file)
    test_files = [f for f in os.listdir(in_dir) if f.endswith(".txt")]
    num_passed = 0
    num_total = 0
    error_summary = []
    for test_file in test_files:
        input_file = os.path.join(in_dir, test_file)
        output_file = os.path.join(out_dir, test_file)
        if not os.path.exists(output_file):
            report_lines.append(f"  {test_file}: Output file missing\n")
            error_summary.append(f"{test_file}: Output file missing")
            num_total += 1
            continue
        for attempt in range(2):
            try:
                ran, actual_out, run_err = run_java(class_dir, class_name, input_file, timeout=30)
                break
            except subprocess.TimeoutExpired:
                if attempt == 0:
                    report_lines.append(f"  {test_file}: Timeout after 30s, retrying...\n")
                else:
                    report_lines.append(f"  {test_file}: Timeout after second 30s, skipping this test.\n")
                    print(f"Timeout after 60s for subdirectory: {subdir}")
                    error_summary.append(f"{test_file}: Timeout after 60s")
                    raise TimeoutError(f"Timeout after 60s for subdirectory: {subdir}")
        num_total += 1
        if not ran:
            report_lines.append(f"  {test_file}: Runtime Error or Timeout:\n{run_err}\n")
            error_summary.append(f"{test_file}: Runtime Error or Timeout: {run_err}")
            continue
        correct, expected_out = compare_output(actual_out, output_file)
        if correct:
            report_lines.append(f"  {test_file}: Passed\n")
            num_passed += 1
        else:
            diff = "\n".join(difflib.unified_diff(
                expected_out.splitlines(), actual_out.splitlines(),
                fromfile='expected', tofile='actual', lineterm=''))
            report_lines.append(f"  {test_file}: Failed\nDiff:\n{diff}\n")
            error_summary.append(f"{test_file}: Failed\nDiff:\n{diff}")
    return (subdir, java_file, num_passed, num_total, report_lines, error_summary)

def main():
    prepare_work_dir()
    java_files = list(get_java_files_from_json())
    total = len(java_files)
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    java_file_results = []
    all_report_lines = []
    failed_json = []
    # Error tracking
    error_type_counts = {
        'compile_error': 0,
        'runtime_error': 0,
        'timeout': 0,
        'output_file_missing': 0,
        'invalid_subdir': 0,
        'unknown_suffix': 0,
        'test_case_dir_missing': 0,
        'other': 0
    }
    total_compile_attempts = 0
    total_compile_failures = 0
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_java_file, args) for args in java_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Java files (parallel)", unit="file"):
            try:
                subdir, java_file, num_passed, num_total, report_lines, error_summary = future.result()
                java_file_results.append((subdir, java_file, num_passed, num_total, error_summary))
                all_report_lines.extend(report_lines)
                total_tests += num_total
                passed_tests += num_passed
                failed_tests += (num_total - num_passed)
                avg = num_passed / num_total if num_total > 0 else 0
                # Error type aggregation
                for err in error_summary:
                    if err.startswith("Compile Error"):
                        error_type_counts['compile_error'] += 1
                        total_compile_failures += 1
                    elif "Timeout" in err:
                        error_type_counts['timeout'] += 1
                    elif "Runtime Error" in err:
                        error_type_counts['runtime_error'] += 1
                    elif "Output file missing" in err:
                        error_type_counts['output_file_missing'] += 1
                    elif "Invalid subdir name" in err:
                        error_type_counts['invalid_subdir'] += 1
                    elif "Unknown suffix" in err:
                        error_type_counts['unknown_suffix'] += 1
                    elif "Test case directories not found" in err:
                        error_type_counts['test_case_dir_missing'] += 1
                    else:
                        error_type_counts['other'] += 1
                total_compile_attempts += 1
                if avg < 0.5:
                    failed_json.append({
                        "subdirectory": subdir,
                        "errors": error_summary
                    })
            except TimeoutError as e:
                print(f"Worker exited due to timeout: {e}")
                error_type_counts['timeout'] += 1
    java_passed = 0
    java_failed = 0
    for subdir, java_file, num_passed, num_total, error_summary in java_file_results:
        avg = num_passed / num_total if num_total > 0 else 0
        if avg >= 0.5:
            java_passed += 1
        else:
            java_failed += 1
            subdir_path = os.path.join(WORK_DIR, subdir)
            if os.path.exists(subdir_path):
                shutil.rmtree(subdir_path)
    summary_lines = []
    summary_lines.append("+++++++++++++++++++++++++++++++++ TEST REPORT SUMMARY ++++++++++++++++++++++++++++++++++\n")
    summary_lines.append(f"Java files summary:\n")
    summary_lines.append(f"Total Java files: {len(java_file_results)}\n")
    summary_lines.append(f"Passed (>=50% test cases): {java_passed}\n")
    summary_lines.append(f"Failed (<50% test cases): {java_failed}\n")
    # Add compile failure percentage
    compile_fail_percent = (error_type_counts['compile_error'] / len(java_file_results)) * 100 if len(java_file_results) > 0 else 0
    summary_lines.append(f"\nFiles failed to compile: {error_type_counts['compile_error']} ({compile_fail_percent:.2f}%)\n")
    # Add error type breakdown
    summary_lines.append("\nError Type Breakdown (by Java file):\n")
    for err_type, count in error_type_counts.items():
        percent = (count / len(java_file_results)) * 100 if len(java_file_results) > 0 else 0
        summary_lines.append(f"  {err_type.replace('_', ' ').capitalize()}: {count} ({percent:.2f}%)\n")
    if total_tests > 0:
        pass_percent = (passed_tests / total_tests) * 100
        fail_percent = (failed_tests / total_tests) * 100
        summary_lines.append(f"\nTotal test cases: {total_tests}\n")
        summary_lines.append(f"Passed: {passed_tests} ({pass_percent:.2f}%)\n")
        summary_lines.append(f"Failed: {failed_tests} ({fail_percent:.2f}%)\n")
    else:
        summary_lines.append("No tests were run.\n")
    summary_lines.append("+++++++++++++++++++++++++++++++++ END OF SUMMARY ++++++++++++++++++++++++++++++++++\n\n")
    with open(REPORT_FILE, "w") as f:
        f.writelines(summary_lines)
        f.writelines(line if line.endswith('\n') else line + '\n' for line in all_report_lines)
    print(f"Report written to {REPORT_FILE}")
    with open("Z_TEST/obfuscated_encryption_GPT35-epoch-2-failed_code.json", "w") as f:
        json.dump(failed_json, f, indent=2)
    print("".join(summary_lines))

if __name__ == "__main__":
    main()