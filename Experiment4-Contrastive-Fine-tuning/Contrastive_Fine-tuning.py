"""
Fine-tuning Pipeline for Contrastive Learning Framework (Mixed Training)

This module handles the fine-tuning process for GPT-3.5-turbo using the mixed training approach
that includes both semantic equivalence recognition and obfuscation generation examples.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
import requests
from tqdm import tqdm

# Import the triplet generator
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from java_jsonl_triplet_generator_mixed import JavaJSONLTripletGenerator, JavaContrastiveDataFormatter

class GPT35FineTuningPipeline:
    """
    Pipeline for fine-tuning GPT-3.5-turbo with mixed contrastive learning.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the fine-tuning pipeline with OpenAI API key.
        
        Args:
            api_key: OpenAI API key (optional, can be set via environment variable)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            print("Warning: No OpenAI API key provided. Set it via the constructor or OPENAI_API_KEY environment variable.")
        
        self.api_base = "https://api.openai.com/v1"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def validate_fine_tuning_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate that the fine-tuning file is properly formatted and analyze its content.
        
        Args:
            file_path: Path to the fine-tuning JSONL file
            
        Returns:
            Dictionary with validation results and statistics
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = 0
                semantic_equivalence_count = 0
                obfuscation_generation_count = 0
                
                for line in f:
                    try:
                        data = json.loads(line)
                        
                        # Check for required fields
                        if "messages" not in data:
                            print(f"Error: Line {line_count + 1} missing 'messages' field")
                            return {"valid": False, "error": f"Line {line_count + 1} missing 'messages' field"}
                        
                        messages = data["messages"]
                        if not isinstance(messages, list) or len(messages) < 2:
                            print(f"Error: Line {line_count + 1} has invalid 'messages' format")
                            return {"valid": False, "error": f"Line {line_count + 1} has invalid 'messages' format"}
                        
                        # Check message format
                        for msg in messages:
                            if "role" not in msg or "content" not in msg:
                                print(f"Error: Line {line_count + 1} has message missing 'role' or 'content'")
                                return {"valid": False, "error": f"Line {line_count + 1} has message missing 'role' or 'content'"}
                            
                            if msg["role"] not in ["system", "user", "assistant"]:
                                print(f"Error: Line {line_count + 1} has invalid role: {msg['role']}")
                                return {"valid": False, "error": f"Line {line_count + 1} has invalid role: {msg['role']}"}
                        
                        # Categorize example type
                        system_content = next((m["content"] for m in messages if m["role"] == "system"), "")
                        user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
                        
                        if "semantically equivalent" in user_content.lower():
                            semantic_equivalence_count += 1
                        elif "obfuscate this java code" in user_content.lower():
                            obfuscation_generation_count += 1
                        
                        line_count += 1
                    except json.JSONDecodeError:
                        print(f"Error: Line {line_count + 1} is not valid JSON")
                        return {"valid": False, "error": f"Line {line_count + 1} is not valid JSON"}
                
                if line_count == 0:
                    print("Error: File is empty")
                    return {"valid": False, "error": "File is empty"}
                
                result = {
                    "valid": True,
                    "total_examples": line_count,
                    "semantic_equivalence_examples": semantic_equivalence_count,
                    "obfuscation_generation_examples": obfuscation_generation_count
                }
                
                print(f"Validation successful: {line_count} examples found")
                print(f"  - Semantic equivalence examples: {semantic_equivalence_count}")
                print(f"  - Obfuscation generation examples: {obfuscation_generation_count}")
                
                return result
        
        except Exception as e:
            print(f"Error validating file: {e}")
            return {"valid": False, "error": str(e)}
    
    def upload_file(self, file_path: str) -> Dict[str, Any]:
        """
        Upload a file to OpenAI for fine-tuning.
        
        Args:
            file_path: Path to the fine-tuning JSONL file
            
        Returns:
            Response from the API
        """
        try:
            with open(file_path, 'rb') as f:
                response = requests.post(
                    f"{self.api_base}/files",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files={"file": (os.path.basename(file_path), f, "application/jsonl")},
                    data={"purpose": "fine-tune"}
                )
            
            if response.status_code == 200:
                result = response.json()
                print(f"File uploaded successfully: {result['id']}")
                return result
            else:
                print(f"Error uploading file: {response.status_code} - {response.text}")
                return {"error": response.text}
        
        except Exception as e:
            print(f"Error uploading file: {e}")
            return {"error": str(e)}
    
    def create_fine_tuning_job(self, file_id: str, model: str = "gpt-3.5-turbo", suffix: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a fine-tuning job.
        
        Args:
            file_id: ID of the uploaded file
            model: Base model to fine-tune
            suffix: Optional suffix for the fine-tuned model name
            
        Returns:
            Response from the API
        """
        try:
            payload = {
                "training_file": file_id,
                "model": model,
                "hyperparameters": {
                    "n_epochs": 2,
                    "batch_size": "auto",
                    "learning_rate_multiplier": "auto"
                }
            }
            
            if suffix:
                payload["suffix"] = suffix
            
            response = requests.post(
                f"{self.api_base}/fine_tuning/jobs",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Fine-tuning job created: {result['id']}")
                print(f"📊 Hyperparameters: {result.get('hyperparameters', {})}")
                return result
            else:
                print(f"Error creating fine-tuning job: {response.status_code} - {response.text}")
                return {"error": response.text}
        
        except Exception as e:
            print(f"Error creating fine-tuning job: {e}")
            return {"error": str(e)}
    
    def get_fine_tuning_job(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a fine-tuning job.
        
        Args:
            job_id: ID of the fine-tuning job
            
        Returns:
            Response from the API
        """
        try:
            response = requests.get(
                f"{self.api_base}/fine_tuning/jobs/{job_id}",
                headers=self.headers
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error getting fine-tuning job: {response.status_code} - {response.text}")
                return {"error": response.text}
        
        except Exception as e:
            print(f"Error getting fine-tuning job: {e}")
            return {"error": str(e)}
    
    def wait_for_fine_tuning_completion(self, job_id: str, poll_interval: int = 60) -> Dict[str, Any]:
        """
        Wait for a fine-tuning job to complete.
        
        Args:
            job_id: ID of the fine-tuning job
            poll_interval: Seconds between status checks
            
        Returns:
            Final job status
        """
        print(f"Waiting for fine-tuning job {job_id} to complete...")
        
        while True:
            job_status = self.get_fine_tuning_job(job_id)
            
            if "error" in job_status:
                return job_status
            
            status = job_status.get("status")
            print(f"Current status: {status}")
            
            if status in ["succeeded", "failed", "cancelled"]:
                return job_status
            
            print(f"Waiting {poll_interval} seconds before checking again...")
            time.sleep(poll_interval)
    
    def run_fine_tuning_pipeline(self, file_path: str, model: str = "gpt-3.5-turbo", suffix: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete fine-tuning pipeline.
        
        Args:
            file_path: Path to the fine-tuning JSONL file
            model: Base model to fine-tune
            suffix: Optional suffix for the fine-tuned model name
            
        Returns:
            Result of the fine-tuning process
        """
        # Validate the file
        validation_result = self.validate_fine_tuning_file(file_path)
        if not validation_result.get("valid", False):
            return {"error": "File validation failed", "details": validation_result}
        
        # Upload the file
        upload_result = self.upload_file(file_path)
        if "error" in upload_result:
            return upload_result
        
        file_id = upload_result["id"]
        
        # Wait for file processing to complete
        print("Waiting for file processing to complete...")
        time.sleep(10)
        
        # Create fine-tuning job
        job_result = self.create_fine_tuning_job(file_id, model, suffix)
        if "error" in job_result:
            return job_result
        
        job_id = job_result["id"]
        
        # Wait for fine-tuning to complete
        final_status = self.wait_for_fine_tuning_completion(job_id)
        
        # Add validation statistics to the result
        if "error" not in final_status:
            final_status["validation_stats"] = {
                "total_examples": validation_result.get("total_examples", 0),
                "semantic_equivalence_examples": validation_result.get("semantic_equivalence_examples", 0),
                "obfuscation_generation_examples": validation_result.get("obfuscation_generation_examples", 0)
            }
        
        return final_status

    def run_direct_fine_tuning(self, 
                              fine_tuning_jsonl_path: str,
                              model: str = "gpt-3.5-turbo",
                              suffix: str = "dead_code-contrastive") -> Dict[str, Any]:
        """
        Run fine-tuning directly with an existing fine-tuning JSONL file.
        
        Args:
            fine_tuning_jsonl_path: Path to the fine-tuning JSONL file
            model: Base model to fine-tune
            suffix: Suffix for the fine-tuned model name
            
        Returns:
            Result of the fine-tuning process
        """
        print("🚀 Starting direct fine-tuning pipeline...")
        
        # Step 1: Validate the fine-tuning file
        print("\n✅ Step 1: Validating fine-tuning data...")
        validation_result = self.validate_fine_tuning_file(fine_tuning_jsonl_path)
        if not validation_result.get("valid", False):
            return {"error": "Fine-tuning file validation failed", "details": validation_result}
        
        # Step 2: Upload file to OpenAI
        print("\n📤 Step 2: Uploading file to OpenAI...")
        upload_result = self.upload_file(fine_tuning_jsonl_path)
        if "error" in upload_result:
            return upload_result
        
        file_id = upload_result["id"]
        
        # Wait for file processing to complete
        print("⏳ Waiting for file processing to complete...")
        time.sleep(10)
        
        # Step 3: Create fine-tuning job
        print("\n🎯 Step 3: Creating fine-tuning job...")
        job_result = self.create_fine_tuning_job(file_id, model, suffix)
        if "error" in job_result:
            return job_result
        
        job_id = job_result["id"]
        
        # Step 4: Wait for fine-tuning to complete
        print("\n⏱️  Step 4: Waiting for fine-tuning to complete...")
        final_status = self.wait_for_fine_tuning_completion(job_id)
        
        # Add validation statistics to the result
        if "error" not in final_status:
            final_status["validation_stats"] = validation_result
        
        print("\n🎉 Direct fine-tuning completed successfully!")
        return final_status

    def setup_complete_pipeline(self, 
                              input_jsonl_path: str,
                              output_triplets_path: str = "dead_code_triplets_mixed.json",
                              output_fine_tuning_path: str = "dead_code_fine_tuning_mixed.jsonl",
                              model: str = "gpt-3.5-turbo", #gpt-3.5-turbo, 
                              suffix: str = "dead_code-contrastive") -> Dict[str, Any]:
        """
        Complete automated pipeline: generate triplets, create fine-tuning data, and start fine-tuning.
        
        Args:
            input_jsonl_path: Path to the input JSONL file with code pairs
            output_triplets_path: Path to save the generated triplets JSON file
            output_fine_tuning_path: Path to save the fine-tuning JSONL file
            model: Base model to fine-tune
            suffix: Suffix for the fine-tuned model name
            
        Returns:
            Result of the complete pipeline
        """
        print("🚀 Starting complete fine-tuning pipeline...")
        
        # Step 1: Generate triplets from input JSONL
        print("\n📊 Step 1: Generating triplets from input data...")
        generator = JavaJSONLTripletGenerator()
        
        # Extract code pairs from JSONL file
        code_pairs = generator.extract_code_pairs_from_jsonl(input_jsonl_path)
        
        if len(code_pairs) < 2:
            return {"error": f"Need at least 2 code pairs, found {len(code_pairs)}"}
        
        # Generate triplets with shuffled unrelated code
        triplets = generator.generate_triplets_with_shuffled_unrelated(code_pairs)
        
        # Save triplets
        generator.save_triplets(triplets, output_triplets_path)
        
        # Step 2: Format for fine-tuning
        print("\n🔄 Step 2: Formatting data for fine-tuning...")
        formatter = JavaContrastiveDataFormatter()
        fine_tuning_data = formatter.format_for_fine_tuning(triplets)
        
        # Save for fine-tuning
        formatter.save_for_fine_tuning(fine_tuning_data, output_fine_tuning_path)
        
        # Step 3: Validate the fine-tuning file
        print("\n✅ Step 3: Validating fine-tuning data...")
        validation_result = self.validate_fine_tuning_file(output_fine_tuning_path)
        if not validation_result.get("valid", False):
            return {"error": "Fine-tuning file validation failed", "details": validation_result}
        
        # Step 4: Upload file to OpenAI
        print("\n📤 Step 4: Uploading file to OpenAI...")
        upload_result = self.upload_file(output_fine_tuning_path)
        if "error" in upload_result:
            return upload_result
        
        file_id = upload_result["id"]
        
        # Wait for file processing to complete
        print("⏳ Waiting for file processing to complete...")
        time.sleep(10)
        
        # Step 5: Create fine-tuning job
        print("\n🎯 Step 5: Creating fine-tuning job...")
        job_result = self.create_fine_tuning_job(file_id, model, suffix)
        if "error" in job_result:
            return job_result
        
        job_id = job_result["id"]
        
        # Step 6: Wait for fine-tuning to complete
        print("\n⏱️  Step 6: Waiting for fine-tuning to complete...")
        final_status = self.wait_for_fine_tuning_completion(job_id)
        
        # Add pipeline statistics to the result
        if "error" not in final_status:
            final_status["pipeline_stats"] = {
                "input_code_pairs": len(code_pairs),
                "generated_triplets": len(triplets),
                "fine_tuning_examples": len(fine_tuning_data),
                "validation_stats": validation_result
            }
        
        print("\n🎉 Pipeline completed successfully!")
        return final_status


def main():
    """
    Main function to run the fine-tuning pipeline from command line.
    """
    # Set all arguments directly in the code
    fine_tuning_file = "/Users/lionel.nikiema/Documents/Thesis/Project2_FineTuning/contrastive_deobfuscation/jsonl_creation/deadcode_graduated_fine_tuning-1K.jsonl"
    model = "gpt-3.5-turbo" #gpt-3.5-turbo, 
    suffix = "dead_code-contrastive_graduated"
    api_key = None  # Will use environment variable OPENAI_API_KEY
    
    print("🚀 Starting fine-tuning pipeline...")
    print(f"📁 Fine-tuning file: {fine_tuning_file}")
    print(f"🤖 Model: {model}")
    print(f"🏷️  Suffix: {suffix}")
    print(f"📊 Epochs: 2")
    
    pipeline = GPT35FineTuningPipeline(api_key=api_key)
    
    # Run direct fine-tuning with the specified file
    result = pipeline.run_direct_fine_tuning(
        fine_tuning_jsonl_path=fine_tuning_file,
        model=model,
        suffix=suffix
    )
    
    print("\nFine-tuning result:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    # Example usage for direct fine-tuning
    # Uncomment the lines below to run direct fine-tuning with existing JSONL file
    
    # pipeline = GPT35FineTuningPipeline()
    # result = pipeline.run_direct_fine_tuning(
    #     fine_tuning_jsonl_path="java_fine_tuning_mixed.jsonl",
    #     model="gpt-3.5-turbo",
    #     suffix="java-mixed-contrastive"
    # )
    # print(json.dumps(result, indent=2))
    
    # Example usage for complete pipeline
    # Uncomment the lines below to run the complete pipeline automatically
    
    # pipeline = GPT35FineTuningPipeline()
    # result = pipeline.setup_complete_pipeline(
    #     input_jsonl_path="/Users/lionel.nikiema/Documents/Thesis/Project2_FineTuning/contrastive_deobfuscation/VariableRenaming_Dataset_1K.jsonl",
    #     model="gpt-3.5-turbo",
    #     suffix="java-mixed-contrastive"
    # )
    # print(json.dumps(result, indent=2))
    
    main()
