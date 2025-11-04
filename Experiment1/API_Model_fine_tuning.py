import os
from openai import OpenAI
from dotenv import load_dotenv
import time
import json

# Load environment variables
load_dotenv()

def upload_file(client, file_path):
    """
    Upload a file to OpenAI
    """
    print(f"Uploading file: {file_path}")
    with open(file_path, 'rb') as f:
        response = client.files.create(
            file=f,
            purpose='fine-tune'
        )
    print(f"File uploaded successfully. File ID: {response.id}")
    return response.id

def create_fine_tuning_job(client, file_id, epoch):
    """
    Create a fine-tuning job for a specific epoch
    """
    print(f"Creating fine-tuning job for epoch {epoch}...")
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model="gpt-3.5-turbo",  # Using GPT-3.5-turbo as the base model
        hyperparameters={
            "n_epochs": epoch,
            "batch_size": 4,
            "learning_rate_multiplier": 2
        }
    )
    print(f"Fine-tuning job created successfully for epoch {epoch}. Job ID: {job.id}")
    return job.id

def monitor_job(client, job_id, epoch):
    """
    Monitor the fine-tuning job progress
    """
    while True:
        try:
            job = client.fine_tuning.jobs.retrieve(job_id)
            print(f"\nJob status for epoch {epoch}: {job.status}")
            
            if job.status == 'succeeded':
                print(f"\nFine-tuning completed successfully for epoch {epoch}!")
                print(f"Fine-tuned model: {job.fine_tuned_model}")
                return job.fine_tuned_model
            elif job.status == 'failed':
                print(f"\nFine-tuning failed for epoch {epoch}!")
                if job.error:
                    print(f"Error: {job.error}")
                return None
            elif job.status == 'queued':
                print(f"Job for epoch {epoch} is in queue. This might take some time...")
                print("You can check the status in the OpenAI dashboard.")
            
            # Wait for 60 seconds before checking again
            time.sleep(60)
        except Exception as e:
            print(f"Error checking job status: {str(e)}")
            time.sleep(60)

def save_model_info(model_info, output_file):
    """
    Save model information to a JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(model_info, f, indent=2)

def main():
    # Initialize OpenAI client with API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    client = OpenAI(api_key=api_key)
    
    # File path for the JSONL data
    jsonl_file = 'Total_Obusfaction_deobfuscation_deadcode.jsonl'
    
    try:
        # Upload the file
        file_id = upload_file(client, jsonl_file)
        
        # Create fine-tuning jobs for each epoch
        total_epochs = 2
        model_info = {}
        
        for epoch in range(1, total_epochs + 1):
            print(f"\nStarting fine-tuning for epoch {epoch}")
            
            # Create fine-tuning job for this epoch
            job_id = create_fine_tuning_job(client, file_id, epoch)
            
            # Monitor the job and get the model name
            model_name = monitor_job(client, job_id, epoch)
            
            if model_name:
                model_info[f"epoch_{epoch}"] = {
                    "model_name": model_name,
                    "job_id": job_id
                }
                
                # Save model information after each successful epoch
                save_model_info(model_info, 'model_checkpoints.json')
                print(f"Saved checkpoint information for epoch {epoch}")
            else:
                print(f"Failed to complete fine-tuning for epoch {epoch}")
        
        print("\nAll epochs completed!")
        print("Final model checkpoints saved in 'model_checkpoints.json'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 