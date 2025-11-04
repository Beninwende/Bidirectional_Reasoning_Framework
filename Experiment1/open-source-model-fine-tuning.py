"""
Fine-tune a language model with LoRA for Java Code Obfuscation (Variable Renaming)
"""

import os
import torch
import gc
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

# ===== Memory optimization settings =====
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.empty_cache()
gc.collect()

# ===== Configuration =====
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # Use a base model to avoid starting with a potentially damaged model
DATASET_PATH = "VariableRenaming_Dataset.csv"  # Make sure this is the correct dataset path
TOKEN_MAX_LENGTH = 2048  # Increased for better context
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 3e-5  # More appropriate learning rate
MAX_EPOCHS = 5  # Train for epochs instead of fixed steps
WARMUP_RATIO = 0.1  # Warm up over 10% of training
OUTPUT_DIR = "./mistral-lora-output-variable-renaming-epochs-2"
VALIDATION_SPLIT = 0.1  # Use 10% of data for validation

print("\n======= STARTING JAVA CODE VARIABLE RENAMING FINE-TUNING =======\n\n")

# ===== Load Model and Tokenizer =====
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model (this may take a while)...")
try:
    # Check for BF16 support
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if use_bf16:
        print("BF16 is supported - using BF16 precision")
        dtype = torch.bfloat16
    else:
        print("BF16 not supported - using FP16 precision")
        dtype = torch.float16
    
    # Load the model using auto device map
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    
    print("Model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# ===== Apply LoRA =====
print("Applying LoRA to model...")

# Configure the model for training first
model.gradient_checkpointing_enable()  # Enable gradient checkpointing
model.config.use_cache = False  # This is important for gradient checkpointing

# Using PEFT's LoRA implementation
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,  # Rank of the update matrices
    lora_alpha=32,  # Alpha parameter for LoRA scaling
    lora_dropout=0.05,  # Dropout probability for LoRA layers
    bias="none",  # Don't train bias parameters
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# Create a new, properly prepared model with LoRA
model = get_peft_model(model, lora_config)

# Print trainable vs non-trainable parameters
model.print_trainable_parameters()

# Verify that trainable parameters exist and require gradients
trainable_params = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        trainable_params += 1
if trainable_params == 0:
    raise ValueError("No trainable parameters found in model! LoRA application failed.")
else:
    print(f"✅ Confirmed {trainable_params} trainable parameters with requires_grad=True")

# Force parameters that should be trainable to require gradients
for name, param in model.named_parameters():
    if any(trainable_name in name for trainable_name in ["lora"]):
        param.requires_grad = True

# ===== Data Processing =====
def validate_java_code(code):
    """Simple validation to ensure code appears to be valid Java"""
    if not isinstance(code, str):
        return False
    
    code = code.strip()
    if not code:
        return False
    
    # Check for some basic Java syntax
    java_indicators = [
        "{", "}", ";", "public", "private", "class", "void", "int", "String"
    ]
    
    # The code should contain at least some Java syntax elements
    if not any(indicator in code for indicator in java_indicators):
        return False
        
    return True

def process_examples(examples):
    """Process and tokenize the dataset examples"""
    texts = []
    
    for instr, orig, obfus in zip(examples["Instruction"], examples["Original_Code"], examples["Obfuscated_Code"]):
        if instr and orig and obfus:
            text = f"{instr}:\n\n```java\n{orig}\n```\n\nResult:\n```java\n{obfus}\n```"
            texts.append(text)
    
    # Use tokenizer without creating tensors yet
    tokenized = tokenizer(
        texts, 
        padding="max_length",
        truncation=True,
        max_length=TOKEN_MAX_LENGTH,
        return_tensors=None
    )
    
    # Set labels equal to input_ids for causal language modeling
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# ===== Load Dataset =====
print("Loading dataset...")
try:
    # Verify dataset file exists
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset file not found: {DATASET_PATH}")
        available_files = os.listdir(os.path.dirname(DATASET_PATH))
        print(f"Available files in directory: {available_files}")
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")
    
    # Load and check dataset
    dataset = load_dataset("csv", data_files={"train": DATASET_PATH})
    print(f"Dataset loaded with {len(dataset['train'])} examples")
    
    # Check column names in dataset
    print(f"Dataset columns: {dataset['train'].column_names}")
    
    # Use correct column names from your dataset
    if "Original_Code" not in dataset["train"].column_names or "Obfuscated_Code" not in dataset["train"].column_names:
        # Check for alternative column names
        columns = dataset["train"].column_names
        if "Java Code" in columns and "Obfuscated Code" in columns:
            # Rename columns to match expected format
            dataset = dataset.rename_column("Java Code", "Original_Code")
            dataset = dataset.rename_column("Obfuscated Code", "Obfuscated_Code")
            print("Renamed columns to match expected format")
        else:
            print(f"Required columns not found. Available columns: {columns}")
            raise ValueError("Dataset missing required columns")
    
    # Split dataset into train/validation
    splits = dataset["train"].train_test_split(test_size=VALIDATION_SPLIT)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]
    
    print(f"Split dataset: {len(train_dataset)} training examples, {len(eval_dataset)} validation examples")
    
    # Process datasets with simplified approach
    print("Processing dataset...")
    processed_train = train_dataset.map(
        process_examples,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    processed_eval = eval_dataset.map(
        process_examples,
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    torch.cuda.empty_cache()
    gc.collect()

    print("Dataset processing complete")
    
except Exception as e:
    print(f"Error preparing dataset: {e}")
    raise

# Verify dataset format
print(f"Processed training dataset size: {len(processed_train)}")
print(f"Processed evaluation dataset size: {len(processed_eval)}")

# ===== Training Setup =====
print("Setting up training configuration...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=MAX_EPOCHS,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=0.01,
    fp16=not use_bf16,
    bf16=use_bf16,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    eval_steps=100,
    save_strategy="epoch",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False,
    gradient_checkpointing=True,
    report_to="none",
    ddp_find_unused_parameters=False,
    dataloader_pin_memory=False,
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
)

# Create data collator for language modeling tasks
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ===== Initialize Trainer =====
print("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_train,
    eval_dataset=processed_eval,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# ===== Training =====
print("\n======= STARTING TRAINING =======\n")
print(f"Training for {MAX_EPOCHS} epochs with effective batch size of {BATCH_SIZE * GRAD_ACCUM_STEPS}")

# Add error handling and model saving
try:
    # Start training
    train_result = trainer.train()
    
    # Log and save training results
    print(f"\n✅ Training completed successfully!")
    print(f"Training metrics: {train_result.metrics}")
    
    # Run a final evaluation
    eval_result = trainer.evaluate()
    print(f"Final evaluation metrics: {eval_result}")
    
    # Save metrics
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_metrics("eval", eval_result)
    
except Exception as e:
    print(f"\n❌ Training error: {e}")
    # Try to save checkpoint even if training fails
    try:
        print("Attempting to save partial model...")
        trainer.save_model(f"{OUTPUT_DIR}/emergency_save")
        print("Emergency save successful")
    except Exception as save_error:
        print(f"Emergency save failed: {save_error}")

# ===== Save Final Model =====
print("\nSaving fine-tuned model...")
try:
    # Save the model
    trainer.save_model(f"{OUTPUT_DIR}/final_model")
    print("✅ Model saved successfully!")
except Exception as e:
    print(f"❌ Error saving model: {e}")

print("\n======= PROCESS COMPLETE =======\n") 