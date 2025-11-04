"""
Contrastive Learning Framework for Open Source Models
Implements triplet-based contrastive learning for Java code obfuscation
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from torch.utils.data import Dataset, DataLoader
import random
from typing import List, Dict, Tuple, Optional

# ===== Memory optimization settings =====
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.empty_cache()
gc.collect()

# ===== Configuration =====
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DATASET_PATH = "fixed_finetune_chatml_variable_renaming-30K.jsonl"  # ChatML format JSONL
TOKEN_MAX_LENGTH = 2048
BATCH_SIZE = 8  # Smaller batch size for instruction fine-tuning
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 3e-5
MAX_EPOCHS = 2
WARMUP_RATIO = 0.1
OUTPUT_DIR = "./contrastive_mistral_finetuning_variable_renaming-output-30K"
VALIDATION_SPLIT = 0.1
TEMPERATURE = 0.1  # Temperature for contrastive loss
MARGIN = 0.3  # Margin for triplet loss

# Force instruction fine-tuning mode
FORCE_INSTRUCTION_MODE = True

print("\n======= STARTING INSTRUCTION FINE-TUNING FRAMEWORK =======\n\n")

class InstructionDataset(Dataset):
    """Dataset for instruction fine-tuning with conversation format"""
    
    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def format_conversation(self, messages):
        """Format conversation messages into a single string"""
        formatted = ""
        for message in messages:
            if message["role"] == "system":
                formatted += f"<|im_start|>system\n{message['content']}<|im_end|>\n"
            elif message["role"] == "user":
                formatted += f"<|im_start|>user\n{message['content']}<|im_end|>\n"
            elif message["role"] == "assistant":
                formatted += f"<|im_start|>assistant\n{message['content']}<|im_end|>\n"
        return formatted
    
    def __getitem__(self, idx):
        conversation = self.data[idx]["messages"]
        formatted_text = self.format_conversation(conversation)
        
        # Tokenize the conversation
        tokenized = self.tokenizer(
            formatted_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": tokenized["input_ids"].squeeze(0).clone()
        }

class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning with triplets"""
    
    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def create_triplet(self, idx):
        """Create a triplet: (anchor, positive, negative)"""
        # Anchor: original code
        anchor = self.data[idx]["Original_Code"]
        
        # Positive: obfuscated code (same meaning, different variable names)
        positive = self.data[idx]["Obfuscated_Code"]
        
        # Negative: randomly select a different obfuscated code
        negative_idx = random.choice([i for i in range(len(self.data)) if i != idx])
        negative = self.data[negative_idx]["Obfuscated_Code"]
        
        return anchor, positive, negative
    
    def tokenize_code(self, code, instruction=""):
        """Tokenize code with optional instruction"""
        if instruction:
            text = f"{instruction}:\n\n```java\n{code}\n```"
        else:
            text = f"```java\n{code}\n```"
        
        tokenized = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return tokenized
    
    def __getitem__(self, idx):
        anchor, positive, negative = self.create_triplet(idx)
        
        # Tokenize all three samples
        anchor_tokens = self.tokenize_code(anchor)
        positive_tokens = self.tokenize_code(positive)
        negative_tokens = self.tokenize_code(negative)
        
        return {
            "anchor_input_ids": anchor_tokens["input_ids"].squeeze(0),
            "anchor_attention_mask": anchor_tokens["attention_mask"].squeeze(0),
            "positive_input_ids": positive_tokens["input_ids"].squeeze(0),
            "positive_attention_mask": positive_tokens["attention_mask"].squeeze(0),
            "negative_input_ids": negative_tokens["input_ids"].squeeze(0),
            "negative_attention_mask": negative_tokens["attention_mask"].squeeze(0),
        }

class ContrastiveLoss(nn.Module):
    """Contrastive loss using temperature-scaled cosine similarity"""
    
    def __init__(self, temperature=0.1, margin=0.3):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        """
        Compute contrastive loss
        Args:
            anchor_embeddings: [batch_size, hidden_size]
            positive_embeddings: [batch_size, hidden_size]
            negative_embeddings: [batch_size, hidden_size]
        """
        # Normalize embeddings
        anchor_norm = F.normalize(anchor_embeddings, p=2, dim=1)
        positive_norm = F.normalize(positive_embeddings, p=2, dim=1)
        negative_norm = F.normalize(negative_embeddings, p=2, dim=1)
        
        # Compute similarities
        pos_sim = F.cosine_similarity(anchor_norm, positive_norm, dim=1) / self.temperature
        neg_sim = F.cosine_similarity(anchor_norm, negative_norm, dim=1) / self.temperature
        
        # Contrastive loss: maximize positive similarity, minimize negative similarity
        loss = torch.clamp(neg_sim - pos_sim + self.margin, min=0.0)
        
        return loss.mean()

class ContrastiveModel(nn.Module):
    """Wrapper model for contrastive learning"""
    
    def __init__(self, base_model, tokenizer):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.contrastive_loss = ContrastiveLoss(temperature=TEMPERATURE, margin=MARGIN)
        
    def get_embeddings(self, input_ids, attention_mask):
        """Extract embeddings from the model"""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use the last hidden state and take mean pooling
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # Mean pooling over sequence length
        embeddings = torch.mean(hidden_states, dim=1)  # [batch_size, hidden_size]
        
        return embeddings
    
    def forward(self, anchor_input_ids, anchor_attention_mask,
                positive_input_ids, positive_attention_mask,
                negative_input_ids, negative_attention_mask):
        
        # Get embeddings for all three samples
        anchor_embeddings = self.get_embeddings(anchor_input_ids, anchor_attention_mask)
        positive_embeddings = self.get_embeddings(positive_input_ids, positive_attention_mask)
        negative_embeddings = self.get_embeddings(negative_input_ids, negative_attention_mask)
        
        # Compute contrastive loss
        loss = self.contrastive_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        
        return {
            "loss": loss,
            "anchor_embeddings": anchor_embeddings,
            "positive_embeddings": positive_embeddings,
            "negative_embeddings": negative_embeddings
        }

class ContrastiveTrainer:
    """Custom trainer for contrastive learning"""
    
    def __init__(self, model, train_dataset, eval_dataset, tokenizer, args):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.num_train_epochs,
            eta_min=0
        )
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            num_workers=0
        )
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs["loss"]
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        return avg_loss
    
    def evaluate(self):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=0
        )
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs["loss"]
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Evaluation loss: {avg_loss:.4f}")
        return avg_loss
    
    def train(self):
        """Main training loop"""
        best_eval_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(self.args.num_train_epochs):
            print(f"\n======= EPOCH {epoch + 1}/{self.args.num_train_epochs} =======")
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Evaluate
            eval_loss = self.evaluate()
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Early stopping
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                patience_counter = 0
                # Save best model
                self.save_model(f"{self.args.output_dir}/best_model")
                print(f"New best model saved! Eval loss: {eval_loss:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
        
        # Save final model
        self.save_model(f"{self.args.output_dir}/final_model")
        print("Training completed!")
    
    def save_model(self, path):
        """Save the model"""
        os.makedirs(path, exist_ok=True)
        self.model.base_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

# ===== Load Model and Tokenizer =====
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
try:
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if use_bf16:
        print("BF16 is supported - using BF16 precision")
        dtype = torch.bfloat16
    else:
        print("BF16 not supported - using FP16 precision")
        dtype = torch.float16
    
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

model.gradient_checkpointing_enable()
model.config.use_cache = False

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ===== Load and Process Dataset =====
print("Loading dataset...")
try:
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset file not found: {DATASET_PATH}")
        available_files = os.listdir(os.path.dirname(DATASET_PATH))
        print(f"Available files in directory: {available_files}")
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")
    
    # Load JSONL dataset
    dataset = load_dataset("json", data_files={"train": DATASET_PATH})
    print(f"Dataset loaded with {len(dataset['train'])} examples")
    
    # Force instruction fine-tuning mode for ChatML format
    if FORCE_INSTRUCTION_MODE or "messages" in dataset["train"].column_names:
        print("Using ChatML format for instruction fine-tuning")
        use_instruction_format = True
        
        # Verify ChatML format
        if "messages" not in dataset["train"].column_names:
            raise ValueError("Dataset must have 'messages' field for ChatML format")
        
        # Split dataset
        splits = dataset["train"].train_test_split(test_size=VALIDATION_SPLIT)
        train_data = splits["train"]
        eval_data = splits["test"]
        
        print(f"Split dataset: {len(train_data)} training examples, {len(eval_data)} validation examples")
        
        # Create instruction datasets
        train_dataset = InstructionDataset(train_data, tokenizer, TOKEN_MAX_LENGTH)
        eval_dataset = InstructionDataset(eval_data, tokenizer, TOKEN_MAX_LENGTH)
        
        # Print sample conversation format
        sample_item = train_data[0]
        print(f"\nSample ChatML format:")
        print(f"System: {sample_item['messages'][0]['content'][:100]}...")
        print(f"User: {sample_item['messages'][1]['content'][:100]}...")
        print(f"Assistant: {sample_item['messages'][2]['content'][:100]}...")
        
    else:
        # Fallback to contrastive learning format (disabled by default)
        print("Contrastive learning mode disabled - using instruction fine-tuning only")
        raise ValueError("Please use ChatML format with 'messages' field for instruction fine-tuning")
    
    print("Dataset processing complete")
    
except Exception as e:
    print(f"Error preparing dataset: {e}")
    raise

# ===== Training Setup =====
print("Setting up ChatML instruction fine-tuning...")

# Use standard HuggingFace Trainer for instruction fine-tuning
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    num_train_epochs=MAX_EPOCHS,
    weight_decay=0.01,
    warmup_ratio=WARMUP_RATIO,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=10,
    save_total_limit=2,
    remove_unused_columns=False,
    # ChatML specific settings
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

print("\n======= STARTING CHATML INSTRUCTION FINE-TUNING =======\n")
print(f"Training for {MAX_EPOCHS} epochs with batch size {BATCH_SIZE}")
print(f"Using ChatML format with system/user/assistant messages")
print(f"Output directory: {OUTPUT_DIR}")

try:
    trainer.train()
    print("\n✅ ChatML instruction fine-tuning completed successfully!")
    
    # Save the final model
    trainer.save_model(f"{OUTPUT_DIR}/final_model")
    print(f"Model saved to {OUTPUT_DIR}/final_model")
    
except Exception as e:
    print(f"\n❌ Training error: {e}")
    import traceback
    traceback.print_exc()

print("\n======= CHATML INSTRUCTION FINE-TUNING COMPLETE =======\n") 