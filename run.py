#!/usr/bin/env python3
"""
Training script for GLiNER2 model on prompt injection detection dataset.

This script:
- Loads the pre-trained model from HuggingFace (fastino/gliner2-large-v1)
- Trains on the prompt injection dataset
- Times the entire training process end-to-end
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import torch
from transformers import TrainingArguments

# Try to import dotenv, but continue if not available
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False
    load_dotenv = None

from gliner2.model import Extractor
from gliner2.trainer import ExtractorDataset, ExtractorDataCollator, ExtractorTrainer


def load_hf_token() -> str:
    """
    Load HuggingFace token from environment variables.
    
    Checks for HF_TOKEN or HUGGINGFACE_TOKEN in environment.
    If .env file exists, loads it first.
    
    Returns:
        str: HuggingFace token
        
    Raises:
        ValueError: If token is not found
    """
    # Load .env file if it exists and dotenv is available
    if HAS_DOTENV:
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)
    
    # Try different environment variable names
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_READ_TOKEN")
    
    if not token:
        raise ValueError(
            "HuggingFace token not found. Please set HF_TOKEN, HUGGINGFACE_TOKEN, "
            "or HF_READ_TOKEN in your .env file or environment variables."
        )
    
    return token


def create_training_arguments(
    output_dir: str = "./output",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 500,
    eval_steps: int = 500,
    save_total_limit: int = 3,
    fp16: bool = True,
    dataloader_num_workers: int = 0,
    gradient_accumulation_steps: int = 1,
) -> TrainingArguments:
    """
    Create TrainingArguments for the trainer.
    
    Args:
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate (will be overridden by trainer's custom LRs)
        warmup_steps: Number of warmup steps
        logging_steps: Logging frequency
        save_steps: Checkpoint saving frequency
        eval_steps: Evaluation frequency
        save_total_limit: Maximum number of checkpoints to keep
        fp16: Whether to use mixed precision training
        dataloader_num_workers: Number of dataloader workers
        gradient_accumulation_steps: Gradient accumulation steps
        
    Returns:
        TrainingArguments object
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,  # Will be overridden by trainer's custom LRs
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        fp16=fp16,
        dataloader_num_workers=dataloader_num_workers,
        gradient_accumulation_steps=gradient_accumulation_steps,
        remove_unused_columns=False,
        report_to="none",  # Disable wandb/tensorboard by default
    )


def main():
    """
    Main training function that orchestrates the entire training process.
    """
    print("=" * 80)
    print("GLiNER2 Prompt Injection Detection Training")
    print("=" * 80)
    
    # Start timing
    start_time = time.time()
    
    # Configuration
    model_name = "fastino/gliner2-large-v1"
    data_path = "data/prompt_injection_dataset_extractor_format.jsonl"
    output_dir = "./output"
    
    # Load HuggingFace token
    print("\n[1/5] Loading HuggingFace token...")
    try:
        hf_token = load_hf_token()
        print(f"✓ Token loaded successfully (length: {len(hf_token)})")
    except ValueError as e:
        print(f"✗ Error: {e}")
        return
    
    # Load model
    print(f"\n[2/5] Loading model from HuggingFace: {model_name}...")
    try:
        # Set token in environment for HuggingFace Hub (multiple variable names for compatibility)
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        
        model = Extractor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print(f"✓ Model loaded successfully")
        print(f"  - Model device: {next(model.parameters()).device}")
        print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Load and prepare dataset
    print(f"\n[3/5] Loading dataset from: {data_path}...")
    try:
        dataset = ExtractorDataset(data_path)
        print(f"✓ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return
    
    # Create data collator
    data_collator = ExtractorDataCollator()
    
    # Create training arguments
    print(f"\n[4/5] Setting up training configuration...")
    training_args = create_training_arguments(
        output_dir=output_dir,
        num_epochs=3,
        batch_size=8,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        fp16=torch.cuda.is_available(),
    )
    print(f"✓ Training arguments configured")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Epochs: {training_args.num_train_epochs}")
    print(f"  - Batch size: {training_args.per_device_train_batch_size}")
    print(f"  - FP16: {training_args.fp16}")
    
    # Create trainer
    print(f"\n[5/5] Initializing trainer...")
    trainer = ExtractorTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        encoder_lr=1e-5,  # Lower LR for encoder
        custom_lr=2e-5,   # Higher LR for classifier and other layers
        weight_decay=0.01,
        finetune_classifier=False,  # Set to True to only train classifier
    )
    print(f"✓ Trainer initialized")
    print(f"  - Encoder LR: {trainer.encoder_lr}")
    print(f"  - Custom LR: {trainer.custom_lr}")
    print(f"  - Weight decay: {trainer.custom_weight_decay}")
    
    # Training
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    train_start_time = time.time()
    train_duration = None
    
    try:
        trainer.train()
        train_end_time = time.time()
        train_duration = train_end_time - train_start_time
        
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
        print(f"Training duration: {train_duration:.2f} seconds ({train_duration/60:.2f} minutes)")
        
    except Exception as e:
        train_end_time = time.time()
        train_duration = train_end_time - train_start_time
        print(f"\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        print(f"Training ran for {train_duration:.2f} seconds before failure")
        return
    
    # Save final model
    print(f"\nSaving final model to {output_dir}...")
    try:
        trainer.save_model()
        print(f"✓ Model saved successfully")
    except Exception as e:
        print(f"✗ Error saving model: {e}")
    
    # End timing
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Print summary
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"Total time (end-to-end): {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    if train_duration is not None:
        print(f"Training time: {train_duration:.2f} seconds ({train_duration/60:.2f} minutes)")
    print(f"Model saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

