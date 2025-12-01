#!/usr/bin/env python3
"""
Training script for GLiNER2 model on prompt injection detection dataset.

This script:
- Loads the pre-trained model from HuggingFace (fastino/gliner2-large-v1)
- Supports LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Trains on the prompt injection dataset
- Times the entire training process end-to-end
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Set tokenizers parallelism before importing transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import TrainingArguments

# Try to import peft for LoRA support
try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("Warning: peft not installed. LoRA training unavailable. Install with: pip install peft")

# Try to import dotenv, but continue if not available
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False
    load_dotenv = None

from gliner2.model import Extractor
from gliner2.trainer import ExtractorDataset, ExtractorDataCollator, ExtractorTrainer


# =============================================================================
# LoRA Configuration
# =============================================================================

def apply_lora_to_model(
    model: Extractor,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
) -> Extractor:
    """
    Apply LoRA (Low-Rank Adaptation) to the model's encoder for efficient fine-tuning.
    
    LoRA freezes the pre-trained model weights and injects trainable low-rank
    decomposition matrices into transformer layers, dramatically reducing the
    number of trainable parameters.
    
    Args:
        model: The Extractor model to apply LoRA to
        r: LoRA rank (lower = fewer params, higher = more capacity). Default 16.
        lora_alpha: LoRA scaling factor. Default 32.
        lora_dropout: Dropout probability for LoRA layers. Default 0.1.
        target_modules: List of module names to apply LoRA to. 
                       If None, targets attention layers (query, key, value, dense).
    
    Returns:
        Model with LoRA applied to the encoder
        
    Raises:
        ImportError: If peft library is not installed
    """
    if not HAS_PEFT:
        raise ImportError(
            "peft library is required for LoRA training. "
            "Install with: pip install peft"
        )
    
    if target_modules is None:
        # Target attention layers - these names work for most BERT-like models
        target_modules = ["query", "key", "value", "dense"]
    
    # Create LoRA configuration
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    
    # Apply LoRA to the encoder
    model.encoder = get_peft_model(model.encoder, lora_config)
    
    # Freeze all parameters except LoRA layers and classifier
    for name, param in model.named_parameters():
        # Keep LoRA parameters trainable
        if "lora" in name.lower():
            param.requires_grad = True
        # Keep classifier trainable
        elif "classifier" in name.lower():
            param.requires_grad = True
        # Keep count prediction head trainable
        elif "count_pred" in name.lower():
            param.requires_grad = True
        # Keep count embedding trainable
        elif "count_embed" in name.lower():
            param.requires_grad = True
        # Freeze everything else
        else:
            param.requires_grad = False
    
    # Calculate and print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\n🔧 LoRA Configuration Applied:")
    print(f"  - Rank (r): {r}")
    print(f"  - Alpha: {lora_alpha}")
    print(f"  - Dropout: {lora_dropout}")
    print(f"  - Target modules: {target_modules}")
    print(f"\n📊 Parameter Summary:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"  - Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
    
    return model


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
    save_steps: int = 1000,
    eval_steps: int = 1000,
    save_total_limit: int = 3,
    fp16: bool = True,
    dataloader_num_workers: int = 4,
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
    
    # ==========================================================================
    # Configuration
    # ==========================================================================
    model_name = "fastino/gliner2-large-v1"
    data_path = "data/prompt_injection_dataset_extractor_format.jsonl"
    output_dir = "./output"
    
    # LoRA Configuration - Set to True for efficient fine-tuning
    USE_LORA = True  # Toggle LoRA training on/off
    LORA_R = 16      # LoRA rank (8-64 typical, higher = more capacity)
    LORA_ALPHA = 32  # LoRA scaling (typically 2x rank)
    LORA_DROPOUT = 0.1
    
    # ==========================================================================
    
    # Load HuggingFace token
    print("\n[1/6] Loading HuggingFace token...")
    try:
        hf_token = load_hf_token()
        print(f"✓ Token loaded successfully (length: {len(hf_token)})")
    except ValueError as e:
        print(f"✗ Error: {e}")
        return
    
    # Load model
    print(f"\n[2/6] Loading model from HuggingFace: {model_name}...")
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
    
    # Apply LoRA if enabled
    if USE_LORA:
        print(f"\n[3/6] Applying LoRA for efficient fine-tuning...")
        try:
            model = apply_lora_to_model(
                model,
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
            )
            print(f"✓ LoRA applied successfully")
        except ImportError as e:
            print(f"✗ Error: {e}")
            print("Falling back to full fine-tuning...")
            USE_LORA = False
        except Exception as e:
            print(f"✗ Error applying LoRA: {e}")
            print("Falling back to full fine-tuning...")
            USE_LORA = False
    else:
        print(f"\n[3/6] Skipping LoRA (full fine-tuning mode)")
    
    # Load and prepare dataset
    print(f"\n[4/6] Loading dataset from: {data_path}...")
    try:
        dataset = ExtractorDataset(data_path)
        print(f"✓ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return
    
    # Create data collator
    data_collator = ExtractorDataCollator()
    
    # Create training arguments
    print(f"\n[5/6] Setting up training configuration...")
    training_args = create_training_arguments(
        output_dir=output_dir,
        num_epochs=3,
        batch_size=8,
        learning_rate=2e-4 if USE_LORA else 2e-5,  # Higher LR for LoRA
        warmup_steps=100,
        logging_steps=50,
        save_steps=1000,
        fp16=torch.cuda.is_available(),
    )
    print(f"✓ Training arguments configured")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Epochs: {training_args.num_train_epochs}")
    print(f"  - Batch size: {training_args.per_device_train_batch_size}")
    print(f"  - FP16: {training_args.fp16}")
    print(f"  - Training mode: {'LoRA' if USE_LORA else 'Full Fine-tuning'}")
    
    # Create trainer with appropriate learning rates
    print(f"\n[6/6] Initializing trainer...")
    
    # LoRA uses higher learning rates since we're training fewer parameters
    if USE_LORA:
        encoder_lr = 3e-4  # Higher LR for LoRA layers
        custom_lr = 2e-4   # Higher LR for classifier
    else:
        encoder_lr = 1e-5  # Lower LR for full encoder fine-tuning
        custom_lr = 2e-5   # Lower LR for classifier
    
    trainer = ExtractorTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        encoder_lr=encoder_lr,
        custom_lr=custom_lr,
        weight_decay=0.01,
        finetune_classifier=False,
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

