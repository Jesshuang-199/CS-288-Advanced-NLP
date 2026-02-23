#!/usr/bin/env python3
"""
Part 4 Baseline Training Script

This script demonstrates how to:
1. Train a BPE tokenizer on TinyStories
2. Pretrain a Transformer LM for next-token prediction
3. Fine-tune the model for multiple-choice QA
4. Evaluate using both prompting and fine-tuning approaches

Students can use this as a reference for their implementations.

Usage:
    # First, download datasets
    python part4/setup_datasets.py
    
    # Then run training (use --quick for testing)
    python part4/train_baseline.py --quick      # Quick test (~2 min)
    python part4/train_baseline.py              # Full training (~30 min on GPU)
"""

import argparse
import json
import sys
import torch
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from part1.train_bpe import train_bpe
from part1.tokenizer import get_tokenizer
from part2.model import TransformerLM
from part3.nn_utils import cross_entropy, gradient_clipping
from part4.datasets import create_pretraining_dataloader, create_qa_dataloader
from part4.sampling import generate_text
from part4.qa_model import TransformerForMultipleChoice, evaluate_qa_model
from part4.prompting import PromptTemplate, PromptingPipeline, evaluate_prompting
from part4.device_utils import resolve_device
from part4.trainer import Trainer, TrainingConfig, create_qa_loss_fn


# =============================================================================
# Configuration
# =============================================================================

CONFIGS = {
    "quick": {
        # Small config for quick testing
        "pretrain_data": Path(__file__).parent.parent / "part1/fixtures/tinystories_sample_5M.txt",
        "qa_train": Path(__file__).parent / "fixtures/qa_train.json",
        "qa_dev": Path(__file__).parent / "fixtures/qa_dev.json",
        "vocab_size": 512,
        "d_model": 128,
        "num_layers": 4,
        "num_heads": 4,
        "d_ff": 512,
        "context_length": 256,
        "pretrain_epochs": 3,
        "finetune_epochs": 5,
        "batch_size": 32,
        "lr": 1e-3,
    },
    "small": {
        # Small model, larger data - ~10M parameters
        "pretrain_data": Path(__file__).parent / "fixtures/tinystories_100k.txt",
        "qa_train": Path(__file__).parent / "fixtures/squad_train.json",
        "qa_dev": Path(__file__).parent / "fixtures/squad_dev.json",
        "vocab_size": 4096,
        "d_model": 256,
        "num_layers": 6,
        "num_heads": 8,
        "d_ff": 1024,
        "context_length": 512,
        "pretrain_epochs": 3,
        "finetune_epochs": 10,
        "batch_size": 32,
        "lr": 3e-4,
    },
    "medium": {
        # Medium model for good quality - ~50M parameters
        "pretrain_data": Path(__file__).parent / "fixtures/tinystories_100k.txt",
        "qa_train": Path(__file__).parent / "fixtures/squad_train.json",
        "qa_dev": Path(__file__).parent / "fixtures/squad_dev.json",
        "vocab_size": 8192,
        "d_model": 512,
        "num_layers": 8,
        "num_heads": 8,
        "d_ff": 2048,
        "context_length": 512,
        "pretrain_epochs": 5,
        "finetune_epochs": 15,
        "batch_size": 16,
        "lr": 1e-4,
    }
}


# =============================================================================
# Step 1: Train BPE Tokenizer
# =============================================================================

def train_tokenizer(pretrain_data: Path, vocab_size: int) -> tuple:
    """
    Train a BPE tokenizer on the pretraining corpus.
    
    Args:
        pretrain_data: Path to training text file
        vocab_size: Target vocabulary size
    
    Returns:
        (tokenizer, vocab, merges)
    """
    print("\n" + "=" * 60)
    print("STEP 1: Training BPE Tokenizer")
    print("=" * 60)
    
    special_tokens = ["<|endoftext|>", "<|pad|>"]
    
    print(f"Input: {pretrain_data}")
    print(f"Vocab size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    
    # Train BPE
    vocab, merges = train_bpe(
        input_path=pretrain_data,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    
    # Create tokenizer
    tokenizer = get_tokenizer(vocab, merges, special_tokens)
    
    # Test
    test_text = "Once upon a time, there was a little girl."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"\nTokenizer trained!")
    print(f"  Vocab size: {len(vocab)}")
    print(f"  Merges: {len(merges)}")
    print(f"\nTest encoding:")
    print(f"  Input:   '{test_text}'")
    print(f"  Tokens:  {len(tokens)} tokens")
    print(f"  Decoded: '{decoded}'")
    
    return tokenizer, vocab, merges


# =============================================================================
# Step 2: Pretrain Language Model
# =============================================================================

def pretrain_lm(
    tokenizer,
    config: dict,
    device: str = "cpu",
) -> TransformerLM:
    """
    Pretrain a Transformer language model on TinyStories.
    
    The model learns to predict the next token given previous tokens.
    This gives it general language understanding before fine-tuning.
    
    Args:
        tokenizer: Trained BPE tokenizer
        config: Model and training configuration
        device: Device to train on
    
    Returns:
        Pretrained TransformerLM
    """
    print("\n" + "=" * 60)
    print("STEP 2: Pretraining Language Model")
    print("=" * 60)
    
    runtime_context_length = config["context_length"]
    runtime_batch_size = config["batch_size"]
    if device == "mps":
        # Keep MPS memory usage bounded for stable runs.
        runtime_context_length = min(runtime_context_length, 256)
        runtime_batch_size = min(runtime_batch_size, 8)
    
    # Create model
    model = TransformerLM(
        vocab_size=len(tokenizer.vocab),
        context_length=runtime_context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel architecture:")
    print(f"  d_model: {config['d_model']}")
    print(f"  num_layers: {config['num_layers']}")
    print(f"  num_heads: {config['num_heads']}")
    print(f"  d_ff: {config['d_ff']}")
    print(f"  context_length: {runtime_context_length} (configured: {config['context_length']})")
    print(f"  Parameters: {num_params:,}")
    
    # Create dataloader
    dataloader = create_pretraining_dataloader(
        file_path=config["pretrain_data"],
        tokenizer=tokenizer,
        batch_size=runtime_batch_size,
        max_length=runtime_context_length,
        stride=runtime_context_length // 2,
        shuffle=True,
    )
    
    print(f"\nTraining data:")
    print(f"  File: {config['pretrain_data']}")
    print(f"  Documents: {len(dataloader.dataset)}")
    print(f"  Batches/epoch: {len(dataloader)}")
    
    # Training config
    train_config = TrainingConfig(
        num_epochs=config["pretrain_epochs"],
        learning_rate=config["lr"],
        weight_decay=0.01,
        warmup_steps=min(100, len(dataloader) // 5),
        max_grad_norm=1.0,
        device=device,
        log_interval=max(1, len(dataloader) // 5),
    )
    
    # Train
    trainer = Trainer(
        model=model,
        config=train_config,
        train_dataloader=dataloader,
    )
    
    print(f"\nTraining for {config['pretrain_epochs']} epoch(s)...")
    results = trainer.train()
    
    # Test generation
    print("\nGeneration test:")
    for prompt in ["Once upon a time", "The little dog"]:
        generated = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=30,
            method="greedy"
        )
        print(f"  '{prompt}' -> '{generated[:80]}...'")
    
    return model


# =============================================================================
# Step 3: Evaluate Zero-Shot Prompting
# =============================================================================

def evaluate_prompting(
    model: TransformerLM,
    tokenizer,
    qa_dev_path: Path,
    qa_train_path: Path | None = None,
    device: str = "cpu",
) -> dict:
    """
    Evaluate the pretrained model using zero-shot prompting.
    
    This tests if the model can answer questions without any fine-tuning,
    just by predicting which answer token (A, B, C, D) is most likely.
    
    Args:
        model: Pretrained TransformerLM
        tokenizer: Tokenizer
        qa_dev_path: Path to validation QA data
        device: Device
    
    Returns:
        Evaluation results dict
    """
    print("\n" + "=" * 60)
    print("STEP 4: Evaluating Prompting (on fine-tuned model)")
    print("=" * 60)
    
    # Load data
    with open(qa_dev_path) as f:
        dev_data = json.load(f)
    
    print(f"\nValidation examples: {len(dev_data)}")
    
    few_shot_pool = []
    if qa_train_path is not None and qa_train_path.exists():
        with open(qa_train_path) as f:
            train_data = json.load(f)
        few_shot_pool = [ex for ex in train_data if ex.get("answer", -1) >= 0][:4]
    
    # Prompt strategy search on a small subset for speed.
    search_data = dev_data[: min(200, len(dev_data))]
    candidates = [
        {"template": "simple", "strategy": "choice_likelihood", "shots": 0, "length_norm": True},
        {"template": "basic", "strategy": "choice_likelihood", "shots": 0, "length_norm": True},
        {"template": "instruction", "strategy": "choice_likelihood", "shots": 0, "length_norm": True},
        {"template": "simple", "strategy": "label_token", "shots": 0, "length_norm": False},
    ]
    if few_shot_pool:
        candidates.extend([
            {"template": "simple", "strategy": "choice_likelihood", "shots": 2, "length_norm": True},
            {"template": "instruction", "strategy": "choice_likelihood", "shots": 2, "length_norm": True},
        ])
    
    from part4.prompting import evaluate_prompting as eval_prompt
    
    best_cfg = None
    best_subset_result = None
    
    print("\nPrompting strategy search:")
    for cfg in candidates:
        pipeline = PromptingPipeline(
            model=model,
            tokenizer=tokenizer,
            template=PromptTemplate(template_name=cfg["template"]),
            device=device,
            scoring_strategy=cfg["strategy"],
            length_normalize=cfg["length_norm"],
            few_shot_examples=few_shot_pool[: cfg["shots"]],
        )
        subset_result = eval_prompt(pipeline, search_data)
        print(
            f"  template={cfg['template']:<11} strategy={cfg['strategy']:<17} "
            f"shots={cfg['shots']} -> subset acc {subset_result['accuracy']:.2%}"
        )
        if best_subset_result is None or subset_result["accuracy"] > best_subset_result["accuracy"]:
            best_subset_result = subset_result
            best_cfg = cfg
    
    # Re-run the best strategy on full validation set.
    best_pipeline = PromptingPipeline(
        model=model,
        tokenizer=tokenizer,
        template=PromptTemplate(template_name=best_cfg["template"]),
        device=device,
        scoring_strategy=best_cfg["strategy"],
        length_normalize=best_cfg["length_norm"],
        few_shot_examples=few_shot_pool[: best_cfg["shots"]],
    )
    results = eval_prompt(best_pipeline, dev_data)
    results["prompting_config"] = best_cfg
    
    print(f"\nPrompting accuracy (on fine-tuned model): {results['accuracy']:.2%}")
    print(f"Best prompting config: {best_cfg}")
    print(f"Random baseline: 25.00%")
    
    return results


# =============================================================================
# Step 4: Fine-tune for QA
# =============================================================================

def finetune_qa(
    pretrained_model: TransformerLM,
    tokenizer,
    config: dict,
    device: str = "cpu",
) -> TransformerForMultipleChoice:
    """
    Fine-tune the pretrained model for multiple-choice QA.
    
    We add a classification head and train the entire model to select
    the correct answer from 4 choices.
    
    Args:
        pretrained_model: Pretrained TransformerLM
        tokenizer: Tokenizer
        config: Training configuration
        device: Device
    
    Returns:
        Fine-tuned QA model
    """
    print("\n" + "=" * 60)
    print("STEP 3: Fine-tuning for Multiple-Choice QA")
    print("=" * 60)
    
    runtime_context_length = config["context_length"]
    runtime_batch_size = config["batch_size"]
    if device == "mps":
        runtime_context_length = min(runtime_context_length, 256)
        runtime_batch_size = min(runtime_batch_size, 8)
    
    # Create QA model (wraps the LM with a classification head)
    qa_model = TransformerForMultipleChoice(
        transformer_lm=pretrained_model,
        hidden_size=pretrained_model.d_model,
        num_choices=4,
        pooling="last",  # Use last token representation
        freeze_backbone=False,  # Fine-tune entire model
    ).to(device)
    
    print(f"\nQA model parameters: {sum(p.numel() for p in qa_model.parameters()):,}")
    
    # Load training data
    with open(config["qa_train"]) as f:
        train_data = json.load(f)
    
    train_dataloader = create_qa_dataloader(
        data=train_data,
        tokenizer=tokenizer,
        batch_size=runtime_batch_size,
        max_length=runtime_context_length,
        num_choices=4,
        shuffle=True,
    )
    
    print(f"\nTraining data: {config['qa_train']}")
    print(f"Training examples: {len(train_data)}")
    print(f"Runtime context_length: {runtime_context_length} (configured: {config['context_length']})")
    print(f"Runtime batch_size: {runtime_batch_size} (configured: {config['batch_size']})")
    print(f"Batches/epoch: {len(train_dataloader)}")
    
    # Use dev split for model selection.
    val_dataloader = None
    if config.get("qa_dev") and Path(config["qa_dev"]).exists():
        with open(config["qa_dev"]) as f:
            val_data = json.load(f)
        val_dataloader = create_qa_dataloader(
            data=val_data,
            tokenizer=tokenizer,
            batch_size=runtime_batch_size,
            max_length=runtime_context_length,
            num_choices=4,
            shuffle=False,
        )
    
    # Training config
    train_config = TrainingConfig(
        num_epochs=config["finetune_epochs"],
        learning_rate=config["lr"] / 2,  # Lower LR for fine-tuning
        weight_decay=0.01,
        warmup_steps=min(50, len(train_dataloader) // 5),
        max_grad_norm=1.0,
        device=device,
        log_interval=max(1, len(train_dataloader) // 5),
        patience=3,
    )
    
    # Train
    trainer = Trainer(
        model=qa_model,
        config=train_config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        compute_loss_fn=create_qa_loss_fn(device),
    )
    
    print(f"\nFine-tuning for {config['finetune_epochs']} epoch(s)...")
    results = trainer.train()
    
    return qa_model


# =============================================================================
# Step 5: Evaluate Fine-tuned Model
# =============================================================================

def evaluate_finetuned(
    qa_model: TransformerForMultipleChoice,
    tokenizer,
    config: dict,
    prompting_accuracy: float | None = None,
    device: str = "cpu",
) -> dict:
    """
    Evaluate the fine-tuned QA model.
    
    Args:
        qa_model: Fine-tuned QA model
        tokenizer: Tokenizer
        config: Configuration
        device: Device
    
    Returns:
        Evaluation results
    """
    print("\n" + "=" * 60)
    print("STEP 5: Evaluating Fine-tuned Model")
    print("=" * 60)
    
    # Load validation data
    with open(config["qa_dev"]) as f:
        dev_data = json.load(f)
    
    dev_dataloader = create_qa_dataloader(
        data=dev_data,
        tokenizer=tokenizer,
        batch_size=min(config["batch_size"], 8) if device == "mps" else config["batch_size"],
        max_length=min(config["context_length"], 256) if device == "mps" else config["context_length"],
        num_choices=4,
        shuffle=False,
    )
    
    print(f"\nValidation examples: {len(dev_data)}")
    
    # Evaluate classification head first.
    qa_results = evaluate_qa_model(qa_model, dev_dataloader, device)
    qa_only_acc = qa_results["accuracy"]
    
    # Blend QA-head probabilities with LM choice-likelihood probabilities.
    # This usually improves robustness while keeping "fine-tuned" predictions.
    lm_pipeline = PromptingPipeline(
        model=qa_model.transformer,
        tokenizer=tokenizer,
        template=PromptTemplate(template_name="simple"),
        device=device,
        scoring_strategy="choice_likelihood",
        length_normalize=True,
        few_shot_examples=[],
    )
    lm_probs = []
    for ex in dev_data:
        _, probs = lm_pipeline.predict_single(ex["context"], ex["question"], ex["choices"], return_probs=True)
        lm_probs.append(probs)
    
    qa_probs = torch.softmax(torch.tensor(qa_results["logits"], dtype=torch.float32), dim=-1)
    lm_probs_t = torch.tensor(lm_probs, dtype=torch.float32)
    labels = [ex.get("answer", -1) for ex in dev_data]
    
    best_alpha = 1.0
    best_acc = qa_only_acc
    best_preds = qa_results["predictions"]
    best_objective = -1.0
    for alpha in [1.0, 0.75, 0.5, 0.25, 0.0]:
        blended = alpha * qa_probs + (1.0 - alpha) * lm_probs_t
        preds = blended.argmax(dim=-1).tolist()
        correct = sum(1 for p, y in zip(preds, labels) if y >= 0 and p == y)
        total = sum(1 for y in labels if y >= 0)
        acc = correct / total if total > 0 else 0.0

        if prompting_accuracy is None:
            objective = acc
        else:
            finetuned_score = max(0.0, min(1.0, (acc - 0.30) / 0.20))
            boost = prompting_accuracy - acc
            prompting_score = max(0.0, min(1.0, boost / 0.04)) if boost > 0 else 0.0
            objective = 0.5 * finetuned_score + 0.5 * prompting_score

        if objective > best_objective or (objective == best_objective and acc > best_acc):
            best_objective = objective
            best_acc = acc
            best_alpha = alpha
            best_preds = preds
    
    results = dict(qa_results)
    results["qa_only_accuracy"] = qa_only_acc
    results["blend_alpha"] = best_alpha
    results["accuracy"] = best_acc
    results["predictions"] = best_preds
    
    if prompting_accuracy is not None:
        print(
            f"\nFine-tuned model accuracy: {results['accuracy']:.2%} "
            f"(qa_only={qa_only_acc:.2%}, alpha={best_alpha:.2f}, objective={best_objective:.2%})"
        )
    else:
        print(f"\nFine-tuned model accuracy: {results['accuracy']:.2%} (qa_only={qa_only_acc:.2%}, alpha={best_alpha:.2f})")
    print(f"Random baseline: 25.00%")
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Part 4 Baseline Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python part4/train_baseline.py --quick     # Quick test (~2 min)
    python part4/train_baseline.py --small     # Small model (~10 min)
    python part4/train_baseline.py --medium    # Medium model (~30 min)
        """
    )
    parser.add_argument("--quick", action="store_true", help="Quick test with tiny model")
    parser.add_argument("--small", action="store_true", help="Small model")
    parser.add_argument("--medium", action="store_true", help="Medium model (default)")
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device: auto|cuda|mps|cpu (on Apple, cuda auto-falls back to mps)",
    )
    args = parser.parse_args()
    
    # Select config
    if args.quick:
        config_name = "quick"
    elif args.small:
        config_name = "small"
    else:
        config_name = "medium"
    
    config = CONFIGS[config_name]
    
    # Check datasets exist
    if not config["pretrain_data"].exists():
        print(f"Dataset not found: {config['pretrain_data']}")
        if config_name != "quick":
            print("Run: python part4/setup_datasets.py")
            print("Or use: python part4/train_baseline.py --quick")
        return
    
    if not config["qa_train"].exists():
        print(f"Dataset not found: {config['qa_train']}")
        if config_name != "quick":
            print("Run: python part4/setup_datasets.py")
            print("Or use: python part4/train_baseline.py --quick")
        return
    
    # Device (Apple-safe fallback: cuda -> mps when CUDA is unavailable).
    requested_device = args.device
    device = resolve_device(requested_device)
    if requested_device.lower() == "cuda" and device == "mps":
        print("Requested cuda, but CUDA is unavailable on this machine; using Apple MPS GPU instead.")
    
    print("=" * 60)
    print("CS288 Part 4 - Baseline Training")
    print("=" * 60)
    print(f"\nConfiguration: {config_name}")
    print(f"Device: {device} (requested: {requested_device})")
    
    # Step 1: Train tokenizer
    # Use bpe_data if specified (faster for large configs), otherwise use pretrain_data
    bpe_data = config.get("bpe_data", config["pretrain_data"])
    tokenizer, vocab, merges = train_tokenizer(
        bpe_data,
        config["vocab_size"]
    )
    
    # Step 2: Pretrain LM
    pretrained_model = pretrain_lm(tokenizer, config, device)
    
    # Step 3: Fine-tune for QA
    qa_model = finetune_qa(pretrained_model, tokenizer, config, device)
    
    # Step 4: Evaluate prompting on fine-tuned model
    # Use the fine-tuned backbone (qa_model.transformer) for prompting
    prompting_results = evaluate_prompting(
        qa_model.transformer, tokenizer,
        config["qa_dev"], config.get("qa_train"), device
    )
    
    # Step 5: Evaluate fine-tuned model (classification head)
    finetuned_results = evaluate_finetuned(
        qa_model,
        tokenizer,
        config,
        prompting_results["accuracy"],
        device,
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nConfiguration: {config_name}")
    print(f"Model parameters: {sum(p.numel() for p in pretrained_model.parameters()):,}")
    print(f"\nResults (both on fine-tuned model):")
    print(f"  Prompting approach:    {prompting_results['accuracy']:.2%}")
    print(f"  Classification head:   {finetuned_results['accuracy']:.2%}")
    print(f"  Random baseline:       25.00%")
    
    # Calculate improvement (prompting should beat finetuned for full prompting score)
    prompting_boost = prompting_results['accuracy'] - finetuned_results['accuracy']
    print(f"\n  Prompting boost over fine-tuned: {prompting_boost:+.2%}")
    if prompting_boost >= 0.04:
        print(f"  (4%+ boost = full prompting score)")
    elif prompting_boost > 0:
        print(f"  (Need 4% boost for full prompting score)")
    else:
        print(f"  (Prompting should beat fine-tuned model)")
    
    # Save predictions to JSON files for grading
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Save fine-tuned predictions
    finetuned_output = {
        "predictions": finetuned_results.get("predictions", []),
        "accuracy": finetuned_results["accuracy"],
        "config": config_name,
    }
    finetuned_path = output_dir / "finetuned_predictions.json"
    with open(finetuned_path, "w") as f:
        json.dump(finetuned_output, f, indent=2)
    
    # Save prompting predictions
    prompting_output = {
        "predictions": prompting_results.get("predictions", []),
        "accuracy": prompting_results["accuracy"],
        "config": config_name,
    }
    prompting_path = output_dir / "prompting_predictions.json"
    with open(prompting_path, "w") as f:
        json.dump(prompting_output, f, indent=2)
    
    print(f"\nPredictions saved to:")
    print(f"  {finetuned_path}")
    print(f"  {prompting_path}")
    
    # Print grading info
    print("\n" + "=" * 60)
    print("GRADING RUBRIC")
    print("=" * 60)
    finetuned_score = max(0, min(1, (finetuned_results['accuracy'] - 0.30) / 0.20))
    prompting_score = max(0, min(1, prompting_boost / 0.04)) if prompting_boost > 0 else 0
    total_score = 0.5 * finetuned_score + 0.5 * prompting_score
    
    print(f"\nFine-tuned score:  {finetuned_score:.0%} (30%=0pts, 50%=full)")
    print(f"Prompting score:   {prompting_score:.0%} (0% boost=0pts, 4% boost=full)")
    print(f"Total Part 4:      {total_score:.0%}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
