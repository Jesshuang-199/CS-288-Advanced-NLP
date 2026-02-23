"""
Neural network utilities for Transformer implementation.
Contains basic building blocks: softmax, cross-entropy, gradient clipping, token accuracy, perplexity.
"""
import torch
from torch import Tensor


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Compute softmax along the specified dimension.
    
    Args:
        x: Input tensor of any shape
        dim: Dimension along which to compute softmax (default: -1)
    
    Returns:
        Tensor of same shape as input with softmax applied along dim
    """
    # TODO: Implement numerically stable softmax. You can re-use the same one
    # used in part 2. But for this problem, you need to implement a numerically stable version to pass harder tests.
    max_x = torch.amax(x, dim=dim, keepdim=True)
    max_x = torch.where(torch.isfinite(max_x), max_x, torch.zeros_like(max_x))
    shifted = x - max_x
    exp_x = torch.exp(shifted)
    exp_x = torch.where(torch.isfinite(x), exp_x, torch.zeros_like(exp_x))
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    return torch.where(sum_exp > 0, exp_x / sum_exp, torch.zeros_like(exp_x))


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Compute cross-entropy loss.
    
    Args:
        logits: Unnormalized log probabilities of shape (N, C) where N is batch size
                and C is number of classes
        targets: Ground truth class indices of shape (N,)
    
    Returns:
        Scalar tensor containing the mean cross-entropy loss
    """
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    n = targets.shape[0]
    return -log_probs[torch.arange(n, device=logits.device), targets].mean()


def gradient_clipping(parameters, max_norm: float) -> Tensor:
    """
    Clip gradients of parameters by global norm.
    
    Args:
        parameters: Iterable of parameters with gradients
        max_norm: Maximum allowed gradient norm
    
    Returns:
        The total norm of the gradients before clipping
    """
    params = [p for p in parameters if p is not None and p.grad is not None]
    if not params:
        return torch.tensor(0.0)
    
    grads = [p.grad.detach() for p in params]
    total_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)
    clip_coef = max_norm / (total_norm + 1e-6)
    
    if clip_coef < 1:
        for p in params:
            p.grad.detach().mul_(clip_coef)
    
    return total_norm


def token_accuracy(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    """
    Compute token-level accuracy for language modeling.
    
    Computes the fraction of tokens where the predicted token (argmax of logits)
    matches the target token, ignoring positions where target equals ignore_index.
    
    Args:
        logits: Predicted logits of shape (N, C) where N is the number of tokens
                and C is the vocabulary size
        targets: Ground truth token indices of shape (N,)
        ignore_index: Target value to ignore when computing accuracy (default: -100)
    
    Returns:
        Scalar tensor containing the accuracy (between 0 and 1)
    
    Example:
        >>> logits = torch.tensor([[2.0, 1.0, 0.5], [0.1, 3.0, 0.2], [1.0, 0.5, 2.5]])
        >>> targets = torch.tensor([0, 1, 2])
        >>> token_accuracy(logits, targets)
        tensor(1.)  # All predictions correct: argmax gives [0, 1, 2]
        
        >>> logits = torch.tensor([[2.0, 1.0], [0.1, 3.0], [1.0, 0.5]])
        >>> targets = torch.tensor([1, 1, 0])
        >>> token_accuracy(logits, targets)
        tensor(0.6667)  # 2 out of 3 correct
    """
    preds = logits.argmax(dim=-1)
    valid_mask = targets != ignore_index
    valid_count = valid_mask.sum()
    
    if valid_count == 0:
        return torch.tensor(0.0, device=logits.device)
    
    correct = (preds[valid_mask] == targets[valid_mask]).float()
    return correct.mean()


def perplexity(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    """
    Compute perplexity for language modeling.
    
    Perplexity is defined as exp(cross_entropy_loss). It measures how well the
    probability distribution predicted by the model matches the actual distribution
    of the tokens. Lower perplexity indicates better prediction.
    
    Args:
        logits: Predicted logits of shape (N, C) where N is the number of tokens
                and C is the vocabulary size
        targets: Ground truth token indices of shape (N,)
        ignore_index: Target value to ignore when computing perplexity (default: -100)
    
    Returns:
        Scalar tensor containing the perplexity (always >= 1)
    
    Example:
        >>> # Perfect predictions (one-hot logits matching targets)
        >>> logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        >>> targets = torch.tensor([0, 1, 2])
        >>> perplexity(logits, targets)
        tensor(1.0001)  # Close to 1 (perfect)
        
        >>> # Uniform predictions (high uncertainty)
        >>> logits = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        >>> targets = torch.tensor([0, 1, 2])
        >>> perplexity(logits, targets)
        tensor(3.)  # Equal to vocab_size (worst case for uniform)
    """
    valid_mask = targets != ignore_index
    if valid_mask.sum() == 0:
        return torch.tensor(float("inf"), device=logits.device)
    
    ce = cross_entropy(logits[valid_mask], targets[valid_mask])
    return torch.exp(ce)
