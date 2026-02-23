"""
Prompting utilities for multiple-choice QA.
Example submission.
"""
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from part3.nn_utils import softmax
from part4.device_utils import resolve_device


class PromptTemplate:
    TEMPLATES = {
        "basic": "Context: {context}\n\nQuestion: {question}\n\nChoices:\n{choices_formatted}\n\nAnswer:",
        "instruction": "Read the following passage and answer the question.\n\nPassage: {context}\n\nQuestion: {question}\n\n{choices_formatted}\n\nSelect the letter:",
        "simple": "{context}\n{question}\n{choices_formatted}\nThe answer is",
        "choice_only": "Context: {context}\n\nQuestion: {question}\n\nAnswer:",
    }
    
    def __init__(self, template_name: str = "basic", custom_template: Optional[str] = None, choice_format: str = "letter"):
        self.template = custom_template if custom_template else self.TEMPLATES.get(template_name, self.TEMPLATES["basic"])
        self.choice_format = choice_format
    
    def _format_choices(self, choices: List[str]) -> str:
        labels = ["A", "B", "C", "D", "E", "F", "G", "H"] if self.choice_format == "letter" else [str(i+1) for i in range(len(choices))]
        return "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
    
    def format(self, context: str, question: str, choices: List[str], **kwargs) -> str:
        return self.template.format(context=context, question=question, choices_formatted=self._format_choices(choices), **kwargs)
    
    def format_with_answer(self, context: str, question: str, choices: List[str], answer_idx: int) -> str:
        prompt = self.format(context, question, choices)
        label = chr(ord('A') + answer_idx) if self.choice_format == "letter" else str(answer_idx + 1)
        return f"{prompt} {label}"


class PromptingPipeline:
    def __init__(
        self,
        model,
        tokenizer,
        template: Optional[PromptTemplate] = None,
        device: str = "mps",
        scoring_strategy: str = "choice_likelihood",
        length_normalize: bool = True,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    ):
        resolved_device = resolve_device(device)
        self.model = model.to(resolved_device) if hasattr(model, 'to') else model
        self.tokenizer = tokenizer
        self.template = template or PromptTemplate("basic")
        self.device = resolved_device
        self.scoring_strategy = scoring_strategy
        self.length_normalize = length_normalize
        self.few_shot_examples = few_shot_examples or []
        self._setup_choice_tokens()
    
    def _setup_choice_tokens(self):
        self.choice_tokens = {}
        for label in ["A", "B", "C", "D"]:
            for prefix in ["", " "]:
                token_ids = self.tokenizer.encode(prefix + label)
                if token_ids:
                    self.choice_tokens[label] = token_ids[-1]
                    break
    
    def _build_prompt(self, context: str, question: str, choices: List[str]) -> str:
        segments = []
        for ex in self.few_shot_examples:
            if "answer" not in ex or ex["answer"] is None or ex["answer"] < 0:
                continue
            segments.append(
                self.template.format_with_answer(
                    ex["context"],
                    ex["question"],
                    ex["choices"],
                    ex["answer"],
                )
            )
        segments.append(self.template.format(context, question, choices))
        return "\n\n".join(segments)
    
    def _trim_to_context(self, token_ids: List[int]) -> List[int]:
        context_len = getattr(self.model, "context_length", None)
        if context_len is None or len(token_ids) <= context_len:
            return token_ids
        return token_ids[-context_len:]
    
    @torch.no_grad()
    def _choice_log_likelihood(self, prompt_ids: List[int], choice_text: str) -> float:
        choice_ids = self.tokenizer.encode(" " + choice_text)
        if len(choice_ids) == 0:
            return float("-inf")
        
        full_ids = prompt_ids + choice_ids
        full_ids = self._trim_to_context(full_ids)
        # Continuation is always the tail after trimming.
        continuation_len = min(len(choice_ids), max(0, len(full_ids) - 1))
        if continuation_len == 0:
            return float("-inf")
        continuation_ids = full_ids[-continuation_len:]
        
        input_ids = torch.tensor([full_ids], device=self.device, dtype=torch.long)
        logits = self.model(input_ids)[0]  # (seq_len, vocab)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Token at position t is predicted by logits[t-1].
        start = len(full_ids) - continuation_len
        positions = torch.arange(start - 1, len(full_ids) - 1, device=self.device)
        targets = torch.tensor(continuation_ids, device=self.device, dtype=torch.long)
        score = log_probs[positions, targets].sum().item()
        
        if self.length_normalize:
            score /= max(1, len(continuation_ids))
        return score
    
    @torch.no_grad()
    def predict_single(self, context: str, question: str, choices: List[str], return_probs: bool = False):
        self.model.eval()
        prompt = self._build_prompt(context, question, choices)
        prompt_ids = self._trim_to_context(self.tokenizer.encode(prompt))
        
        choice_scores = []
        if self.scoring_strategy == "choice_likelihood":
            for choice in choices:
                choice_scores.append(self._choice_log_likelihood(prompt_ids, choice))
        else:
            input_ids = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)
            logits = self.model(input_ids)[:, -1, :]
            choice_labels = ["A", "B", "C", "D"][:len(choices)]
            for label in choice_labels:
                if label in self.choice_tokens:
                    choice_scores.append(logits[0, self.choice_tokens[label]].item())
                else:
                    choice_scores.append(float("-inf"))
        
        choice_logits = torch.tensor(choice_scores, device=self.device)
        probs = softmax(choice_logits, dim=-1).cpu()
        prediction = probs.argmax().item()
        
        if return_probs:
            return prediction, probs.tolist()
        return prediction
    
    @torch.no_grad()
    def predict_batch(self, examples: List[Dict[str, Any]], batch_size: int = 8) -> List[int]:
        return [self.predict_single(ex["context"], ex["question"], ex["choices"]) for ex in examples]


def evaluate_prompting(pipeline, examples: List[Dict[str, Any]], batch_size: int = 8) -> Dict[str, Any]:
    predictions = pipeline.predict_batch(examples, batch_size)
    correct = sum(1 for p, ex in zip(predictions, examples) if ex.get("answer", -1) >= 0 and p == ex["answer"])
    total = sum(1 for ex in examples if ex.get("answer", -1) >= 0)
    return {"accuracy": correct / total if total > 0 else 0.0, "predictions": predictions}
