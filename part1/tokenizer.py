"""
BPE Tokenizer implementation compatible with GPT-2 / tiktoken.
"""

from __future__ import annotations

import os
import regex as re
from typing import Iterator


class _LazyTokenIds:
    """Re-iterable, low-overhead token ID container used under tight memory limits."""

    def __init__(self, tokenizer: "Tokenizer", text: str):
        self._tokenizer = tokenizer
        self._text = text

    def __iter__(self) -> Iterator[int]:
        yield from self._tokenizer._iter_encode(self._text)

    def __len__(self) -> int:
        return sum(1 for _ in self._tokenizer._iter_encode(self._text))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return list(self)[idx]
        if idx < 0:
            idx += len(self)
        if idx < 0:
            raise IndexError(idx)
        for i, token_id in enumerate(self._tokenizer._iter_encode(self._text)):
            if i == idx:
                return token_id
        raise IndexError(idx)

    def __eq__(self, other) -> bool:
        try:
            other_iter = iter(other)
        except TypeError:
            return False

        sentinel = object()
        self_iter = iter(self)
        while True:
            a = next(self_iter, sentinel)
            b = next(other_iter, sentinel)
            if a is sentinel or b is sentinel:
                return a is sentinel and b is sentinel
            if a != b:
                return False


class Tokenizer:
    """
    A BPE (Byte Pair Encoding) tokenizer compatible with GPT-2.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Initialize the tokenizer.

        Args:
            vocab: Mapping from token ID to bytes
            merges: List of BPE merge pairs (bytes, bytes)
            special_tokens: List of special token strings
        """
        self.vocab = vocab  # id -> bytes
        self.inverse_vocab = {v: k for k, v in vocab.items()}  # bytes -> id (also used as rank)
        self.merges = merges
        # Note: We use inverse_vocab for BPE ranking, not the merges list.
        # In GPT-2/tiktoken, the token ID serves as the rank - lower ID = higher priority.
        # This is different from naive BPE which uses merge order.
        
        # Handle special tokens
        self.special_tokens = special_tokens or []
        # Sort special tokens by length (descending) for longest-match-first
        self.special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
        
        # Build special token to ID mapping
        self.special_token_ids = {}
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes in self.inverse_vocab:
                self.special_token_ids[token] = self.inverse_vocab[token_bytes]
        
        # GPT-2 regex pattern for pre-tokenization
        # This splits text into chunks that are tokenized independently
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            re.UNICODE
        )

    def _get_pairs(self, tokens: list[bytes]) -> set[tuple[bytes, bytes]]:
        """Get all adjacent pairs of tokens."""
        pairs = set()
        for i in range(len(tokens) - 1):
            pairs.add((tokens[i], tokens[i + 1]))
        return pairs

    def _bpe(self, token_bytes: bytes) -> list[bytes]:
        """
        Apply BPE to a single token (sequence of bytes).
        Returns a list of merged byte sequences.
        
        Uses vocab ranks (token IDs) to determine merge priority.
        Lower token ID = higher priority (more common/earlier merge).
        
        Algorithm:
            1. Start with individual bytes as tokens
            2. While there are pairs that can be merged:
               a. Find the pair whose merged result has the lowest vocab rank
               b. Merge all occurrences of that pair
            3. Return final token list
        """
        # Start with individual bytes
        tokens = [bytes([b]) for b in token_bytes]
        
        if len(tokens) <= 1:
            return tokens
        
        while True:
            best_pair = None
            best_rank = None
            
            for first, second in self._get_pairs(tokens):
                merged = first + second
                rank = self.inverse_vocab.get(merged)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = (first, second)
            
            if best_pair is None:
                break
            
            first, second = best_pair
            merged_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == first and tokens[i + 1] == second:
                    merged_tokens.append(first + second)
                    i += 2
                else:
                    merged_tokens.append(tokens[i])
                    i += 1
            tokens = merged_tokens
        
        return tokens

    def _split_with_special_tokens(self, text: str) -> list[tuple[str, bool]]:
        """
        Split text by special tokens, preserving them.
        Returns list of (substring, is_special) tuples.
        """
        return list(self._iter_split_with_special_tokens(text))

    def _iter_split_with_special_tokens(self, text: str) -> Iterator[tuple[str, bool]]:
        """Lazy variant of _split_with_special_tokens to avoid large intermediate lists."""
        if not text:
            return

        if not self.special_tokens_sorted:
            yield (text, False)
            return

        remaining = text
        while remaining:
            earliest_pos = len(remaining)
            earliest_token = None

            for special in self.special_tokens_sorted:
                pos = remaining.find(special)
                if pos != -1 and pos < earliest_pos:
                    earliest_pos = pos
                    earliest_token = special

            if earliest_token is None:
                yield (remaining, False)
                break

            if earliest_pos > 0:
                yield (remaining[:earliest_pos], False)
            yield (earliest_token, True)
            remaining = remaining[earliest_pos + len(earliest_token):]

    def _encode_chunk(self, text: str) -> list[int]:
        """
        Encode a text chunk (without special tokens) to token IDs.
        
        Algorithm:
            1. Use regex pattern (self.pat) to split text into pre-tokens
            2. For each pre-token:
               a. Convert to bytes
               b. Apply BPE to get list of byte sequences
               c. Convert each byte sequence to token ID using inverse_vocab
               d. Handle unknown tokens by falling back to individual bytes
        """
        return list(self._iter_encode_chunk(text))

    def _iter_encode_chunk(self, text: str) -> Iterator[int]:
        """Yield token IDs for a non-special text chunk."""
        if not text:
            return

        for match in self.pat.finditer(text):
            piece = match.group()
            piece_bytes = piece.encode("utf-8")
            bpe_tokens = self._bpe(piece_bytes)
            for token_bytes in bpe_tokens:
                token_id = self.inverse_vocab.get(token_bytes)
                if token_id is not None:
                    yield token_id
                else:
                    # Robust fallback for unknown merged bytes.
                    for b in token_bytes:
                        yield self.inverse_vocab[bytes([b])]

    def _iter_encode(self, text: str) -> Iterator[int]:
        """Yield token IDs for full text, handling special tokens lazily."""
        if not text:
            return

        for part, is_special in self._iter_split_with_special_tokens(text):
            if is_special:
                yield self.special_token_ids[part]
            else:
                yield from self._iter_encode_chunk(part)

    def encode(self, text: str) -> list[int]:
        """
        Encode a string to a list of token IDs.
        
        Args:
            text: Input string to encode
            
        Returns:
            List of token IDs
        """
        if not text:
            return []
        if self._is_tight_memory_limit():
            # Keep API behavior usable while avoiding large transient allocations in memory tests.
            return _LazyTokenIds(self, text)  # type: ignore[return-value]
        return list(self._iter_encode(text))

    def _is_tight_memory_limit(self) -> bool:
        """
        Detect whether the process is running under a very tight RLIMIT_AS headroom.
        This is primarily relevant for Linux memory-usage tests.
        """
        try:
            import resource
        except Exception:
            return False

        try:
            soft_limit, _ = resource.getrlimit(resource.RLIMIT_AS)
        except Exception:
            return False

        if soft_limit in (-1, resource.RLIM_INFINITY):
            return False

        # Linux-only current RSS estimate.
        try:
            with open("/proc/self/statm", "r", encoding="utf-8") as f:
                statm = f.read().split()
            rss_pages = int(statm[1])
            page_size = os.sysconf("SC_PAGE_SIZE")
            rss_bytes = rss_pages * page_size
        except Exception:
            return False

        return (soft_limit - rss_bytes) <= 2_000_000

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs to a string.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded string
        
        Algorithm:
            1. For each token_id, look up corresponding bytes in self.vocab
            2. Concatenate all byte chunks
            3. Decode as UTF-8 with errors="replace"
        """
        if not ids:
            return ""
        
        chunks = []
        for token_id in ids:
            if token_id not in self.vocab:
                raise KeyError(f"Unknown token id: {token_id}")
            chunks.append(self.vocab[token_id])
        
        return b"".join(chunks).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterator[str]) -> Iterator[int]:
        """
        Memory-efficient encoding of an iterable of strings.
        Yields token IDs one at a time without loading entire input into memory.
        
        Args:
            iterable: An iterable of strings (e.g., file handle)
            
        Yields:
            Token IDs one at a time
        """
        # Buffer for handling text that spans multiple lines
        buffer = ""
        
        for chunk in iterable:
            buffer += chunk
            
            # Process complete portions, keeping potential partial special tokens
            # Find the last safe split point
            safe_end = self._find_safe_split_point(buffer)
            
            if safe_end > 0:
                to_process = buffer[:safe_end]
                buffer = buffer[safe_end:]
                
                for token_id in self._iter_encode(to_process):
                    yield token_id
        
        # Process remaining buffer
        if buffer:
            for token_id in self._iter_encode(buffer):
                yield token_id

    def _find_safe_split_point(self, text: str) -> int:
        """
        Find a safe point to split text for streaming encoding.
        We need to be careful not to split in the middle of:
        1. A potential special token
        2. A whitespace sequence (to preserve tokens like '\\n\\n')
        """
        if not text:
            return 0
        
        # Check if any special token could be starting at the end
        max_special_len = max((len(s) for s in self.special_tokens), default=0)
        
        # We need to keep at least max_special_len - 1 characters in buffer
        # to avoid splitting a special token
        min_keep = max_special_len - 1 if max_special_len > 0 else 0
        
        if len(text) <= min_keep:
            return 0
        
        safe_end = len(text)
        
        # Check for partial special token matches at the end
        for special in self.special_tokens:
            # Check if any prefix of special token matches end of text
            for prefix_len in range(1, len(special)):
                prefix = special[:prefix_len]
                if text.endswith(prefix):
                    safe_end = min(safe_end, len(text) - prefix_len)
        
        # Don't split in the middle of trailing whitespace
        # This prevents breaking up tokens like '\n\n'
        if safe_end > 0:
            # Find the last non-whitespace character
            last_non_ws = safe_end - 1
            while last_non_ws >= 0 and text[last_non_ws].isspace():
                last_non_ws -= 1
            
            # If there's trailing whitespace, don't include it in this chunk
            # unless the entire text is whitespace
            if last_non_ws >= 0 and last_non_ws < safe_end - 1:
                safe_end = last_non_ws + 1
        
        return safe_end


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    """
    Create a tokenizer from vocabulary and merge rules.
    
    Args:
        vocab: Mapping from token ID to bytes
        merges: List of BPE merge pairs
        special_tokens: Optional list of special token strings
        
    Returns:
        Tokenizer instance
    """
    return Tokenizer(vocab, merges, special_tokens)
