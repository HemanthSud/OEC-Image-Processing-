from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, Optional
import pickle
from pathlib import Path


class NGramModel:
    """
    N-gram language model

    Learn n-gram probability distributions from index sequences.
    Supports start token (-1) and end token (-2).

    Improvements:
    - Supports predefined vocabulary (initial_vocab)
    - Supports model save/load
    """

    def __init__(self,
                 n: int = 3,
                 k: float = 0.00001,
                 start_token: int = -1,
                 end_token: int = -2,
                 initial_vocab: Optional[Set[int]] = None):
        """
        Initialize N-gram model.

        Args:
            n: n-gram order
            k: additive smoothing factor
            start_token: start symbol index
            end_token: end symbol index
            initial_vocab: predefined vocabulary set
        """
        self.n = n
        self.k = k
        self.start_token = start_token
        self.end_token = end_token
        self.initial_vocab = initial_vocab
        self.vocab = set()
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.prob_distribution = {}

    def fit(self, sequences: List[List[int]]):
        """
        Train the model.

        Args:
            sequences: list of index sequences
        """
        processed_sequences = []
        for seq in sequences:
            padded_seq = [self.start_token] * (self.n - 1) + seq + [self.end_token]
            processed_sequences.append(padded_seq)

        all_indices = []
        for seq in processed_sequences:
            all_indices.extend(seq)

        if self.initial_vocab is not None:
            self.vocab = set(self.initial_vocab)
            self.vocab.update(all_indices)
        else:
            self.vocab = set(all_indices)

        V = len(self.vocab)

        for seq in processed_sequences:
            for i in range(len(seq) - self.n + 1):
                ngram = tuple(seq[i:i + self.n])
                context = ngram[:-1]

                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

        self.prob_distribution = defaultdict(dict)

        for context in self.context_counts.keys():
            context_count = self.context_counts[context]
            for next_char in self.vocab:
                ngram = context + (next_char,)
                ngram_count = self.ngram_counts.get(ngram, 0)

                numerator = ngram_count + self.k
                denominator = context_count + self.k * V

                probability = numerator / denominator
                self.prob_distribution[context][next_char] = probability

    def save(self, filepath: str):
        """
        Save model using pickle.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'n': self.n,
            'k': self.k,
            'start_token': self.start_token,
            'end_token': self.end_token,
            'initial_vocab': self.initial_vocab,
            'vocab': self.vocab,
            'ngram_counts': self.ngram_counts,
            'context_counts': self.context_counts,
            'prob_distribution': dict(self.prob_distribution)
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Model saved to: {filepath}")
        print(f"File size: {filepath.stat().st_size / 1024:.2f} KB")

    @classmethod
    def load(cls, filepath: str) -> 'NGramModel':
        """
        Load model from pickle.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model does not exist: {filepath}")

        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)

        model = cls(
            n=save_dict['n'],
            k=save_dict['k'],
            start_token=save_dict['start_token'],
            end_token=save_dict['end_token'],
            initial_vocab=save_dict['initial_vocab']
        )

        model.vocab = save_dict['vocab']
        model.ngram_counts = save_dict['ngram_counts']
        model.context_counts = save_dict['context_counts']
        model.prob_distribution = defaultdict(dict, save_dict['prob_distribution'])

        print(f"Model Loaded: {filepath}")
        print(f"n={model.n}, k={model.k}, vocab_size={len(model.vocab)}")

        return model

    def get_probability_distribution(self) -> Dict[Tuple, Dict[int, float]]:
        """
        Return probability distribution.
        """
        return dict(self.prob_distribution)

    def get_next_char_prob(self, context: Tuple[int, ...]) -> Dict[int, float]:
        """
        Get probability distribution for next char.
        """
        if context in self.prob_distribution:
            return self.prob_distribution[context]
        else:
            V = len(self.vocab)
            return {char: 1.0 / V for char in self.vocab}

    def predict_next(self, context: Tuple[int, ...]) -> int:
        """
        Predict next character.
        """
        prob_dist = self.get_next_char_prob(context)
        return max(prob_dist.items(), key=lambda x: x[1])[0]

    def get_probability(self, ngram: Tuple[int, ...]) -> float:
        """
        Get probability of an n-gram.
        """
        if len(ngram) != self.n:
            raise ValueError(f"ngram length must be {self.n}")

        context = ngram[:-1]
        next_char = ngram[-1]

        if context in self.prob_distribution:
            return self.prob_distribution[context].get(next_char, 0.0)
        else:
            return 1.0 / len(self.vocab)

    def get_start_context(self) -> Tuple[int, ...]:
        """
        Return (n-1) start tokens.
        """
        return tuple([self.start_token] * (self.n - 1))

    def is_end_token(self, token: int) -> bool:
        """
        Check if token is end symbol.
        """
        return token == self.end_token

    def get_model_info(self) -> Dict:
        """
        Return summary info.
        """
        return {
            'n': self.n,
            'k': self.k,
            'vocab_size': len(self.vocab),
            'num_unique_ngrams': len(self.ngram_counts),
            'num_unique_contexts': len(self.context_counts),
            'has_initial_vocab': self.initial_vocab is not None,
            'start_token': self.start_token,
            'end_token': self.end_token
        }
