import random
import numpy as np
from collections import defaultdict

class SequenceTracker:
    def __init__(self):
        # This will hold: 
        # {position: {sequence: {'frequency': count, 'generation': most_recent_generation}}}
        self.data = defaultdict(lambda: defaultdict(lambda: {'frequency': 0, 'generation': 0}))

    def encode_sequence(self, array):
        """
        Convert the NumPy integer array (solution encoding) into a string (or tuple) for comparison.
        Return a string or tuple depending on use case.
        """
        return tuple(array)  # Alternatively: ' '.join(map(str, array))

    def add_sequence(self, position, sequence, generation):
        """
        Add a sequence (in encoded form) at a specific position in a specific generation.
        """
        encoded_sequence = self.encode_sequence(sequence)
        seq_data = self.data[position][encoded_sequence]
        seq_data['frequency'] += 1
        seq_data['generation'] = generation

    def normalize_frequencies(self, position, generation_cutoff):
        """
        Normalize the frequencies of sequences at a given position,
        considering only sequences observed in recent generations.
        """
        total_count = sum(
            seq_data['frequency'] for seq_data in self.data[position].values()
            if seq_data['generation'] >= generation_cutoff
        )
        if total_count == 0:
            return {}  # Avoid division by zero

        # Normalize the frequencies based on recent generation cutoff
        normalized = {
            sequence: seq_data['frequency'] / total_count
            for sequence, seq_data in self.data[position].items()
            if seq_data['generation'] >= generation_cutoff
        }
        return normalized

    def select_sequence(self, position, generation_cutoff):
        """
        Select a sequence at a random position based on observed frequency.
        """
        normalized_frequencies = self.normalize_frequencies(position, generation_cutoff)
        if not normalized_frequencies:
            return None  # No sequences to select from

        # Randomly select based on frequency
        sequences, probabilities = zip(*normalized_frequencies.items())
        selected_sequence = random.choices(sequences, weights=probabilities, k=1)[0]
        # convert the selected sequence back to NumPy integer array
        return np.array(selected_sequence).astype(int)

    def get_sequence_stats(self, position, sequence):
        """
        Get the frequency and most recent generation for a specific sequence at a given position.
        """
        encoded_sequence = self.encode_sequence(sequence)
        seq_data = self.data[position].get(encoded_sequence, None)
        if seq_data:
            return seq_data['frequency'], seq_data['generation']
        else:
            return 0, None

if __name__ == "__main__":
    pass
