import torch
from transformers import Wav2Vec2Processor
from dataclasses import dataclass
from typing import Dict, List, Union, Optional


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        jyutping_label_features = [
            {"input_ids": feature["jyutping_labels"]} for feature in features
        ]
        tone_label_features = [
            {"input_ids": feature["tone_labels"]} for feature in features
        ]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        jyutping_labels_batch = self.processor.pad(
            labels=jyutping_label_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        tone_labels_batch = self.processor.pad(
            labels=tone_label_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        jyutping_labels = jyutping_labels_batch["input_ids"].masked_fill(
            jyutping_labels_batch.attention_mask.ne(1), -100
        )
        tone_labels = tone_labels_batch["input_ids"].masked_fill(
            tone_labels_batch.attention_mask.ne(1), -100
        )

        batch["jyutping_labels"] = jyutping_labels
        batch["tone_labels"] = tone_labels

        return batch
