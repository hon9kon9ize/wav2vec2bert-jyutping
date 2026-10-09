"""Single-head nucleus+tone CTC model with an auxiliary tone head.

Decoding uses only ``lm_head`` (no stream syncing). ``tone_head`` is a second CTC
head over tone digits that exists only as a training loss, so tone statistics are
shared across all rimes while the output stays one monotonic token stream.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import Wav2Vec2BertModel, Wav2Vec2BertPreTrainedModel
from transformers.modeling_outputs import ModelOutput


@dataclass
class JyutpingAuxCTCOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    tone_logits: torch.FloatTensor = None
    main_loss: Optional[torch.FloatTensor] = None
    tone_loss: Optional[torch.FloatTensor] = None


class Wav2Vec2BertForJyutpingCTCAuxTone(Wav2Vec2BertPreTrainedModel):
    def __init__(self, config, tone_vocab_size: int = 9, aux_tone_weight: float = 0.3):
        super().__init__(config)

        self.wav2vec2_bert = Wav2Vec2BertModel(config)
        self.dropout = nn.Dropout(config.final_dropout)
        hidden = (
            config.output_hidden_size
            if getattr(config, "add_adapter", False)
            else config.hidden_size
        )
        self.lm_head = nn.Linear(hidden, config.vocab_size)
        self.tone_head = nn.Linear(hidden, tone_vocab_size)
        self.tone_vocab_size = tone_vocab_size
        self.aux_tone_weight = aux_tone_weight

        self.post_init()

    def _ctc(self, logits, labels, input_lengths):
        mask = labels >= 0
        log_probs = nn.functional.log_softmax(
            logits, dim=-1, dtype=torch.float32
        ).transpose(0, 1)
        with torch.backends.cudnn.flags(enabled=False):
            return nn.functional.ctc_loss(
                log_probs,
                labels.masked_select(mask),
                input_lengths,
                mask.sum(-1),
                blank=self.config.pad_token_id,
                reduction=self.config.ctc_loss_reduction,
                zero_infinity=self.config.ctc_zero_infinity,
            )

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        tone_labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, JyutpingAuxCTCOutput]:
        hidden_states = self.wav2vec2_bert(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)
        tone_logits = self.tone_head(hidden_states)

        loss = main_loss = tone_loss = None
        if labels is not None and tone_labels is not None:
            if attention_mask is None:
                attention_mask = torch.ones(
                    input_features.shape[:2],
                    device=input_features.device,
                    dtype=torch.long,
                )
            input_lengths = self._get_feat_extract_output_lengths(
                attention_mask.sum(-1)
            ).to(torch.long)
            main_loss = self._ctc(logits, labels, input_lengths)
            tone_loss = self._ctc(tone_logits, tone_labels, input_lengths)
            loss = main_loss + self.aux_tone_weight * tone_loss

        return JyutpingAuxCTCOutput(
            loss=loss,
            logits=logits,
            tone_logits=tone_logits,
            main_loss=main_loss,
            tone_loss=tone_loss,
        )
