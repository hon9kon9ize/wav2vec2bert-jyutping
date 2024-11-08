from dataclasses import dataclass
from transformers import (
    Wav2Vec2BertModel,
    Wav2Vec2BertPreTrainedModel,
    Wav2Vec2BertProcessor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model,
)
from pycantonese.jyutping.parse_jyutping import ONSETS
import re
from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import (
    _HIDDEN_STATES_START_POSITION,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2AttnAdapterLayer
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerPreTrainedModel,
    Wav2Vec2ConformerModel,
    Wav2Vec2ConformerForCTC,
)
from transformers.modeling_outputs import ModelOutput
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


@dataclass
class JuytpingOutput(ModelOutput):
    """
    Output type of Wav2Vec2BertForCantonese
    """

    loss: Optional[torch.FloatTensor] = None
    jyutping_logits: torch.FloatTensor = None
    tone_logits: torch.FloatTensor = None
    jyutping_loss: Optional[torch.FloatTensor] = None
    tone_loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Wav2Vec2BertForCantonese(Wav2Vec2BertPreTrainedModel):
    """
    Wav2Vec2BertForCantonese is a Wav2Vec2BertModel with a language model head on top (a linear layer on top of the hidden-states output) that outputs Jyutping and tone logits.
    """

    def __init__(
        self,
        config,
        tone_vocab_size: int = 9,
    ):
        super().__init__(config)

        self.wav2vec2_bert = Wav2Vec2BertModel(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.tone_vocab_size = tone_vocab_size

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2BertForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size
            if hasattr(config, "add_adapter") and config.add_adapter
            else config.hidden_size
        )
        self.jyutping_head = nn.Linear(output_hidden_size, config.vocab_size)
        self.tone_head = nn.Linear(output_hidden_size, tone_vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        jyutping_labels: Optional[torch.Tensor] = None,
        tone_labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, JuytpingOutput]:
        if (
            jyutping_labels is not None
            and jyutping_labels.max() >= self.config.vocab_size
        ):
            raise ValueError(
                f"Label values must be <= vocab_size: {self.config.vocab_size}"
            )

        if tone_labels is not None and tone_labels.max() >= self.tone_vocab_size:
            raise ValueError(
                f"Label values must be <= tone_vocab_size: {self.tone_vocab_size}"
            )

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.wav2vec2_bert(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        jyutping_logits = self.jyutping_head(hidden_states)
        tone_logits = self.tone_head(hidden_states)

        loss = None
        if jyutping_labels is not None and tone_labels is not None:
            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask
                if attention_mask is not None
                else torch.ones(
                    input_features.shape[:2],
                    device=input_features.device,
                    dtype=torch.long,
                )
            )
            input_lengths = self._get_feat_extract_output_lengths(
                attention_mask.sum([-1])
            ).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            jyutping_labels_mask = jyutping_labels >= 0
            jyutping_target_lengths = jyutping_labels_mask.sum(-1)
            jyutping_flattened_targets = jyutping_labels.masked_select(
                jyutping_labels_mask
            )

            # ctc_loss doesn't support fp16
            jyutping_log_probs = nn.functional.log_softmax(
                jyutping_logits, dim=-1, dtype=torch.float32
            ).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                jyutping_loss = nn.functional.ctc_loss(
                    jyutping_log_probs,
                    jyutping_flattened_targets,
                    input_lengths,
                    jyutping_target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

            tone_labels_mask = tone_labels >= 0
            tone_target_lengths = tone_labels_mask.sum(-1)
            tone_flattened_targets = tone_labels.masked_select(tone_labels_mask)

            tone_log_probs = nn.functional.log_softmax(
                tone_logits, dim=-1, dtype=torch.float32
            ).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                tone_loss = nn.functional.ctc_loss(
                    tone_log_probs,
                    tone_flattened_targets,
                    input_lengths,
                    tone_target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

            loss = jyutping_loss + tone_loss

        if not return_dict:
            output = (jyutping_logits, tone_logits) + outputs[
                _HIDDEN_STATES_START_POSITION:
            ]
            return ((loss,) + output) if loss is not None else output

        return JuytpingOutput(
            loss=loss,
            jyutping_logits=jyutping_logits,
            tone_logits=tone_logits,
            jyutping_loss=jyutping_loss,
            tone_loss=tone_loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def inference(
        self,
        processor: Wav2Vec2BertProcessor,
        tone_tokenizer: Wav2Vec2CTCTokenizer,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        outputs = self.forward(
            input_features=input_features,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        jyutping_logits = outputs.jyutping_logits
        tone_logits = outputs.tone_logits
        jyutping_pred_ids = torch.argmax(jyutping_logits, dim=-1)
        tone_pred_ids = torch.argmax(tone_logits, dim=-1)
        jyutping_pred = processor.batch_decode(jyutping_pred_ids)[0]
        tone_pred = tone_tokenizer.batch_decode(tone_pred_ids)[0]
        jyutping_list = jyutping_pred.split(" ")
        tone_list = tone_pred.split(" ")
        jyutping_output = []

        for jypt in jyutping_list:
            is_initial = jypt in ONSETS

            if is_initial:
                jypt = "_" + jypt
            else:
                jypt = jypt + "_"

            jyutping_output.append(jypt)

        jyutping_output = re.sub(
            r"\s+", " ", "".join(jyutping_output).replace("_", " ").strip()
        ).split(" ")

        if len(tone_list) > len(jyutping_output):
            tone_list = tone_list[: len(jyutping_output)]
        elif len(tone_list) < len(jyutping_output):
            # repeat the last tone if the length of tone list is shorter than the length of jyutping list
            tone_list = tone_list + [tone_list[-1]] * (
                len(jyutping_output) - len(tone_list)
            )

        return (
            " ".join(
                [f"{jypt}{tone}" for jypt, tone in zip(jyutping_output, tone_list)]
            ),
            jyutping_logits,
            tone_logits,
        )


class Wav2Vec2ForCantonese(Wav2Vec2PreTrainedModel):
    def __init__(
        self, config, tone_vocab_size: int = 9, target_lang: Optional[str] = None
    ):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)

        self.tone_vocab_size = tone_vocab_size
        self.target_lang = target_lang

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2ForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )

        output_hidden_size = (
            config.output_hidden_size
            if hasattr(config, "add_adapter") and config.add_adapter
            else config.hidden_size
        )
        self.jyutping_head = nn.Linear(output_hidden_size, config.vocab_size)
        self.tone_head = nn.Linear(output_hidden_size, tone_vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def tie_weights(self):
        Wav2Vec2ForCTC.tie_weights(self)

    def freeze_feature_extractor(self):
        Wav2Vec2ForCTC.freeze_feature_extractor(self)

    def freeze_feature_encoder(self):
        Wav2Vec2ForCTC.freeze_feature_encoder(self)

    def freeze_base_model(self):
        Wav2Vec2ForCTC.freeze_base_model(self)

    def _get_adapters(self):
        if self.config.adapter_attn_dim is None:
            raise ValueError(
                f"{self.__class__} has no adapter layers. Make sure to define `config.adapter_attn_dim`."
            )

        adapter_weights = {}
        for name, module in self.named_modules():
            if isinstance(module, Wav2Vec2AttnAdapterLayer):
                for param_name, param in module.named_parameters():
                    adapter_weights[".".join([name, param_name])] = param

        if isinstance(self, Wav2Vec2ForCTC):
            for name, param in self.jyutping_head.named_parameters():
                adapter_weights[".".join(["jyutping_head", name])] = param
            for name, param in self.tone_head.named_parameters():
                adapter_weights[".".join(["tone_head", name])] = param

        return adapter_weights

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        jyutping_labels: Optional[torch.Tensor] = None,
        tone_labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, JuytpingOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (
            jyutping_labels is not None
            and jyutping_labels.max() >= self.config.vocab_size
        ):
            raise ValueError(
                f"Label values must be <= vocab_size: {self.config.vocab_size}"
            )

        if tone_labels is not None and tone_labels.max() >= self.tone_vocab_size:
            raise ValueError(
                f"Label values must be <= tone_vocab_size: {self.tone_vocab_size}"
            )

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        jyutping_logits = self.jyutping_head(hidden_states)
        tone_logits = self.tone_head(hidden_states)

        loss = None
        if jyutping_labels is not None and tone_labels is not None:
            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask
                if attention_mask is not None
                else torch.ones(
                    input_values.shape[:2],
                    device=input_values.device,
                    dtype=torch.long,
                )
            )
            input_lengths = self._get_feat_extract_output_lengths(
                attention_mask.sum([-1])
            ).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            jyutping_labels_mask = jyutping_labels >= 0
            jyutping_target_lengths = jyutping_labels_mask.sum(-1)
            jyutping_flattened_targets = jyutping_labels.masked_select(
                jyutping_labels_mask
            )

            # ctc_loss doesn't support fp16
            jyutping_log_probs = nn.functional.log_softmax(
                jyutping_logits, dim=-1, dtype=torch.float32
            ).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                jyutping_loss = nn.functional.ctc_loss(
                    jyutping_log_probs,
                    jyutping_flattened_targets,
                    input_lengths,
                    jyutping_target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

            tone_labels_mask = tone_labels >= 0
            tone_target_lengths = tone_labels_mask.sum(-1)
            tone_flattened_targets = tone_labels.masked_select(tone_labels_mask)

            tone_log_probs = nn.functional.log_softmax(
                tone_logits, dim=-1, dtype=torch.float32
            ).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                tone_loss = nn.functional.ctc_loss(
                    tone_log_probs,
                    tone_flattened_targets,
                    input_lengths,
                    tone_target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

            loss = jyutping_loss + tone_loss

        if not return_dict:
            output = (jyutping_logits, tone_logits) + outputs[
                _HIDDEN_STATES_START_POSITION:
            ]
            return ((loss,) + output) if loss is not None else output

        return JuytpingOutput(
            loss=loss,
            jyutping_logits=jyutping_logits,
            tone_logits=tone_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Wav2Vec2ConformerForCantonese(Wav2Vec2ConformerPreTrainedModel):
    def __init__(
        self, config, tone_vocab_size: int = 9, target_lang: Optional[str] = None
    ):
        super().__init__(config)

        self.wav2vec2_conformer = Wav2Vec2ConformerModel(config)
        self.dropout = nn.Dropout(config.final_dropout)

        self.tone_vocab_size = tone_vocab_size
        self.target_lang = target_lang

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2ConformerForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )

        output_hidden_size = (
            config.output_hidden_size
            if hasattr(config, "add_adapter") and config.add_adapter
            else config.hidden_size
        )

        self.jyutping_head = nn.Linear(output_hidden_size, config.vocab_size)
        self.tone_head = nn.Linear(output_hidden_size, tone_vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_encoder(self):
        Wav2Vec2ConformerForCTC.freeze_feature_encoder(self)

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        jyutping_labels: Optional[torch.Tensor] = None,
        tone_labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, JuytpingOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (
            jyutping_labels is not None
            and jyutping_labels.max() >= self.config.vocab_size
        ):
            raise ValueError(
                f"Label values must be <= vocab_size: {self.config.vocab_size}"
            )

        if tone_labels is not None and tone_labels.max() >= self.tone_vocab_size:
            raise ValueError(
                f"Label values must be <= tone_vocab_size: {self.tone_vocab_size}"
            )

        outputs = self.wav2vec2_conformer(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        jyutping_logits = self.jyutping_head(hidden_states)
        tone_logits = self.tone_head(hidden_states)

        loss = None
        if jyutping_labels is not None and tone_labels is not None:
            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask
                if attention_mask is not None
                else torch.ones(
                    input_values.shape[:2],
                    device=input_values.device,
                    dtype=torch.long,
                )
            )
            input_lengths = self._get_feat_extract_output_lengths(
                attention_mask.sum([-1])
            ).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            jyutping_labels_mask = jyutping_labels >= 0
            jyutping_target_lengths = jyutping_labels_mask.sum(-1)
            jyutping_flattened_targets = jyutping_labels.masked_select(
                jyutping_labels_mask
            )

            # ctc_loss doesn't support fp16
            jyutping_log_probs = nn.functional.log_softmax(
                jyutping_logits, dim=-1, dtype=torch.float32
            ).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                jyutping_loss = nn.functional.ctc_loss(
                    jyutping_log_probs,
                    jyutping_flattened_targets,
                    input_lengths,
                    jyutping_target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

            tone_labels_mask = tone_labels >= 0
            tone_target_lengths = tone_labels_mask.sum(-1)
            tone_flattened_targets = tone_labels.masked_select(tone_labels_mask)

            tone_log_probs = nn.functional.log_softmax(
                tone_logits, dim=-1, dtype=torch.float32
            ).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                tone_loss = nn.functional.ctc_loss(
                    tone_log_probs,
                    tone_flattened_targets,
                    input_lengths,
                    tone_target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

            loss = jyutping_loss + tone_loss

        if not return_dict:
            output = (jyutping_logits, tone_logits) + outputs[
                _HIDDEN_STATES_START_POSITION:
            ]
            return ((loss,) + output) if loss is not None else output

        return JuytpingOutput(
            loss=loss,
            jyutping_logits=jyutping_logits,
            tone_logits=tone_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ == "__main__":
    import torch
    import librosa

    # from transformers import (
    #     SeamlessM4TFeatureExtractor,
    #     Wav2Vec2BertProcessor,
    #     Wav2Vec2CTCTokenizer,
    # )
    # tokenizer = Wav2Vec2CTCTokenizer(
    #     "vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    # )
    # feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
    #     "facebook/w2v-bert-2.0"
    # )
    # processor = Wav2Vec2BertProcessor(
    #     feature_extractor=feature_extractor, tokenizer=tokenizer
    # )
    # wav, sr = librosa.load("/notebooks/projects/wav2vec2-yue/test_nei1.wav", sr=16000)

    # input_features = processor(wav, sampling_rate=sr).input_features[0]

    # model = Wav2Vec2BertForCantonese.from_pretrained(
    #     "facebook/w2v-bert-2.0",
    #     tone_vocab_size=6,
    #     vocab_size=32,
    #     attention_dropout=0.2,
    #     hidden_dropout=0.2,
    #     feat_proj_dropout=0.0,
    #     mask_time_prob=0.0,
    #     layerdrop=0.0,
    #     ctc_loss_reduction="mean",
    #     add_adapter=True,
    #     pad_token_id=processor.tokenizer.pad_token_id,
    # )

    # print("input_features", input_features.shape)
    # print(wav.shape)

    # # Test forward pass
    # input_features = torch.randn(1, 123, 160)
    # jyutping_labels = torch.randint(0, 32, (1, 10))
    # tone_labels = torch.randint(0, 6, (1, 10))
    from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    tokenizer = Wav2Vec2CTCTokenizer(
        "vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    processor = Wav2Vec2Processor(
        feature_extractor=processor.feature_extractor, tokenizer=tokenizer
    )
    model = Wav2Vec2ForCantonese.from_pretrained(
        "TencentGameMate/chinese-hubert-base",
        tone_vocab_size=6,
        vocab_size=32,
        ctc_loss_reduction="mean",
        # pad_token_id=processor.tokenizer.pad_token_id,
        # mask_time_prob=0.0,  # 0.05
        # mask_time_length=10,  # 10
        # mask_feature_prob=0.3,  # 0
        # mask_feature_length=10,  # 10
    )
    # model.freeze_feature_extractor()

    wav, sr = librosa.load(
        "/home/pj24001684/ku40000295/jc/projects/wav2vec2bert-jyutping/test2.wav",
        sr=16000,
    )

    input_values = processor(wav, sampling_rate=sr).input_values[0]
    input_values = torch.from_numpy(input_values).unsqueeze(0)
    # input_values = torch.randn(16000 * 10)
    jyutping_labels = torch.randint(0, 32, (1, 10))
    tone_labels = torch.randint(0, 6, (1, 10))

    output = model(
        input_values,
        jyutping_labels=jyutping_labels,
        tone_labels=tone_labels,
    )

    print(output.loss, output.jyutping_logits.shape, output.tone_logits.shape)
