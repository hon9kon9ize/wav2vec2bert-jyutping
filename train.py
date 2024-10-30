import os
import argparse
from model import Wav2Vec2BertForCantonese
from data import DataCollatorCTCWithPadding
from datasets import load_metric, load_from_disk
import time
from transformers import (
    Wav2Vec2CTCTokenizer,
    TrainingArguments,
    AddedToken,
    Trainer,
    Wav2Vec2BertProcessor,
    SeamlessM4TFeatureExtractor,
)
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


wer_metric = load_metric("wer")


def train(model_id: str, dataset: str, output_dir: str):
    # load dataset
    ds = load_from_disk(dataset)

    # load tokenizer
    tokenizer = Wav2Vec2CTCTokenizer(
        "vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    tone_tokens = Wav2Vec2CTCTokenizer(
        "tone_vocab.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    # fix token splitting problem
    for key in tokenizer.get_vocab().keys():
        if key not in tokenizer.special_tokens_map.values():
            idx = tokenizer.get_vocab()[key]
            tokenizer._added_tokens_decoder[idx] = AddedToken(
                key, lstrip=False, rstrip=False
            )
    for key in tone_tokens.get_vocab().keys():
        if key not in tone_tokens.special_tokens_map.values():
            idx = tone_tokens.get_vocab()[key]
            tone_tokens._added_tokens_decoder[idx] = AddedToken(
                key, lstrip=False, rstrip=False
            )

    # load processor
    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(model_id)
    processor = Wav2Vec2BertProcessor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    # load model
    model = Wav2Vec2BertForCantonese.from_pretrained(
        model_id,
        attention_dropout=0.2,
        hidden_dropout=0.2,
        feat_proj_dropout=0.0,
        mask_time_prob=0.0,
        layerdrop=0.0,
        add_adapter=True,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    model.config.update(
        {
            "vocab_size": len(tokenizer),
            "tone_vocab_size": len(tone_tokens),
        }
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    tone_tokenizer = Wav2Vec2CTCTokenizer(
        "tone_vocab.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )

    def compute_metrics(pred):
        jyutping_logits = pred.predictions[0]
        tone_logits = pred.predictions[1]
        jyutping_pred_ids = np.argmax(jyutping_logits, axis=-1)
        tone_pred_ids = np.argmax(tone_logits, axis=-1)

        # replace -100 with padding
        pred.label_ids[0][pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred.label_ids[1][pred.label_ids == -100] = processor.tokenizer.pad_token_id

        jyutping_pred_str = processor.batch_decode(jyutping_pred_ids)
        tone_pred_str = tone_tokenizer.batch_decode(tone_pred_ids)

        # we do not want to group tokens when computing the metrics
        jyutping_label_str = processor.batch_decode(
            pred.label_ids[0], group_tokens=False
        )
        tone_label_str = tone_tokens.batch_decode(pred.label_ids[1], group_tokens=False)

        jyutping_wer = wer_metric.compute(
            predictions=jyutping_pred_str, references=jyutping_label_str
        )
        tone_wer = wer_metric.compute(
            predictions=tone_pred_str, references=tone_label_str
        )

        return {"per": jyutping_wer, "ter": tone_wer}

    training_args = TrainingArguments(
        output_dir=output_dir,
        label_names=["jyutping_labels", "tone_labels"],
        group_by_length=True,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        num_train_epochs=30,
        bf16=True,
        gradient_checkpointing=True,
        overwrite_output_dir=False,  # set to False to continue training
        save_steps=1000,
        eval_steps=1000,
        logging_steps=100,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="per",
        greater_is_better=False,
        report_to="wandb",
        run_name="wav2vec2-yue" + time.strftime("%Y-%m-%d-%H-%M-%S"),
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        processing_class=processor,
    )

    # trainer.evaluate()
    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("model_id", type=str)
    args.add_argument("dataset", type=str)
    args.add_argument("--output_dir", type=str, default="checkpoints")
    args = args.parse_args()

    train(args.model_id, args.dataset, args.output_dir)
