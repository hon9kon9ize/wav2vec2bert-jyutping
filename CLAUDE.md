# wav2vec2bert-jyutping Development Guide

## Project Goal

This repository fine-tunes speech self-supervised models for Cantonese Jyutping recognition. The published model is:

- Hugging Face: `hon9kon9ize/wav2vec2bert-jyutping`
- Base model: `facebook/w2v-bert-2.0`
- Dataset: Mozilla Common Voice 17 Cantonese, loaded from a prepared local Hugging Face dataset via `datasets.load_from_disk(...)`

The current main-branch implementation predicts Jyutping with two separate CTC heads:

1. `jyutping_head`: onset/nucleus/coda pieces without tone
2. `tone_head`: tone digits `1`-`6`

The next branch should train a **single CTC head using nucleus+tone tokens**, instead of a separate tone head.

## Current Repository Structure

```text
wav2vec2bert-jyutping/
├── data.py                  # CTC data collators; currently pads jyutping_labels + tone_labels
├── model.py                 # Wav2Vec2Bert/Wav2Vec2/Wav2Vec2Conformer Cantonese models
├── train_w2v2bert.py        # Main Wav2Vec2-BERT training script
├── train_w2v2.py            # Wav2Vec2 training variant
├── train_w2v2conformer.py   # Wav2Vec2-Conformer training variant
├── test.py                  # Evaluation script for existing two-head model
├── vocab.json               # Current non-tone Jyutping unit vocab
├── tone_vocab.json          # Current tone-only vocab
└── README.md
```

## Existing Two-Head Design

`vocab.json` currently contains:

- `[PAD]`: CTC blank / pad token, ID 0
- `[UNK]`: ID 1
- `|`: word delimiter, ID 2
- tone-less Jyutping units such as `b`, `gw`, `aa`, `aai`, `aak`, `ng`, `yu`, etc.

`tone_vocab.json` contains:

- `[PAD]`: ID 0
- `[UNK]`: ID 1
- `|`: ID 2
- tone digits `1`-`6`

`model.py` implements `Wav2Vec2BertForCantonese` as:

```python
self.jyutping_head = nn.Linear(output_hidden_size, config.vocab_size)
self.tone_head = nn.Linear(output_hidden_size, tone_vocab_size)
loss = jyutping_loss + tone_loss
```

Inference decodes both streams, rebuilds syllables from onset/nucleus/coda pieces, then zips tone digits onto decoded syllables. This can fail when the two CTC streams have different lengths or alignment errors.

## Target Nucleus+Tone Approach

Replace the two-head design with one CTC target stream.

Recommended tokenization:

- Keep onsets as tone-less tokens: `b`, `c`, `d`, `f`, `g`, `gw`, `h`, `j`, `k`, `kw`, `l`, `m`, `n`, `ng`, `p`, `s`, `t`, `w`, `z`
- Encode the rime/nucleus+coda together with tone: `aa1`, `aa2`, ..., `aak3`, `aak6`, `eoi5`, `ung6`, etc.
- Keep standalone syllabic nasals with tone as single tokens where needed: `m4`, `m5`, `ng4`, `ng5`, etc.
- Keep `[PAD]`, `[UNK]`, and `|` with the same special-token meaning.

Example:

```text
maa4 maa1 go3 jiu4 juk6 zeoi3
```

Token stream:

```text
m aa4 m aa1 g o3 j iu4 j uk6 z eoi3
```

This keeps CTC monotonic and avoids requiring a separate tone CTC alignment.

## Implementation Plan For The New Branch

### 1. Build a new vocab

Create a new vocabulary file, for example:

```text
vocab_nucleus_tone.json
```

It should contain:

- special tokens `[PAD]`, `[UNK]`, `|`
- onset tokens
- rime+nucleus+tone tokens
- syllabic nasal+tone tokens if they occur in the dataset

Do not use `tone_vocab.json` for the new model.

### 2. Add a Jyutping tokenizer helper

Add helper functions that convert full Jyutping syllables into the new token stream.

Suggested behavior:

```python
"gwong2" -> ["gw", "ong2"]
"aa3"    -> ["aa3"]
"m4"     -> ["m4"]
"ng5"    -> ["ng5"]
```

Use `pycantonese.jyutping.parse_jyutping.ONSETS` or a fixed longest-match onset list. Always match longer onsets first, e.g. `gw` before `g`, `kw` before `k`, and `ng` before `n`.

### 3. Simplify the data collator

In `data.py`, add a single-label collator for Wav2Vec2-BERT, or adapt `Wav2Vec2BertDataCollatorCTCWithPadding` so it only expects:

```python
feature["labels"]
```

and returns:

```python
batch["labels"]
```

For Hugging Face `Trainer`, prefer the conventional `labels` field unless there is a strong reason to keep `jyutping_labels`.

### 4. Add a single-head model class

Add a new class instead of breaking the existing published model class. Suggested name:

```python
Wav2Vec2BertForJyutpingCTC
```

It should:

- wrap `Wav2Vec2BertModel`
- keep one `lm_head`
- compute one CTC loss from `labels`
- return standard CTC-style output fields where practical
- expose a simple `inference(...)` method that decodes the single token stream directly

Avoid changing `Wav2Vec2BertForCantonese` unless you intentionally want to break compatibility with existing checkpoints.

### 5. Add a new training script

Add a separate script rather than rewriting the old one immediately:

```text
train_w2v2bert_nucleus_tone.py
```

Recommended changes from `train_w2v2bert.py`:

- use `vocab_nucleus_tone.json`
- remove `tone_tokenizer`
- in `prepare_dataset`, create `batch["labels"]` from the nucleus+tone token stream
- set `label_names=["labels"]`
- make `compute_metrics` decode one prediction stream
- report `per` or token/syllable error consistently

The old script currently uses:

```python
per_device_train_batch_size=256
per_device_eval_batch_size=64
learning_rate=1e-3
warmup_steps=200
num_train_epochs=10
bf16=True
gradient_checkpointing=True
```

Use these as the baseline unless there is a clear reason to change them.

## Validation Checklist

Before launching a full training run:

- [ ] Tokenizer encodes and decodes a few hand-picked examples correctly.
- [ ] The new vocab contains every generated target token from train and test.
- [ ] A single batch passes through the data collator.
- [ ] A single forward pass computes finite CTC loss.
- [ ] `Trainer` can run for a few debug steps.
- [ ] Evaluation reports the new single-stream error rate.

Suggested smoke examples:

```text
maa4 maa1 go3 jiu4 juk6 zeoi3
gwong2 dung1 waa2
ngo5 hai6 heoi3
m4 goi1
ng5
```

## Important Notes

- CTC blank is `processor.tokenizer.pad_token_id`; keep `[PAD]` at ID 0.
- Hugging Face CTC tokenizers use `|` as the word delimiter.
- The current code manually patches `AddedToken` objects so multi-character Jyutping tokens are not split. Keep that pattern for the new vocab.
- Do not compare old `per` and `ter` directly with the new single-stream `per`; the target units are different.
- For fair model comparison, decode both old and new models into full Jyutping syllables and compute syllable error rate or normalized token error rate on the same test split.

## Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Existing Wav2Vec2-BERT training command shape:

```bash
python train_w2v2bert.py facebook/w2v-bert-2.0 /path/to/dataset --output_dir checkpoints
```

Expected new branch command shape:

```bash
python train_w2v2bert_nucleus_tone.py facebook/w2v-bert-2.0 /path/to/dataset --output_dir checkpoints_nucleus_tone
```

## Branch Goal

The branch is successful if the single-head nucleus+tone model:

- matches or beats the published two-head model on full Jyutping syllable accuracy
- removes tone-stream length mismatch handling from inference
- produces simpler, more reliable decode output
- remains compatible with Hugging Face `Trainer` and `Wav2Vec2BertProcessor`
