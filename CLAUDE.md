# wav2vec2bert-jyutping Development Guide

## Project Goal and Branches

Fine-tune `facebook/w2v-bert-2.0` for Cantonese Jyutping recognition (audio -> toned Jyutping syllables).

| Branch | What it is |
|---|---|
| `main` | Published model `hon9kon9ize/wav2vec2bert-jyutping`: two CTC heads (`jyutping_head` = onset/final pieces without tone, `tone_head` = tone digits), trained on Common Voice 17 Cantonese. |
| `neclueus-tone` (typo in the name) | **Latest Cantonese-only work**: one CTC head over nucleus+tone tokens, ONNX serving, error analysis, head-design comparison harness. |
| `multilingual` | Experiment: one head over yue/zh/en phoneme tokens. Not working yet (see Known Issues). |

## Repository Structure

```text
jyutping.py                    # Jyutping <-> token helpers (label tokens, decoding to syllables)
ctc_utils.py                   # load_ctc_tokenizer, ctc_ids_to_token_text, HeadLRMixin, LengthAwareTrainer
data.py                        # collators (two-head and single-head)
model.py                       # Wav2Vec2BertForCantonese (two-head, published), Wav2Vec2BertForJyutpingCTC (single head), Wav2Vec2/Conformer variants
model_aux.py                   # single head + auxiliary tone-CTC head (training signal only)
dataset_utils.py               # dataset loading + the fixed train/validation/test split
train_w2v2bert_nucleus_tone.py # single-head training (the Cantonese recipe)
compare_heads.py               # light head-design comparison: single | two_head | single_aux
eval_w2v2bert_nucleus_tone.py  # test-split eval of single-head checkpoints
eval_original_w2v2bert.py      # test-split eval of the published two-head model
analyze_jyutping_errors.py     # error analysis
export_onnx.py, serve.py       # ONNX export + CPU FastAPI serving (Dockerfile.serve-onnx, docker-compose.*.yml)
train_w2v2bert.py, train_w2v2.py, train_w2v2conformer.py, test.py   # original two-head scripts (main branch)
build_nucleus_tone_vocab.py    # builds vocab_nucleus_tone.json from the dataset
vocab_nucleus_tone.json        # single-head vocab (281 entries)
vocab.json, tone_vocab.json    # published two-head vocabs
```

## Dataset and Annotation

- Default dataset: `indiejoseph/tts20250516` (a TTS-training corpus; 261,524 clips, 16 kHz). Columns used: `audio`, `phone` (inline toned Jyutping, e.g. `zung3 so2 zau1 zi1 .`). Others (`speech_token`, `spk_emb`, `phone_token_ids`, ...) are ignored.
- Split: `dataset_utils.make_train_validation_test_split` shuffles with seed 42, then 500 test, 500 validation, the rest train. Never change it: results are only comparable on this split.
- Clips are short: mean 2.2 s, 95th percentile 4.0 s, max 14.5 s; only 0.28% are longer than 7 s.
- Labels look like automatic dictionary conversion (canonical readings), not hand annotation. Known consequences: colloquial mergers (`n/l`, `ng/∅`, `gw/g`, tones `3/6`, `4/6`, `2/5`) are scored as errors; a few conversion mistakes exist (e.g. 擦 `caat4`, 賊 `caak2`); 385 tokens of stop-final syllables carry a tone outside 1/3/6 (mostly tone 2, e.g. `dip2`, `guk2`; most likely genuine changed tones, not errors).
- Syllables without an onset are 0.49% of data (`aa3`, `on1`), syllabic nasals (`m4`, `ng5`) 0.94%. The dataset's own phone arrays have no placeholder for a missing slot, and we do not add one (see Decisions).
- All 261,524 rows parse and every label token is in `vocab_nucleus_tone.json`.

## Label Format (single head) - important

One token stream: onset tokens (tone-less) and rime+tone tokens, with `|` **only between syllables**:

```text
zung3 so2 zau1 zi1   ->   z ung3 | s o2 | z au1 | z i1
```

- Vocab (281): `[PAD]`=0 (CTC blank, never in a label), `[UNK]`=1, `|`=2, 19 onsets (`b c d f g gw h j k kw l m n ng p s t w z`), 257 rime+tone tokens (`aa4`, `ung3`, ...), plus 2 tokenizer extras. Syllables with no onset (`aa3`) and syllabic nasals (`m4`, `ng5`) are one token.
- Build labels with `jyutping.annotation_to_nucleus_tone_label_tokens(...)` and map with `tokenizer.convert_tokens_to_ids(tokens)`.
- **Do not** build labels with `processor(text="m aa4 m aa1")`: Wav2Vec2CTCTokenizer turns every space into `|`, so a `|` lands between each onset and its rime (the old checkpoints were trained that way; targets are ~35% longer).
- Decoding: `ctc_utils.ctc_ids_to_token_text(tokenizer, ids, length)` collapses repeats, drops blanks and `|`, and gives `z ung3 s o2 ...`; `jyutping.nucleus_tone_text_to_jyutping` pairs onsets with rimes. Both label styles decode to the same text, so old and new checkpoints are scored identically.
- Always pass the real CTC output `length` when decoding a padded batch; the model emits ids for padding frames. `LengthAwareTrainer` returns `(logits, output_lengths)` for this reason.
- `load_ctc_tokenizer` keeps multi-character tokens whole (the `AddedToken` patch). Reuse it; the training/eval scripts still carry their own copy.
- CTC blank is `processor.tokenizer.pad_token_id`; keep `[PAD]` at ID 0. `len(tokenizer)` is vocab entries + 2, which is the head size.

## Training Recipe

**One learning rate for everything does not work at small batch sizes**: with these labels the loss sits at ~3 (all-blank plateau, SER ~100%). Give the output head its own, higher rate:

```bash
python train_w2v2bert_nucleus_tone.py facebook/w2v-bert-2.0 --output_dir checkpoints_nucleus_tone \
  --learning_rate 5e-5 --head_learning_rate 1e-3 --per_device_train_batch_size 16 --warmup_steps 100 --precision bf16
```

- Validated only at batch 16 (300 steps -> 18.7% test SER; 1,000 steps on 6,000 clips -> 4.8%). The original defaults (batch 256, lr 1e-3, uniform) are untested with these labels: watch the first 300 steps (loss should fall below ~1.5; a flat ~3.0 means the plateau).
- Fixed in the model setup: adapter on (`add_adapter=True`), `ctc_zero_infinity=True`, dropout 0.2, `mask_time_prob=0.0` (**no SpecAugment**; untested change worth trying).
- The script's other defaults: `num_train_epochs=10` hard-coded, precision fp16 by default (bf16 on Ampere/H100), `--eval_steps` default 1000. The best full-data run (2026-05-09, genkai) used 6 epochs and eval every 508 steps.
- Checkpoint choice: the validation SER logged by older runs was inflated by decoding padding frames, so "best checkpoint" picks from those runs are unreliable. `checkpoint-3048` (final step) is as good as the old "best" (test SER 1.75% vs 1.84%).

## Evaluation

- Always report syllable error rate (SER) on full toned Jyutping, on the fixed 500-clip test split, with the length-aware decode:
  `python eval_w2v2bert_nucleus_tone.py --model_id <ckpt> --split test --batch_size 1`
  (published model: `python eval_original_w2v2bert.py --model_id <dir> --split test --batch_size 1`).
- Do not compare old `per`/`ter` with single-stream `per`; the units differ. Compare SER.
- The test split is in-domain TTS audio. It says little about real speech. `tmp/labels.md` lists 6 hand-labelled wavs (78 syllables) for a quick real-audio check; it is far too small for decisions.

## Results So Far (single seed unless noted)

| Model | Test SER (500 clips) | 6 real wavs (78 syll.) |
|---|---|---|
| Ours, full data (`checkpoint-3048`, run 2026-05-09) | 1.75% (PER 0.89%) | 23 errors (29.5%) |
| Published two-head (Common Voice 17) | 5.24% | 16 errors (20.5%) |

- The two tie (6/45 each) on the 5 short wavs; the published model's lead is one 7 s clip that our training data barely covers.
- In-domain errors: tone ~2/3 of substituted syllables, then onset (mostly same-place pairs like `l/n`, `z/s`), then rime; 96% of substitutions are wrong in one part only. On real wavs tone errors are farther from the truth.
- Light head comparison (`compare_heads.py`, 6,000 clips, 1,000 steps, identical recipe): single 4.84%, two-head 4.56%, single + auxiliary tone loss 4.75% test SER. No significant SER differences; two-head gets more clips exactly right (69.8% vs 64.8%, p~0.015); two-head tone/syllable length mismatch 0.6%.
- CTC beam search (64) + syllable trigram LM trained on the training transcripts: the gold sentence is in the n-best for 94% of in-domain error clips; rescoring removes ~30% of errors in-domain (optimistic: weight tuned on the test split). On the real wavs the gold is in the n-best for 3 of 6 clips, so gains are small. The LM pushes toward canonical readings: keep it optional where actual pronunciation matters.

## Decisions (do not revisit without new evidence)

- Keep the single head. No measurable SER cost vs two heads, and no tone-stream sync failure.
- No auxiliary tone head (no net gain: fewer tone errors, more rime errors), no three-head design.
- No placeholder token for syllables without an initial or final: rare (1.4%), already unambiguous via `|`, models get syllabic nasals 40/40 right.
- Use `checkpoint-3048` for serving/ONNX (`serve.py` and `export_onnx.py` default to it).

## Known Issues / TODO

- Real-speech gap (ours ~30% on 6 wavs vs 1.75% in-domain): needs a larger labelled real-speech eval set, real-speech training data (Common Voice, other Cantonese corpora), augmentation (SpecAugment, speed/noise, long-form clips), and possibly initialising the encoder from the published model. Test ideas with `compare_heads.py` (~10 min per run on a 3090).
- `serve.py` returns raw token pieces, not syllables; no chunking for long audio; latency not measured.
- Multilingual branch: single head on yue/zh/en tokens collapsed at a uniform learning rate (plausibly the plateau above: try `--head_learning_rate`). `eval_multilingual.py` splits glued decoded tokens wrongly and hard-codes `add_adapter=False`; the language-mask buffer is not saved/loaded; run19 was never submitted.
- F0 option: `from_pretrained(..., use_f0=True)` on a saved F0 checkpoint silently drops the F0 projection (the config consumes the kwarg).
- The genkai repo (`/home/pj24001684/ku40000295/jc/projects/wav2vec2bert-jyutping`) is behind and dirty; its git index holds ~13 GB of staged checkpoint blobs.

## Commands

```bash
pip install -r requirements.txt                      # CPU env; training needs a CUDA build of torch
python train_w2v2bert.py facebook/w2v-bert-2.0 /path/to/dataset --output_dir checkpoints   # original two-head (main)
python compare_heads.py single --output_dir checkpoints/headcmp/run --max_steps 1000 --eval_steps 250 \
  --warmup_steps 100 --learning_rate 5e-5 --head_lr_mult 20          # also: two_head, single_aux
```

On the dev machine the CPU env is `.venv`; a CUDA env with matching package versions is at `~/.venvs/w2v-bert-cuda` (RTX 3090). Datasets and the base model are in the Hugging Face cache; use `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` to avoid network calls.

## Gotchas

- `.gitignore` ignores `*.json` and `*.sh`; vocab files need `git add -f`. `checkpoints/`, `wandb/`, `onnx/`, `tmp/` are ignored but `checkpoints_*` directories are not.
- Never `pkill -f <script>` from a command whose own text contains that script name: it kills your own shell.
