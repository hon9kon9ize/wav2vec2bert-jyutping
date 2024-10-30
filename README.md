# Fine-Tune Wav2Vec2 for Jyutping Recognition

![Wav2Vec2Cantonese](Wav2Vec2Cantonese.png)

This repository contains the code for fine-tuning the [Wav2Vec Bert 2.0](https://huggingface.co/facebook/w2v-bert-2.0) model on the Common Voice 17 Cantonese dataset for Jyutping recognition. The model is trained on the [Common Voice 17 Cantonese dataset](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0).

## Requirements

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Training

To train the model, run the following command:

```bash
python train.py
```