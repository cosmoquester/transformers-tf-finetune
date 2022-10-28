# Transformers tf finetune

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![cosmoquester](https://circleci.com/gh/cosmoquester/transformers-tf-finetune.svg?style=svg)](https://app.circleci.com/pipelines/github/cosmoquester/transformers-tf-finetune)
[![codecov](https://codecov.io/gh/cosmoquester/transformers-tf-finetune/branch/master/graph/badge.svg?token=GTsIlZy6oG)](https://codecov.io/gh/cosmoquester/transformers-tf-finetune)

- Scripts and notebooks to train hugginface transformers models with Tensorflow 2.
- You can train models with jupyter or python script using a separate machine, and even without it, you can learn with two clicks from the colab.
- Select Task below, enter `[Open in Colab]`, and click `[Runtime]` - `[Run all]` to automatically load, learn, and evaluate data.
- All code support GPU and TPU both.

<br/>

- Tensorflow 2를 이용해 Transformer 모델들을 파인튜닝합니다.
- 별도의 머신을 이용해 노트북이나 스크립트로 학습할 수 있으며 그게 없더라도 colab 에서 클릭 두 번으로 학습할 수 있습니다.
- 아래에서 Task를 골라 `[Open in Colab]`으로 들어간 뒤에 `[Runtime]` - `[Run all]` 을 클릭하면 데이터로딩과 학습, 평가까지 자동으로 수행됩니다.
- 모든 코드는 GPU, TPU 디바이스를 전부 지원합니다.

## Tasks

| TaskName | Supported Models | Script | Colab |
| --- | --- | --- | --- |
| [Chatbot](https://github.com/songys/Chatbot_data) | EncoderDecoder (e.g. BART, T5, ...) | [Link](https://github.com/cosmoquester/transformers-tf-finetune/blob/master/scripts/train_chatbot.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cosmoquester/transformers-tf-finetune/blob/master/notebooks/train_chatbot.ipynb) |
| [HateSpeech](https://github.com/kocohub/korean-hate-speech) | BART | [Link](https://github.com/cosmoquester/transformers-tf-finetune/blob/master/scripts/train_hate_speech.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cosmoquester/transformers-tf-finetune/blob/master/notebooks/train_hate_speech.ipynb) |
| [KLUE NLI](https://github.com/KLUE-benchmark/KLUE/tree/main/klue_benchmark/klue-nli-v1.1) | SequenceClassification (e.g. BERT, BART, GPT, ...) | [Link](https://github.com/cosmoquester/transformers-tf-finetune/blob/master/scripts/train_klue_nli.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cosmoquester/transformers-tf-finetune/blob/master/notebooks/train_klue_nli.ipynb) |
| [KLUE STS (Bi-Encoder)](https://github.com/KLUE-benchmark/KLUE/tree/main/klue_benchmark/klue-sts-v1.1) | SequenceClassification (e.g. BERT, BART, ...) | [Link](https://github.com/cosmoquester/transformers-tf-finetune/blob/master/scripts/train_klue_sts.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cosmoquester/transformers-tf-finetune/blob/master/notebooks/train_klue_sts.ipynb) |
| [KLUE TC](https://github.com/KLUE-benchmark/KLUE/tree/main/klue_benchmark/ynat-v1.1) | SequenceClassification (e.g. BERT, BART, GPT, ...) | [Link](https://github.com/cosmoquester/transformers-tf-finetune/blob/master/scripts/train_klue_tc.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cosmoquester/transformers-tf-finetune/blob/master/notebooks/train_klue_tc.ipynb) |
| [KorSTS (Bi-Encoder)](https://github.com/kakaobrain/KorNLUDatasets/tree/master/KorSTS) | SequenceClassification (e.g. BERT, BART, ...) | [Link](https://github.com/cosmoquester/transformers-tf-finetune/blob/master/scripts/train_korsts.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cosmoquester/transformers-tf-finetune/blob/master/notebooks/train_korsts.ipynb) |
| [NSMC](https://github.com/e9t/nsmc) | SequenceClassification (e.g. BERT, BART, GPT, ...) | [Link](https://github.com/cosmoquester/transformers-tf-finetune/blob/master/scripts/train_nsmc.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cosmoquester/transformers-tf-finetune/blob/master/notebooks/train_nsmc.ipynb) |
| [QuestionPair](https://github.com/aisolab/nlp_classification/tree/master/BERT_pairwise_text_classification/qpair) | SequenceClassification (e.g. BERT, BART, GPT, ...) | [Link](https://github.com/cosmoquester/transformers-tf-finetune/blob/master/scripts/train_question_pair.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cosmoquester/transformers-tf-finetune/blob/master/notebooks/train_question_pair.ipynb) |
