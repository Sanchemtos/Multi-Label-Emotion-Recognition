# 🤖 Multi-Label Emotion Recognition with BERT

This project focuses on detecting multiple emotions from English text using a fine-tuned **BERT** model. It leverages the [GoEmotions](https://huggingface.co/datasets/go_emotions) dataset — a large-scale human-annotated dataset of Reddit comments labeled with 27 emotions + neutral.

---

## 📁 Dataset

- **Source**: [GoEmotions Dataset](https://huggingface.co/datasets/go_emotions)
- **Size**: 58k+ Reddit comments
- **Labels**: 27 fine-grained emotions + 1 neutral label
- **Type**: Multi-label (each text can have multiple emotions)

---

## ⚙️ Preprocessing

- **Label Encoding**: Multi-hot encoding using `MultiLabelBinarizer`
- **Tokenization**: `BertTokenizerFast` with `max_length=128`, truncation, and padding
- **Data Format**: Converted to Hugging Face `Dataset` and PyTorch tensors

---

## 🧠 Model

- **Base Model**: `bert-base-uncased` from Hugging Face Transformers
- **Architecture**: Custom classification head with sigmoid activation
- **Loss Function**: `BCEWithLogitsLoss` (suitable for multi-label classification)
- **Trainer**: Hugging Face `Trainer` API with custom `compute_metrics`

---


