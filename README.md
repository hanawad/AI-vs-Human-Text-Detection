# 🧠 Human vs. AI Text Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/TensorFlow-RNN%20%7C%20LSTM-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/PyTorch-BERT-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Environment-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white" alt="Google Colab">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License">
</p>

A binary text-classification project comparing traditional machine-learning models, RNN/LSTM networks, and fine-tuned BERT to distinguish human-written text from AI-generated text.

> Predictions are probabilistic signals, not proof of authorship. Do not use this model alone for high-stakes decisions.

## 🎯 Problem

- `0`: Human-written
- `1`: AI-generated

The project explores text cleaning, feature representation, model comparison, and error analysis using classification metrics and visualizations.

## ⚙️ Workflow

1. Load and combine CSV datasets; inspect missing values, duplicates, class balance, and text characteristics.
2. Clean text and filter samples by length.
3. Train `TF-IDF` pipelines with `Naive Bayes`, `KNN`, `Decision Tree`, `Logistic Regression`, and `SVM`.
4. Train `SimpleRNN` and `LSTM` with tokenization and padding; fine-tune `bert-base-uncased` in a separate notebook.
5. Evaluate with `Accuracy`, `F1-Score`, `ROC-AUC`, `Confusion Matrix`, and loss where available.

## 🧰 Tech Stack

| Area | Tools |
|---|---|
| Data & NLP | `pandas`, `NumPy`, `NLTK`, `re` |
| Classical ML | `scikit-learn`, `TF-IDF` |
| Deep learning | `TensorFlow` / `Keras`, `PyTorch`, `Hugging Face Transformers` |
| Visualization | `Matplotlib`, `Seaborn`, `WordCloud` |
| Runtime | Python, Jupyter Notebook, Google Colab |

## 📊 Results

The values below are from saved notebook outputs and may vary with the data split, runtime, and hardware. Classical-model reports used 2,500 test samples.

| Model | Accuracy | Macro F1 | ROC-AUC | Training time |
|---|---:|---:|---:|---:|
| Naive Bayes | 0.82 | 0.82 | 0.9244 | Not recorded |
| KNN | 0.74 | 0.73 | 0.8494 | Not recorded |
| Decision Tree | 0.81 | 0.81 | 0.8582 | Not recorded |
| Logistic Regression | 0.88 | 0.88 | 0.9612 | Not recorded |
| **SVM** | **0.91** | **0.91** | **0.9731** | **144.55 s** |
| SimpleRNN | 0.9040 | Not reported | Not reported | Not recorded |
| LSTM | Not available | Not available | Not available | Not recorded |
| BERT | Not available | Not available | Not available | Training output is incomplete |

`SVM` achieved the strongest recorded classical-model results. The `SimpleRNN` test loss was `0.4589`; its confusion matrix is shown below.

| Actual \ Predicted | Human | AI |
|---|---:|---:|
| Human | 891 | 89 |
| AI | 103 | 917 |

## 🖼️ Visualizations

Add exported notebook figures to `assets/` using these filenames:

![ROC curve comparison](assets/roc-curve-comparison.png)
![SVM confusion matrix](assets/svm-confusion-matrix.png)
![Class distribution](assets/class-distribution.png)
![Text length distribution](assets/text-length-distribution.png)
![SimpleRNN training curves](assets/rnn-training-curves.png)

## 🚀 Getting Started

### Requirements

- Python with Jupyter Notebook, or Google Colab
- The project CSV files
- Additional runtime and memory for deep-learning models; BERT downloads a large pretrained checkpoint

### Install

Run in a notebook cell:

```python
%pip install pandas numpy matplotlib seaborn wordcloud nltk scikit-learn
%pip install tensorflow transformers torch
```

### Run

1. Open either notebook and install the required packages.
2. Update the CSV paths to match your environment.
3. Confirm the expected columns: `text` and `label` for the main notebook; `processed_text` and `label` for BERT.
4. Run cells from top to bottom. Download the NLTK `stopwords` resource if needed.

The notebooks currently contain machine-specific Windows and Google Drive paths. Some install cells also use bare `pip install`; use `%pip install` or `!pip install` in notebooks. Saved outputs do not guarantee a fresh run will complete successfully.

## 🔎 Notes & Next Steps

- The RNN/LSTM `Tokenizer` is fitted before the train/test split, exposing test-set vocabulary. Split first and fit the tokenizer on training data only.
- Use stratified splits, deduplicate before splitting, and report the same metrics for every model.
- Complete BERT evaluation and record hardware, runtime, and model configuration.
- Test generalization on datasets from different sources and time periods.

## 📚 Data Sources

- [Kaggle: AI vs Human Text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text)
- [GitHub: human-v-ai](https://github.com/rsickle1/human-v-ai)

Check each dataset's terms before redistribution or commercial use.

Datasets are not included in this repository. Download them from their original sources and follow their respective terms; the MIT license below applies to the project code only.

## 📄 License

The source code is released under the [MIT License](LICENSE). Dataset and pretrained-model terms remain separate.

## 📁 Project Structure

```text
.
├── AI_Human_Detection_Haneen_Awad.ipynb
├── BERT_AI_Human_Detection_.ipynb
├── README.md
├── LICENSE
```
