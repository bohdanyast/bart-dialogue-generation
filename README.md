# BART for Task-Oriented Dialogue Generation

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/bodyastar/bart-dialogue-generation)

## Overview

This project fine-tunes the pretrained `facebook/bart-base` model to generate responses in task-oriented service dialogues. The dataset contains conversations related to finding service providers and scheduling appointments.

The project demonstrates a complete sequence-to-sequence NLP pipeline, including data preprocessing, context construction, model fine-tuning, quantitative evaluation, and qualitative error analysis.

## Methodology

- Construct input-response pairs from dialogue histories.
- Split data into training, validation, and test sets at the dialogue level.
- Analyze input and target sequence lengths before truncation.
- Fine-tune BART using PyTorch and Hugging Face Transformers.
- Freeze encoder parameters and optimize the remaining trainable parameters.
- Monitor training and validation loss with early stopping.
- Evaluate generated responses using BLEU and ROUGE-L.
- Inspect incorrect or potentially unsupported responses.

## Dataset

The dataset is loaded from a JSON file containing dialogue turns. Each training example consists of a conversation context and the corresponding system response.

**Dataset source:** [Schema-Guided Dialogue (SGD)](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) by Google Research. This project uses a subset of 125 dialogues from the `Services_1` (salons) and `Services_2` (dentists) services.

> Rastogi, A., Zang, X., Sunkara, S., Gupta, R., Khaitan, P. *Towards Scalable Multi-Domain Conversational Agents: The Schema-Guided Dialogue Dataset.* arXiv:1909.05855.

**License:** The dataset is released under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The dataset file is not redistributed in this repository; please download it from the source above.

## Setup

**Option 1: Run on Kaggle (recommended).** Open the [Kaggle notebook](https://www.kaggle.com/code/bodyastar/bart-dialogue-generation) and enable a GPU accelerator in the notebook settings. The dataset is already attached, so no path changes are needed.

**Option 2: Run locally.**

1. Download the dialogue data from the source listed in the Dataset section.
2. Install the required packages:

   ```bash
   pip install torch transformers scikit-learn pandas numpy matplotlib evaluate sacrebleu rouge_score
   ```

3. In the notebook, update `DATA_PATH`, `CHECKPOINT_DIR` and `OUTPUT_DIR`, which point to Kaggle directories by default.
Update the dataset path (`DATA_PATH`) in the notebook before running the experiment.

## Running the experiment

1. Load and preprocess the JSON dataset.
2. Construct dialogue context-response pairs.
3. Split dialogues into training, validation, and test sets.
4. Fine-tune BART and monitor validation loss.
5. Evaluate the best checkpoint on the test set.
6. Review generated predictions and error analysis.
7. Save the model, configuration, metrics, and training history.

## Evaluation

The model is evaluated using BLEU, ROUGE-L, and test loss. Quantitative metrics are complemented by manual inspection of generated responses.

### Data

The dataset yields 1,123 context-response examples from 125 dialogues. Splitting by dialogue gives 908 training, 102 validation and 113 test examples.

### Results

| Metric | Value |
|---|---|
| BLEU (SacreBLEU, test) | 12.84 |
| ROUGE-L (test) | 0.3196 |
| Test loss | 1.9633 |
| Best validation loss (epoch 8) | 2.2478 |
| Exact match with reference (test) | 1.77% (2 of 113) |

Training ran for all 10 epochs (early stopping was not triggered). Training loss decreased steadily from 3.24 to 1.25, while validation loss plateaued around 2.25 after epoch 5, which indicates a growing generalization gap.

### Key findings

- The model learns the surface structure of system responses and copies slot values such as the city name from the user's request.
- It frequently over-produces the "Please confirm the following details..." template in places where the reference asks for a missing slot (date, time) or reports a completed booking.
- It sometimes ignores information that is already in the context (for example, asking for the city after the user named it).
- It can hallucinate provider names, addresses and phone numbers that do not appear in the input.

## Limitations

- The dataset is relatively small.
- Free-form generation may produce unsupported business names, addresses, or phone numbers.
- BLEU and ROUGE-L do not fully measure factual correctness or conversational appropriateness.
- Freezing the encoder may limit adaptation to the dialogue domain.
- The model does not independently verify provider information or appointment availability.

## Reproducibility

The notebook fixes random seeds and saves the best model checkpoint, tokenizer, experiment configuration, training history, and test predictions.

Exact reproducibility may still depend on library versions, hardware, and GPU execution behavior.

## Future improvements

- Compare frozen-encoder fine-tuning with full-model fine-tuning.
- Evaluate different context-window sizes.
- Compare BART with a baseline model.
- Add factual consistency and task-completion evaluation.
- Integrate a verified provider database for real-world use.

## License

The code in this repository is released under the MIT License (see `LICENSE`). The dataset is not part of this repository and remains under CC BY-SA 4.0 (see the Dataset section).
