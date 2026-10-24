# BART for Task-Oriented Dialogue Generation

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

**Dataset source:** [Add the original dataset URL or repository here.]

**License:** [Add the dataset license after verifying its terms.]

## Setup

Install the required packages:

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib evaluate sacrebleu rouge_score
```

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
