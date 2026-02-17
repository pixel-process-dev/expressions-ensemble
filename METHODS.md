# Exaggerated Expressions: Methods

## Motivation & Goals

Exaggerated Expressions (ExE) demonstrates an end-to-end pipeline for collecting images with minimal manual curation for classification with ecological validity. Classification of emotional expressions often struggles to bridge the gap between controlled datasets and "in the wild" expressions. Reported metrics here inform how this approach performs across multiple validation scenarios.

### Multiple Validation
- **Ecological**: Facial expressions from well known movies are extracted and classified. Although accuracy on individual faces is not assessed here, clear patterns support that the approach generalizes well.
- **Standardized**: Evaluation against two established datasets of emotional facial expressions (FER 2013, RAF-DB) provides comparison with established metrics.
- **Performance**: Classification metrics, confusion matrices, and performance metrics demonstrate that this process yields stable signal with minimal curation.

## Keyword Strategy & Search Design

Two search patterns were tested across two stock photo APIs:

- **Face-specific:** `"[emotion] face"` (e.g., "happy face")
- **Multi-keyword:** Multiple adjectives per emotion (e.g., "happy", "smiling", "joyful")

| Emotion   | Keywords                              |
|-----------|---------------------------------------|
| Angry     | "angry", "mad", "irate"              |
| Fear      | "fear", "afraid", "scared"           |
| Happy     | "happy", "smiling", "joyful"         |
| Sad       | "sad", "crying", "unhappy"           |
| Surprise  | "surprised", "shocked", "astonished" |

**Dropped classes:** Disgust was removed due to low hit counts across both APIs. Neutral was removed due to ambiguity — future work may consider low-confidence predictions on trained emotions as an indicator of neutral-type expressions.


## Experiment Tracking & Model Selection

Nine training runs were tracked via MLflow, varying data source (Pixabay, Pexels, combined) and data version (v1, v2, v3 corresponding to increasing amounts of pulled data after deduplication).

### MLflow Experiment Summary

| Run Name | Source | Total Images | Val Accuracy | Best Val Acc | Recall: Happy | Sad | Angry | Fear | Surprise |
|---|---|---|---|---|---|---|---|---|---|
| pixabay-light-v1 | Pixabay | 1,553 | 47.9% | 53.7% | 47% | 68% | 48% | 22% | 22% |
| **pixabay-light-v2** | **Pixabay** | **2,132** | **70.7%** | **71.0%** | **81%** | **73%** | **74%** | **40%** | **45%** |
| pixabay-light-v3 | Pixabay | 3,136 | 60.4% | 60.4% | 87% | 57% | 29% | 18% | 33% |
| pexels-light-v1 | Pexels | 2,242 | 60.4% | 63.9% | 63% | 55% | 45% | 75% | 65% |
| pexels-light-v2 | Pexels | 4,517 | 52.3% | 54.8% | 74% | 58% | 38% | 38% | 45% |
| pexels-light-v3 | Pexels | 5,836 | 53.3% | 55.2% | 64% | 68% | 36% | 46% | 40% |
| pexpix-light-v1 | Combined | 3,795 | 61.9% | 62.2% | 70% | 72% | 52% | 50% | 66% |
| pexpix-light-v2 | Combined | 6,648 | 56.5% | 59.7% | 75% | 62% | 50% | 37% | 36% |
| pexpix-light-v3 | Combined | 8,970 | 52.8% | 56.2% | 69% | 62% | 38% | 37% | 43% |

### Why Pixabay V2?

The selected model (**pixabay-light-v2**, 71.0% best validation accuracy) was chosen based on a combination of factors:

1. **Highest overall accuracy** across all 9 runs, despite being trained on only 2,132 images — less than a quarter the size of the largest run (8,970 images).

2. **Balanced recall across classes.** Many runs achieved high recall on one or two emotions while collapsing on others (e.g., pixabay-v3 hits 87% happy recall but drops to 18% fear and 29% angry). Pixabay V2 maintains 40%+ recall on every class.

3. **More data ≠ better model.** Across all three sources, v2 outperforms v3 despite v3 having ~50% more data. This suggests diminishing returns or increased noise as keyword searches pull more peripheral results.

4. **Class imbalance did not determine selection.** All runs have imbalanced classes (happy is always the largest). The selection was driven by validation performance and recall balance, not by which run had the most even distribution.

### Class Distribution (Pixabay V2)

| Class | Count | % of Total |
|-------|-------|------------|
| Happy | 988 | 46.3% |
| Sad | 542 | 25.4% |
| Fear | 233 | 10.9% |
| Surprise | 203 | 9.5% |
| Angry | 166 | 7.8% |
| **Total** | **2,132** | |

Training split: 1,705 train / 427 validation (80/20).

## Design Decisions & Generalizability

### Why Face Detection Was Critical

This approach relied on MediaPipe's face detector for several reasons:

1. **Consistent framing:** Stock photos contain full scenes; face detection isolates the relevant emotional signal
2. **Domain alignment:** Training crops must match inference crops
3. **Noise reduction:** Filters out images with no faces, multiple faces, etc.

**Implication for generalization:** This approach is specific to facial emotion recognition. Extending to full-body emotions, activities, or object states would require different detectors and likely different keyword strategies.

### Why This Approach May Generalize

The core principle — multi-keyword weak supervision matched to validation domain — likely transfers to other classification problems where labeled data is scarce but keyword-searchable images are abundant. The validation methodology (temporal patterns in narrative content) could extend to any domain with interpretable temporal structure.
