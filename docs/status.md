---
layout: default
title: Status
---

## Project Summary

This project builds a multi-modal skin lesion classifier that combines dermoscopic smartphone images with structured patient metadata to classify skin lesions into six diagnostic categories: Basal Cell Carcinoma (BCC), Squamous Cell Carcinoma (SCC), Actinic Keratosis (ACK), Seborrheic Keratosis (SEK), Melanoma (MEL), and Melanocytic Nevus (NEV). The system is trained and evaluated on the PAD-UFES-20 dataset, a real-world clinical collection of smartphone-captured skin lesion images annotated with rich patient metadata including age, Fitzpatrick skin type, lesion symptoms, lifestyle factors, and family history. The core research question is whether fusing visual features with tabular patient metadata — via learned attention mechanisms — meaningfully improves classification performance over image-only baselines, particularly on an imbalanced, small-scale clinical dataset. Looking ahead, this classification backbone is intended to serve as the first stage of a broader health education pipeline that will incorporate Retrieval-Augmented Generation (RAG) to generate structured, citation-backed educational summaries for each predicted condition — though that integration is planned for a later phase after the classification component reaches a satisfactory level of performance.

## Approach

**Dataset.** We use the [PAD-UFES-20 dataset](https://www.kaggle.com/datasets/mahdavi1202/skin-cancer) (Pacheco et al., 2020), which contains 2,298 skin lesion samples collected from smartphone images at a Brazilian clinical study. Each sample consists of one or more PNG images and a row in a metadata CSV with 26 structured clinical features. The original dataset includes a seventh class (Bowen's Disease / BOD), which we merge into SCC following the dataset authors' recommendation, leaving six final diagnostic classes. The training set class distribution is highly imbalanced: ACK (527), BCC (572), SEK (161), NEV (165), SCC (128), and MEL (33) — a roughly 17:1 ratio between the most and least frequent class.

We perform a **patient-level split** into train (70%), validation (15%), and test (15%) using a fixed random seed of 42. This is critical: multiple images from the same patient are kept within the same split to prevent data leakage, since a model that has seen other lesions from the same patient at training time would have an unrealistically inflated test performance. All images are resized to 224×224 pixels. During training, we apply the following augmentation pipeline: random crop (from a 257×257 resize), random horizontal and vertical flip, color jitter (brightness, contrast, saturation ±0.3; hue ±0.05), random rotation (±20°), random affine translation and scale, Gaussian blur, and random erasing (p=0.2) to simulate hair or occlusion. Validation and test images receive only resizing and normalization using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

To handle class imbalance during training, we use a `WeightedRandomSampler` that samples each training example with probability proportional to the inverse frequency of its class label. We additionally apply **inverse-frequency class weighting** in the loss function and **label smoothing** of 0.1 to reduce overconfidence on the majority classes. The loss is therefore:

$$\mathcal{L} = -\sum_{c=1}^{6} w_c \left[ (1 - \epsilon) \cdot \mathbf{1}[y=c] + \frac{\epsilon}{6} \right] \log \hat{p}_c$$

where $w_c = \frac{1/n_c}{\sum_j 1/n_j}$ are the inverse-frequency class weights, $\epsilon = 0.1$ is the label smoothing coefficient, $y$ is the ground-truth label, and $\hat{p}_c$ is the predicted probability for class $c$.

**Metadata encoding.** The 26 clinical metadata features consist of 3 numerical columns (age, diameter_1, diameter_2) and 19 categorical columns (e.g., gender, Fitzpatrick skin type, smoking/drinking habits, symptom flags like itch, bleed, and elevation). Numerical features are z-score normalized using training set statistics. Categorical features are ordinally encoded. All imputation statistics (means and modes) are computed exclusively on the training set and applied to val/test to avoid leakage.

**Model 1: SpatialVisionFusion.** Our second model is a cross-attention fusion architecture combining ResNet-50 and a Vision Transformer (ViT-Base/16), both pretrained on ImageNet. The ResNet-50 backbone extracts spatial feature maps of shape (B, 2048, 7, 7), which are projected to a shared dimension of 512 via a 1×1 convolution and then flattened to 49 spatial tokens of shape (B, 49, 512). The ViT-Base/16 independently processes the same input image and its CLS token — a learned global representation — is extracted as a (B, 768)-dimensional vector and projected to (B, 512) via a linear layer. Cross-attention is then computed with the ViT CLS token as the query and the ResNet spatial tokens as keys and values:

**Model 2: MultiModalCNNFusion.** Our first model uses EfficientNet-B0 (pretrained on ImageNet) as a visual backbone, extracting a 1280-dimensional feature vector after global average pooling. Metadata is processed by a small MLP: Linear(22→64) → BatchNorm → ReLU → Linear(64→64) → ReLU → Dropout(0.3), producing a 64-dimensional metadata embedding. The visual and metadata embeddings are concatenated (1344-dim) and passed through a final classifier: Linear(1344→256) → ReLU → Dropout(0.5) → Linear(256→6).

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

where $d_k = 512$ and multi-head attention uses 8 heads. Patient metadata (26 features) is separately projected to 512 dimensions and used as a second query against the same ResNet spatial tokens via an additional cross-attention pass. The two attended vectors are concatenated (1024-dim) and passed through the same classifier head as above: Linear(1024→256) → ReLU → Dropout(0.5) → Linear(256→6).

**Training configuration.** Model 1 is trained for 20 epochs with batch size 32. We use the Adam optimizer with differential learning rates: the visual backbone is finetuned slowly at 1e-5, while the fusion head and metadata layers use 1e-3. The learning rate is annealed via a cosine schedule (`CosineAnnealingLR`, T_max=20, eta_min=1e-6). The best model checkpoint is selected by lowest validation loss, and final evaluation is run on the held-out test set using that checkpoint. Model 2 was recently implemented so we dont currently have any training on it, but will in the future.

## Evaluation

**Quantitative results.** We have completed full training and evaluation of the MultiModalCNNFusion model. Over 20 epochs, the model's validation accuracy improved steadily from 65.73% at epoch 1 to a best of **79.49%** at epoch 10 (validation loss: 0.6641), before plateauing in the high-70s for the remaining epochs. The model saved at best validation loss (epoch 7, loss=0.6538, accuracy=77.25%) achieved a **test accuracy of 63.76%** on the held-out test set. The gap between best validation accuracy (79.49%) and test accuracy (63.76%) indicates some degree of overfitting or distribution shift, which is not unusual for a small clinical dataset with only ~2,298 total samples. Training loss converged from ~1.8 in early batches down to the 0.03–0.5 range by epoch 15, confirming the model is learning effectively.

The table below summarizes training set class distribution, which motivates our use of class-weighted loss and weighted sampling:

| Class | Description               | Train Samples |
|-------|---------------------------|---------------|
| BCC   | Basal Cell Carcinoma      | 572           |
| ACK   | Actinic Keratosis         | 527           |
| NEV   | Melanocytic Nevus         | 165           |
| SEK   | Seborrheic Keratosis      | 161           |
| SCC   | Squamous Cell Carcinoma   | 128           |
| MEL   | Melanoma                  | 33            |

Per-class accuracy on the test set is tracked in TensorBoard but has not yet been extracted from the latest experimental run. This is a priority for the next evaluation cycle, as per-class breakdown is essential for understanding how well the model handles the rarest and most clinically dangerous class (MEL, n=33).

**TensorBoard logging.** All training and validation losses, per-epoch accuracy, per-class validation accuracy, and learning rate schedules are logged via TensorBoard under the `tensor_board/` directory. Experiments are auto-indexed (e.g., `MultiModalCNNFusion_experiment_1`, `SpatialVisionFusion_experiment_1`) so that results from different runs are kept separate. The SpatialVisionFusion model has been implemented and an initial experiment directory has been created, but full training and test evaluation of this model are in progress.

**Qualitative evaluation.** Qualitative analysis has not yet been conducted. Planned qualitative evaluation includes visual inspection of attention weight maps from the SpatialVisionFusion model (to understand which spatial regions of the image are most attended to for each class), as well as case study analysis of representative misclassifications — particularly between visually similar classes like MEL and NEV.

## Remaining Goals and Challenges

Our most immediate goal is to complete the training and evaluation of the SpatialVisionFusion model and compare its performance against MultiModalCNNFusion. We expect the cross-attention architecture to leverage the complementary strengths of ResNet-50 (local spatial features) and ViT (global context), particularly for lesion classes with distinctive global structure. We also want to extract and report per-class test accuracy for both models, since aggregate accuracy masks performance on rare but high-stakes classes like Melanoma. A further goal is to conduct hyperparameter tuning — particularly the shared projection dimension, number of attention heads, and dropout rates — and to document these experiments systematically. Finally, once classification performance is satisfactory, we plan to design and implement the RAG pipeline that will consume a predicted class and retrieve relevant medical literature chunks to generate a structured educational response, depending on remaining time.

In terms of challenges, the most significant concern is the severe class imbalance — particularly the MEL class with only 33 training samples. Even with weighted sampling and class-weighted loss, it is difficult to learn a reliable MEL classifier from so few examples; we may need to investigate stronger data augmentation strategies or few-shot learning approaches for this class. A second challenge is the validation-to-test accuracy gap (~16 percentage points for MultiModalCNNFusion), which suggests the model may be overfitting to validation-set characteristics; we will investigate whether this is due to sampling variance in the small dataset or whether regularization needs to be strengthened. The SpatialVisionFusion model is also considerably heavier than EfficientNet-B0, and training time may be a limiting factor given our hardware constraints. Finally, integrating the RAG pipeline — including setting up a vector database, embedding medical literature, and designing the retrieval-augmented generation prompt — will require significant additional engineering effort, and it is not yet certain that there will be enough time to complete this component to a high standard before the final report deadline.

## Resources Used

**Dataset.** PAD-UFES-20 skin lesion dataset, accessed via [Kaggle](https://www.kaggle.com/datasets/mahdavi1202/skin-cancer) using the `kagglehub` library. Original paper: Pacheco, A. G. C., et al. (2020). "PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones." *Data in Brief*, 32.

**Deep learning framework.** PyTorch (v2.10.0) and torchvision (v0.25.0) for model definition, training, and data loading. The `timm` library (v1.0.24) was used to load the pretrained ViT-Base/16 model with `timm.create_model('vit_base_patch16_224', pretrained=True)`. Default hyperparameter choices (batch size, Adam learning rate, cosine annealing schedule) were informed by standard transfer learning practice documented in the PyTorch and timm documentation.

**Experiment tracking.** TensorBoard (v2.20.0) for logging training/validation loss, accuracy, per-class accuracy, and learning rate curves across experiments.

**Supporting libraries.** scikit-learn for patient-level train/val/test splitting (`train_test_split`), label encoding (`LabelEncoder`), and class-balanced sampling support. pandas and numpy for metadata loading, preprocessing, and imputation.

**AI tools.** Claude (via Claude Code / Cowork) was used for assistance with writing and structuring portions of this status report, as well as for code review and explanation of implementation details during development. For the data loader we heavily relied upon AI to write it as we didnt want to waste our times with getting our data loader working with the meta data and wanted to focus on actually building the model. AI was also used in reviewing and giving insight on how we could move along with our project given our current results and such.

<iframe width="560" height="315" src="https://www.youtube.com/embed/tbwp25ODoh0?si=Hxs-NovS_qAsg3dF" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>