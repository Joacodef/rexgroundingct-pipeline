# Official Rules: ReXGroundingCT Challenge @ MICCAI 2026

> Structured transcript of the official rules published at `https://rexrank.ai/ReXGroundingCT/challenge.html` and the submission guidelines at `https://rexrank.ai/explore/submission_guideline_ct.html`. Last verified: April 17, 2026. In case of any discrepancy with the official page, the official page takes precedence.

---

## 1. About the Challenge

The ReXGrounding Challenge is an official MICCAI 2026 challenge designed to evaluate models on the localization of radiological findings described in unrestricted natural language, producing precise 3D segmentation masks in volumetric thoracic CTs.

Unlike previous challenges focused on lesion or organ segmentation by category, this benchmark requires models to interpret diverse clinical language — including anatomical descriptors, spatial relationships, and morphological attributes — and accurately anchor them in volumetric space. The dataset includes focal and diffuse abnormalities, covers a wide range of radiological patterns, and reflects real-world variability in how radiologists report.

The challenge is built on CT-RATE (a large-scale dataset of non-contrast thoracic CTs paired with free-text radiology reports), extended with expert-verified pixel-level 3D segmentations corresponding to individual findings from the reports.

Host: public leaderboard at `https://rexrank.ai/ReXGroundingCT/index.html`.

---

## 2. Task

**Single task: free-text finding grounding.**

Model input:
- A thoracic CT volume.
- A finding in natural language extracted from a radiology report.

Expected output:
- A 3D segmentation mask corresponding to the description.

### The 14 Categories

**Typically non-focal (6):**
1. Bronchial wall thickening
2. Bronchiectasis
3. Emphysema
4. Septal thickening
5. Micronodules
6. Other non-focal

**Typically focal (8):**
1. Linear opacities
2. Atelectasis / consolidation
3. Ground-glass opacity
4. Pulmonary nodules / masses
5. Pleural effusion / thickening
6. Honeycombing
7. Pneumothorax
8. Other focal

---

## 3. Dataset

| Split | Cases | Annotation type |
|---|---|---|
| Training | 2,992 CT scans | Partial (up to 3 instances per finding) |
| Validation | 200 CT scans | Exhaustive (all visible instances) |
| Test | 300 CT scans | Exhaustive (all visible instances) |

All annotations are pixel-level 3D masks linked to free-text findings extracted from radiology reports. The validation and test sets were annotated exclusively by board-certified radiologists.

### Note on the Split
The official paper/HuggingFace split (2,992 train + 50 val + 100 test) differs from the challenge split. For the MICCAI challenge, val is expanded to 200 and test to 300.

---

## 4. Timeline

| Date | Milestone |
|---|---|
| Now — May 2026 | Pre-registration open. Training data already publicly available. |
| June 2026 | Challenge launch: registration opens, validation set (200) released. |
| June — September 2026 | Development phase: evaluate on val set, submit multiple runs. |
| September 2026 | Submission deadline. Final evaluation on held-out test set. |
| Late September 2026 | Results announced and challenge session at MICCAI 2026. |

---

## 5. Evaluation Metrics

### Ranking Metric
**Average Dice Coefficient (DSC), computed per finding and per case.**

### Overlap-Based Metrics

| Metric | Definition |
|---|---|
| Dice (primary) | Average DSC per finding per case |
| Hit Rate | Proportion of findings where the global Dice ≥ 0.1 |
| Instance Precision | TP / (TP + FP), where a TP is a predicted instance with Dice ≥ 0.2 |
| Instance Recall | TP / (TP + FN) with the same criterion |
| Instance F1 | Harmonic mean of Instance Precision and Instance Recall |

### Distance-Based Metrics

| Metric | Definition |
|---|---|
| Distance Precision | TP / (TP + FP), where a TP is a predicted instance with ASSD (non-focal) or centroid distance (focal) ≤ 2× max voxel spacing |
| Distance Recall | TP / (TP + FN) with the same distance-based matching criterion |
| Distance F1 | Harmonic mean of Distance Precision and Distance Recall |

### Important Note on Hit Rate Threshold
The official challenge Hit Rate uses **Dice ≥ 0.1** as the threshold. The VoxTell paper reports HIT₅% (threshold 0.05), which is different. Make sure local evaluation uses 0.1.

---

## 6. Participation Rules

1. **Pre-registration** via the official form to receive updates when the challenge formally opens in June.
2. **Training data already available.** Method development can begin right away.
3. **External data permitted.** Any public or private training data, including pre-trained models and external datasets, may be used. All external data sources must be described in the submission.
4. **Fully automatic predictions.** All predictions on the test set must be generated automatically. The following are prohibited:
   - Manual intervention.
   - Post-hoc editing.
   - Case-specific tuning.
5. **During the development phase**, teams may evaluate on the val set and submit multiple runs. Only the final submission before the deadline is officially ranked.

---

## 7. Awards and Publication

- The top 3 teams receive:
  - Certificates.
  - Invited oral / spotlight presentation at the challenge session.
  - Recognition on the public leaderboard.
  - Recognition in the post-challenge paper.
- **Co-authorship in the challenge publication**: members of top-3 teams qualify, up to 8 authors per team.
- **No embargo period**: all teams may publish their results independently without any time restriction.

---

## 8. Technical Submission Process

### Prediction Format

1. Run inference on the ReXGroundingCT test set.
2. Save predictions as individual files with the **same names as the original CT files**.
3. Each prediction file must have shape `(F, H, W, D)`, where:
   - `F` = number of findings for that scan.
   - `H, W, D` = spatial dimensions that must match the ground truth.
4. Compress all prediction files into a single archive (`.zip` or `.gz`).

### Submission

Submission is by **email**, not through a web platform.

Email:
- **Recipient**: Mohammed Baharoon — `mohammed_baharoon@hms.harvard.edu`
- **Subject**: `ReXrankCT Submission: [Model Name]`
- **Attachment**: compressed file with predictions.
- **Email body must include**:
  - Model name.
  - Brief model description.
  - Link to paper or code repository (if available).
  - Institution name.

### Leaderboard Removal

To withdraw a model from the leaderboard, email Mohammed Baharoon with subject `ReXrankCT Leaderboard Removal Request` and the model name in the body.

### Operational Implications

- The expected output format is **identical to the original 4D mask format** (F, H, W, D). This means that if your pipeline works internally with separate 3D NIfTIs (one per finding), a final **stacking** step is needed to return to the 4D format before submitting.
- There is no continuous automated evaluation platform: turnaround depends on when the organizer evaluates your submission. This implies that "quick leaderboard test" experiments have latency, and solid local validation should come first.
- Having the submission pipeline ready and tested before June is key: when the 200-case val set is released, you want to be able to submit quickly.

---

## 9. Organizers

- Mohammed Baharoon — Harvard Medical School, USA (primary contact)
- Pranav Rajpurkar — Harvard Medical School, USA
- Luyang Luo — Harvard Medical School, USA
- Xiaoman Zhang — Harvard Medical School, USA
- Mahmoud Hussain Alabbad — King Fahad Hospital, Saudi Arabia
- Sungeun Kim — Harvard Medical School, USA

General challenge contact: `MohammedSalimAB@outlook.com`

---

## 10. Official Resources

- Leaderboard: `https://rexrank.ai/ReXGroundingCT/index.html`
- Challenge page: `https://rexrank.ai/ReXGroundingCT/challenge.html`
- Submission guidelines: `https://rexrank.ai/explore/submission_guideline_ct.html`
- Dataset on HuggingFace: `https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT`
- Evaluation code: `https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT/blob/main/rexrank_eval.py`
- ReXGroundingCT paper: `https://arxiv.org/abs/2507.22030`

---

## 11. Official Citations

**ReXGroundingCT:**
```
@article{baharoon2025rexgroundingct,
  title={ReXGroundingCT: A 3D Chest CT Dataset for Segmentation of Findings from Free-Text Reports},
  author={Baharoon, Mohammed and Luo, Luyang and Moritz, Michael and Kumar, Abhinav and Kim, Sung Eun and Zhang, Xiaoman and Zhu, Miao and Alabbad, Mahmoud Hussain and Alhazmi, Maha Sbayel and Mistry, Neel P and others},
  journal={arXiv preprint arXiv:2507.22030},
  year={2025}
}
```

**CT-RATE:**
```
@article{hamamci2026generalist,
  title={Generalist foundation models from a multimodal dataset for 3D computed tomography},
  author={Hamamci, Ibrahim Ethem and Er, Sezgin and Wang, Chenyu and Almas, Furkan and Simsek, Ayse Gulnihan and Esirgun, Sevval Nil and Dogan, Irem and Durugol, Omer Faruk and Hou, Benjamin and Shit, Suprosanna and others},
  journal={Nature Biomedical Engineering},
  pages={1--19},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
```

---

## 12. Critical Observations Extracted from the Rules

These are not official rules but operational implications derived from reading them together. Useful for giving actionable responses.

1. **The test set (300) is completely different from the private paper test set (100).** What the leaderboard currently reports (e.g., VoxTell 0.285) was computed on the private paper test set, not on the challenge test set. Final challenge scores should be expected to differ.

2. **The challenge split is not yet fully public.** The 200 validation cases are released in June; the 300 test cases are never directly accessible. Before June, the only reliable local validation uses the 50 cases from the paper.

3. **The output format (F, H, W, D)** is non-trivial. The coordination between per-finding predictions and the ordering of findings in the file must exactly replicate the ground truth order. This should be verified with `rexrank_eval.py` before submitting anything real.

4. **There is no explicit submission limit before the deadline**, but the process is manual (email to the organizer). This suggests avoiding overuse — better to send polished, not experimental, submissions.

5. **Co-authorship requires top-3.** Top-5 or top-10 do not qualify for the challenge paper. However, the "no embargo period" rule allows independent publication regardless of placement, which serves as a safety net for a doctoral thesis.

6. **Partial training annotations.** This is acknowledged in the original paper, but the challenge rules do not specify how to handle it. The challenge assumes it is the participant's responsibility.

7. **"External data permitted" is explicit.** This enables:
   - Fine-tuning from VoxTell (public checkpoint).
   - Using CT-RATE for pretraining.
   - Integrating auxiliary pulmonary lesion datasets (LIDC, NSCLC Radiomics, etc.).
   - Using any pre-trained foundation model.
   All sources must be documented in the submission.

8. **"Fully automatic" without exceptions.** This rules out any case-specific post-processing, adaptive per-case thresholds, etc. The same pipeline must run on all 300 cases without modification.

9. **Practical timeline**: from launch (June) to deadline (September) there are ~12 effective weeks. Any critical work on the submission pipeline should be ready before the 200-case val set is released.