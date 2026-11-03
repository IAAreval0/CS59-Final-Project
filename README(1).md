# CS559 Final Project Proposal

This folder contains the project proposal and code scaffolding for the CS559 Visual Perception and Learning final project.

**Author:** Anthony Alarcon

## Topic
**Classical Feature-based Methods vs. Convolutional Neural Networks for Image Classification on CIFAR-10**

Chosen because:
- Directly builds on Lecture 06 (KNN, linear classifiers, CIFAR-10)
- Aligns with the object detection lecture (HOG features)
- Matches the project guide’s recommendation to compare classical and deep models
- Feasible within a month

## Files
- `CS559_Final_Project_Proposal.docx` — Proposal document with problem statement, method outline, experiment plan, and next steps.
- `cifar10_starter.py` — Python scaffolding for data loading, a simple CNN, ResNet transfer learning skeleton, and classical (HOG+SVM) placeholders.

## Next Steps
1. Review and expand the proposal document (Related Work, methods, etc.).
2. Set up a shared Git repository and environment (`requirements.txt`).
3. Implement and run the classical baseline.
4. Train deep models (start with the SimpleCNN or fine-tune ResNet-18).
5. Generate all required experimental results (curves, tables, confusion matrices, qualitative analysis).
6. Write the full report using the structure in the proposal.

## Quick Start
```bash
pip install -r requirements.txt
python cifar10_starter.py
```

The code provides a starting point for implementation and experiments.
