# Project Outline

The aims, objectives, and research questions for this project are outlined in the [Project Outline](plan.md).

# Overview

This project provides adaptations of the transformers library code to facilitate the training of an MLLM for purposes of medical Visual Question Answering (VQA) tasks. Early works pretrained a visual encoder through Contrastive Language Image Pretraining (CLIP); however, this was later changed to use the pretrained CLIP model from OpenAI. Through feature alignment by a Multi-layer Perceptron, the embedded representation of the image is projected to an embedding space recognizable by a pre-trained Large Language Model (LLM). The original model architecture can be seen below:

<img src="Images/Outline.png" alt="Image 1" style="width: 100%; height: auto;">

# Project Progress Notes

Welcome to the project’s weekly progress updates. Here, you’ll find summaries of work completed, challenges encountered, and plans for the upcoming weeks.

## Weekly Progress

- [Week starting 18-11-24](week-18-11-24.md)
- [Week starting 25-11-24](week-25-11-24.md)
- [Week starting 09-12-24](week-09-12-24.md)

Each link above provides details for that week’s work.

## Gantt Chart

```mermaid
gantt
    title Project Progress
    dateFormat  YYYY-MM-DD
    section Week 1
    Task 1           :done,    des1, 2024-11-18,2024-11-24
    section Week 2
    Task 2           :active,  des2, 2024-11-25,2024-12-01
    section Week 3
    Task 3           :         des3, 2024-12-02,2024-12-08
    section Week 4
    Task 4           :         des4, 2024-12-09,2024-12-15