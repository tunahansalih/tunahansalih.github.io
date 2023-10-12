---
layout: post
title: 'Simple but Effective: CLIP Embeddings for Embodied AI'
excerpt: 'A blog post on the paper "Simple but Effective: CLIP Embeddings for Embodied AI"'
modified: '10/11/2023, 15:46:00'
tags:
  - intro
  - beginner
  - jekyll
  - tutorial
comments: true
category: blog
published: true
---

- [Introduction](#introduction)
- [Prior Work](#prior-work)
  - [Visual Encoders in Embodied AI](#visual-encoders-in-embodied-ai)
  - [CLIP and CLIP based models](#clip-and-clip-based-models)
- [Simple Baselines with CLIP Backbones](#simple-baselines-with-clip-backbones)
- [Conclusion](#conclusion)


## Introduction

Embodied AI is an interesting subfield of artificial intelligence research where the aim is to create intelligent agents that can interact with the physical world in a human-like way. The main challenge in Embodied AI is to develop models that can understand, maybe even reason, about the visual, linguistic or sensory information that they receive from the environment, and use this information to perform tasks in a static or dynamic environments.

In this blog post I will talk about a paper titled ["Simple but Effective: CLIP Embeddings for Embodied AI"](https://arxiv.org/abs/2111.09888)[^1] by [Apoorv Khandelwal](https://apoorvkh.com/) et al which is a recent work that uses CLIP embeddings to perform Embodied AI tasks. The paper is published in the CVPR 2022.

In this blog post, I will briefly mention the prior work in Embodied AI, and then talk about the CLIP embeddings and how they can be used to perform Embodied AI tasks.

## Prior Work

Embodied AI is a fast growing field of research. The advancements in simulators and introduction of new tasks and challenges have led to a lot of progress in this field. Since the task is to create agents that can interact with the physical world, embodied AI research can and should benefit from different aspects of AI research such as computer vision, natural language processing, reinforcement learning, robotics, etc. 

### Visual Encoders in Embodied AI

Visual information is one of the mains sources of information that Embodied AI agents receive from the environment. Therefore, it is important to have a good visual encoder that can extract useful information from the visual input. Due to the inefficiency and latency of larger models, most of the prior work in EAI benefited from only lightweight CNN or ResNet models as visual encoders.

However, recent advancements in computer vision research have led to the development of more powerful visual encoders such as ViT, DeiT, etc. These models have shown to perform better than CNNs and ResNets in many computer vision tasks. Therefore, it is natural to think that these models can also be used as visual encoders in Embodied AI tasks. Although some research have shown that using larger ResNet models anly leads to marginal improvements in performance, we will talk about how CLIP embeddings can increase the performance as visual encoders in different Embodied AI tasks.

### CLIP and CLIP based models

CLIP (Contrastive Language–Image Pre-training)[^2] is introduced by OpenAI and addresses the problems of zero-shot image classification problem by achieving 

<img src="{{ site.github.url }}/images/blog/clip.png" alt="{{ author.name }} bio photo">

## Simple Baselines with CLIP Backbones



## Conclusion


[^1]: Khandelwal, Apoorv, et al. "Simple but effective: Clip embeddings for embodied ai." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

[^2]: Radford, Alec, et al. "Learning transferable visual models from natural language supervision." International conference on machine learning. PMLR, 2021.

***This blog post has been written as part of the "CS 6604: Embodied AI" course taught by [Ismini Lourentzou](https://isminoula.github.io/)***


