---
layout: post
title: 'Embodied CLIP'
excerpt: 'A blog post on the paper "Simple but Effective: CLIP Embeddings for Embodied AI"'
modified: '11/10/2023, 15:46:00'
tags:
  - intro
  - beginner
  - jekyll
  - tutorial
comments: true
category: blog
published: false
---

- [Introduction](#introduction)
- [Prior Work](#prior-work)
  - [Visual Encoders in Embodied AI](#visual-encoders-in-embodied-ai)
  - [CLIP Model](#clip-model)
- [Using CLIP in Embodied AI](#using-clip-in-embodied-ai)
- [Results](#results)
- [Conclusion](#conclusion)


## Introduction

Embodied AI is a fascinating subfield of artificial intelligence research. The aim is to create intelligent agents that can interact with the physical world in a human-like way. The main challenge in Embodied AI is to develop models that can understand and reason about the visual, linguistic, or sensory information that they receive from their surroundings and use this information to perform tasks in static or dynamic environments.

<figure style="text-align:center;">
  <img src="{{ site.url | default: '' }}{{ site.baseurl }}/images/blog/embodied-ai-intelligent-agent.png" width="50%" heighth="50%">
</figure>

In this blog post, I will talk about a paper titled ["Simple but Effective: CLIP Embeddings for Embodied AI"](https://arxiv.org/abs/2111.09888)[^1] by [Khandelwal](https://apoorvkh.com/) et al which is a recent work that uses CLIP embeddings to perform various Embodied AI tasks. The work was published in the CVPR 2022.

I will briefly mention the prior work in Embodied AI and then talk about the CLIP embeddings and how the model can perform Embodied AI tasks.

## Prior Work

Embodied AI is a fast-growing field of research. The advancements in simulators and the introduction of new tasks and challenges have led to considerable progress in this field. Since the task is to create agents that can interact with the physical world, embodied AI research can and should benefit from different aspects of AI research, such as computer vision, natural language processing, reinforcement learning, robotics, etc. 

### Visual Encoders in Embodied AI

Visual information is one of the fundamental sources of information that Embodied AI agents receive from the environment. Therefore, it is necessary to have a reliable vision encoder that can extract useful information from the visual input. Due to the inefficiency and latency of larger models, most of the prior work in EAI benefited from only lightweight CNN or ResNet models as visual encoders.

However, recent advancements in computer vision research have led to the development of more powerful visual encoders such as ViT, DeiT, etc. These models perform better than CNNs and ResNets in many computer vision tasks. Therefore, it is natural to think that these models can also be used as visual encoders in Embodied AI tasks. Although some research has shown that using larger ResNet models only leads to marginal improvements in performance, we will talk about how CLIP embeddings can increase the performance as visual encoders in different Embodied AI tasks.

### CLIP Model

OpenAI introduced CLIP (Contrastive Language–Image Pre-training)[^2] in PRML on 2021. They show that scaling a simple pre-training task is powerful enough to achieve competitive zero-shot results on various image classification benchmarks. They use the texts paired with images from the internet as the training data. 

<figure style="text-align:center;">
  <img src="{{ site.url | default: '' }}{{ site.baseurl }}/images/blog/clip.png">
  <figcaption> Source of image: https://openai.com/research/clip </figcaption>
</figure>

CLIP pre-trains an image encoder and a text encoder simultaneously to predict which images were paired with which texts from a large dataset. Based on this behavior, CLIP performs zero-shot classification on a variety of image classification benchmarks. The classification task can be performed by turning labels into image descriptions. For instance, the description "a photo of a dog" can be created from the label "dog". Then, the CLIP model can be used to predict the label of an image by comparing the image with the description.

CLIP models are highly efficient, flexible, and generic. It uses Vision Transformers over ResNets, which provides a 3x gain in compute efficiency. It is flexible in the sense that it can be used for various tasks such as image classification, object detection, image segmentation, etc. Generic in the sense that it can be used for a variety of domains such as natural images, medical images, satellite images, etc.

While it usually performs well in recognizing regular objects, it has difficulty understanding abstract concepts, such as counting and prepositional phrases, and also complex tasks. It can also fall short of giving fine-grained classification results such as models of cars or breeds of dogs.


## Using CLIP in Embodied AI

Although CLIP is a simple model, this visual backbone can be used to perform Embodied AI tasks. The authors of the paper show that CLIP embeddings can be used to perform Embodied AI tasks such as object goal navigation, point goal navigation, and room rearrangement. They show that CLIP embeddings can be used as a substitute for visual encoders used to solve these challenges.

<figure style="text-align:center;">
  <img src="{{ site.url | default: '' }}{{ site.baseurl }}/images/blog/high-level-clip-embodied-ai.png" width="50%" height="50%">
  <figcaption> Source of image: Khandelwal et al, 2022 </figcaption>
</figure>

The high-level architecture used to solve each challenge is similar to that shown above. The RGB image is fed into the frozen CLIP model and then a simple trainable CNN, The embeddings and the goal embedding and then fed into a GRU network to predict the action. The authors show that this simple high-level approach can be used to solve all three challenges.

## Results

This simple architecture achieves SOTA or near-SOTA approaches on most of the challenges the authors have tested.


<table cellspacing="10px" cellpadding="10px" style="text-align:center;">
    <tr>
        <td style="text-align: center;">
          <img src="{{ site.url | default: '' }}{{ site.baseurl }}/images/blog/embodied-ai-robothor-objectnav.png">
        </td>
        <td style="text-align: center;">
            <img src="{{ site.url | default: '' }}{{ site.baseurl }}/images/blog/embodied-ai-ithor-rearrangement.png">
        </td>
    </tr>
    <tr>
        <td style="text-align: center;">
          <img src="{{ site.url | default: '' }}{{ site.baseurl }}/images/blog/embodied-ai-habitat-objectnav.png">
        </td>
        <td style="text-align: center;">
          <img src="{{ site.url | default: '' }}{{ site.baseurl }}/images/blog/embodied-ai-habitat-pointnav.png">
        </td>
    </tr>
    <caption style="caption-side:bottom">Several Benchmark Results. Source: Khandelwal et al, 2022</caption>
</table>

Additionally, the authors investigated CLIP performance on sub-tasks in some designated experiments to support the hypothesis that CLIP features natively embed visual information that is relevant to navigational and similar embodied tasks. The experiments show that CLIP embeddings represent the primitives of the physical world in tasks such as object presence, object localization, free space, and reachability.

<figure style="text-align:center;">
  <img src="{{ site.url | default: '' }}{{ site.baseurl }}/images/blog/embodied-ai-visual-encoder-evaluations.png">
  <figcaption> Examples of visual encoder evaluation tasks. Source: Khandelwal et al, 2022 </figcaption>
</figure>

In the reachability evaluation, they train the models to predict whether a particular object is reachable from the robot arm. 

In the object presence on grid evaluation, they train the models to predict whether a particular object is present in one of the grid cells or not in a 3x3 grid.

In the object presence evaluation, they train the models to predict whether a particular object is present in the scene.

In the free space evaluation, they train the models to predict the amount of free space available in front of the agent.

<figure style="text-align: center;">
  <img src="{{ site.url | default: '' }}{{ site.baseurl }}/images/blog/embodied-ai-visual-encoder-evaluations-table.png" width="50%" height="50%">
  <figcaption> Results of visual encoder evaluation tasks. Source: Khandelwal et al, 2022 </figcaption>
</figure>



## Conclusion

In this paper, the authors show that the visual encoders were a bottleneck for embodied AI research. They show that CLIP embeddings can be used as a substitute for visual encoders and achieve SOTA or near-SOTA results on several Embodied AI tasks. They also show that CLIP embeddings can be used to perform Embodied AI tasks in different environments such as RoboTHOR, iTHOR, and Habitat. 

More importantly, they show that like every other field of AI, Embodied AI research also benefited extensively from the advancements in other fields of AI research. I think that this paper is a good example of how the advancements in computer vision research can be used to perform Embodied AI tasks.

[^1]: Khandelwal, Apoorv, et al. "Simple but effective: Clip embeddings for embodied ai." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

[^2]: Radford, Alec, et al. "Learning transferable visual models from natural language supervision." International conference on machine learning. PMLR, 2021.

***This blog post has been written as part of the "CS 6604: Embodied AI" course taught by [Ismini Lourentzou](https://isminoula.github.io/).***


