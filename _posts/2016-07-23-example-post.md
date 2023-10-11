---
layout: post
title: 'Simple but Effective: CLIP Embeddings for Embodied AI'
excerpt: 'Contrastive language image pretraining (CLIP) encoders have been shown to be beneficial for a range of visual tasks from classification and detection to captioning and image manipulation. We investigate the effectiveness of CLIP visual backbones for Embodied AI tasks. We build incredibly simple baselines, named EmbCLIP, with no task specific architectures, inductive biases (such as the use of semantic maps), auxiliary tasks during training, or depth maps—yet we find that our improved baselines perform very well across a range of tasks and simulators. EmbCLIP tops the RoboTHOR ObjectNav leaderboard by a huge margin of 20 pts (Success Rate). It tops the iTHOR 1-Phase Rearrangement leaderboard, beating the next best submission, which employs Active Neural Mapping, and more than doubling the % Fixed Strict metric (0.08 to 0.17). It also beats the winners of the 2021 Habitat ObjectNav Challenge, which employ auxiliary tasks, depth maps, and human demonstrations, and those of the 2019 Habitat PointNav Challenge. We evaluate the ability of CLIP’s visual representations at capturing semantic information about input observations—primitives that are useful for navigation-heavy embodied tasks—and find that CLIP’s representations encode these primitives more effectively than ImageNet-pretrained backbones. Finally, we extend one of our baselines, producing an agent capable of zero-shot object navigation that can navigate to objects that were not used as targets during training. Our code and models are available at https://github.com/allenai/embodied-clip.'
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

This is a website template created with Jekyll that is designed to be hosted on Github pages. Jekyll is a static website generator, and Github pages provides a free and easy way to host websites created using Jekyll.

## What is a static website generator?
A static website generator is a program that allows for a website to be created using alternatives to HTML. In this case we are using a simple text markup language called Markdown to create and format the content on the pages. Jekyll can interpret this and convert it to html that can be rendered in any browser.

## Why should I use a static website?
A static website, simply put, is easier to manage than just about any other option out there for a simple website. Since it does not rely on any additional web application, it can be hosted on any webhosting server. It does not require regular updates like many dynamic websites such as WordPress or Drupal require. Everything that the website needs to work is contained within one directory, making it incredibly easy to move.

## How is this template different?
This template has been optimized for ease of use. In the next set of instructions you will see that there are less than five files that you need to edit in order to customize the look of the website.
