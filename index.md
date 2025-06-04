---
layout: page
excerpt: About Me...
published: true
---

> _I worked on making sense of images,\
> Now I am working on images making sense._

# About Me

I am Tuna ([tu-nah]), a Ph.D. student in Computer Science at **Virginia Tech**, advised by [Dr. Pinar Yanardag Delul](https://pinguar.org/), and a member of the [GemLab](https://gemlab-vt.github.io/).

My research focuses on developing **controllable** and **interpretable** generative models across **images**, **video**, and **language**. I design alignment objectives for **diffusion** and **autoregressive** architectures to enable efficient, user-aligned generation without post-hoc tuning.

Before joining VT, I worked across startups and industry labs, building real-time image generation services and scalable ML systems in production. This mix of applied and theoretical experience enables me to create generative models that are both research-grade and deployable.

At the core of my work is a belief that generative AI should not only create high-quality content, but do so transparently and in alignment with user goals.

## Current Interests

- Controllable generation in diffusion and autoregressive models
- Token-level interpretability in transformers (image/video/LLM)
- Steering of foundation models  
- Zero-shot image/video editing  

{% include _news-recent.html %}

## Latest Blog Posts

{% for post in site.posts limit:3 %}
<div class="latest-post">
  <h3><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
</div>
{% endfor %}

{% include _publications-featured.html %}
