---
layout: page
excerpt: About Me...
published: true
---

<div class="hero-quote">
> <em>I worked on making sense of images,<br>
> Now I am working on images making sense.</em>
</div>

# About Me

<div class="lead-text">
I am Tuna ([tu-nah]), a Ph.D. student in Computer Science at <strong>Virginia Tech</strong>, advised by <a href="https://pinguar.org/">Dr. Pinar Yanardag Delul</a>, and a member of the <a href="https://gemlab-vt.github.io/">GemLab</a>.
</div>

My research focuses on developing **controllable** and **interpretable** generative models across **images**, **video**, and **language**. I design alignment objectives for **diffusion** and **autoregressive** architectures to enable efficient, user-aligned generation without post-hoc tuning.

Before joining VT, I worked across startups and industry labs, building real-time image generation services and scalable ML systems in production. This mix of applied and theoretical experience enables me to create generative models that are both research-grade and deployable.

At the core of my work is a belief that generative AI should not only create high-quality content, but do so transparently and in alignment with user goals.

## Current Research Interests

<div class="research-interests">
<ul>
<li><strong>Autoregressive models</strong> for image, and video generation</li>
<li><strong>Controllable generation</strong> in diffusion and autoregressive models</li>
<li><strong>Token-level interpretability</strong> in transformers (image/video/LLM)</li>
<li><strong>Zero-shot image/video editing</strong></li>
</ul>
</div>  

{% include _news-recent.html %}

## Latest Blog Posts

{% for post in site.posts limit:3 %}
<div class="latest-post">
  <h3><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
</div>
{% endfor %}

{% include _publications-featured.html %}
