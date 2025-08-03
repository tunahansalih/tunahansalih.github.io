---
layout: page
excerpt: About Me...
published: true
---

My research focuses on developing **controllable** and **interpretable** generative models across **images**, **video**, and **language**. I design alignment objectives for **diffusion** and **autoregressive** architectures to enable efficient, user-aligned generation without post-hoc tuning.

Before joining VT, I worked across startups and industry labs, building real-time image generation services and scalable ML systems in production. This mix of applied and theoretical experience enables me to create generative models that are both research-grade and deployable.

At the core of my work is a belief that generative AI should not only create high-quality content, but do so transparently and in alignment with user goals. *I worked on making sense of images, now I am working on images making sense.*

<div class="academic-section">
  <h2>Research Interests</h2>
  <div class="research-interests-academic">
    <div class="research-interest-card">
      <h4>Autoregressive Vision Models</h4>
      <p>Developing next-generation autoregressive architectures for high-fidelity image and video synthesis, focusing on scalability and controllability.</p>
    </div>
    <div class="research-interest-card">
      <h4>Controllable Generation</h4>
      <p>Creating alignment objectives for diffusion and autoregressive models that enable precise user control without compromising generation quality.</p>
    </div>
    <div class="research-interest-card">
      <h4>Mechanistic Interpretability</h4>
      <p>Understanding how transformers process visual and textual information through token-level analysis and attention mechanism studies.</p>
    </div>
    <div class="research-interest-card">
      <h4>Zero-shot Editing</h4>
      <p>Enabling intuitive image and video manipulation through natural language interfaces and novel training-free approaches.</p>
    </div>
  </div>
</div>  

{% include _news-recent.html %}

<div class="academic-section">
  <h2>Latest Blog Posts</h2>
  {% for post in site.posts limit:3 %}
  <div class="latest-post">
    <h3><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
  </div>
  {% endfor %}
</div>

{% include _publications-featured.html %}
