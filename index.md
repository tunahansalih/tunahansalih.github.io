---
layout: page
excerpt: About Me...
published: true
---

## Hey there! I’m Tuna [tu-nah]👋

I research **generative AI** at Virginia Tech, mainly figuring out how to tell image and video models *exactly* what to do (and why they do it). My code often runs after midnight, so I keep the coffee machine busy and the GPU fans even busier.

### What I’m proud of

* **CVPR 2024**, **ICML 2025 (oral)**, and **ICCV 2025 (highlight)** papers that tinker with diffusion models, personalization and interpretability.
* Internships with **Amazon AGI** and **Adobe FireFly**, plus fun collabs with **Google**.  
* Co-organizer of the **P13N workshop on Personalization in Generative AI at ICCV 2025**.  
* Once deployed an image-generation services that millions of people actually used—without it catching fire.

### Why I do this

I want creators to treat generative models like trusty sidekicks, not black boxes. My work mixes theory, experiments, and a dose of engineering pragmatism so the results can ship, not just sit on arXiv.

### Outside the lab

You’ll occasionally find fresh paper notes and open-source snippets on my blog or Twitter. Offline, I’m usually looking at my monitor—coffee in hand—while stress-testing newly trained models.

### Let's chat

If you're into vision generative AI, or just want to trade GPU tales, drop me a line!

<div class="academic-section">
  <h2>📝 Latest from the Blog</h2>
  <div style="margin-bottom: 2rem;">
    {% for post in site.posts limit:3 %}
    <div class="blog-post">
      <h3 class="blog-post-title">
        <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      </h3>
      <div class="blog-post-meta">
        📅 {{ post.date | date: "%B %d, %Y" }}
        {% if post.excerpt %}
        • ⏱️ {{ post.content | number_of_words | divided_by: 200 }} min read
        {% endif %}
      </div>
      {% if post.excerpt %}
      <div class="blog-post-content">
        {{ post.excerpt | strip_html | truncate: 200 }}
      </div>
      {% endif %}
    </div>
    {% endfor %}
    <div style="text-align: center; margin-top: 1.5rem;">
      <a href="{{ site.url | default: '' }}{{ site.baseurl }}/blog" class="download-button" style="display: inline-block; padding: 0.75rem 2rem; text-decoration: none;">
        View All Posts →
      </a>
    </div>
  </div>
</div>

<div class="academic-section">
  <h2>Research Interests</h2>
  <div class="research-interests-academic">
    <div class="research-interest-card">
      <h4>Autoregressive Vision Models</h4>
      <p>Developing next-generation autoregressive architectures for high-fidelity and efficient image and video synthesis.</p>
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

{% include _publications-featured.html %}
