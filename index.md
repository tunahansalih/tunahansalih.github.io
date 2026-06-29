---
layout: page
excerpt: About Me...
published: true
---

<div class="home-hero">
  <p class="home-hero__title">Ph.D. Candidate in Computer Science, Virginia Tech</p>
  <p class="home-hero__summary">
    I am a Ph.D. candidate at Virginia Tech, advised by <a href="https://pinguar.org/" target="_blank" rel="noopener">Prof. Pinar Yanardag</a>.
    I study the internal representations of generative models, how they encode meaning, structure, and style, to make
    image and video generation easier to understand and control.
  </p>
  <p class="home-hero__summary">
    My recent work spans controllable generation, diffusion model distillation, and analysis of how generative models
    represent concepts internally. I want to understand why these systems behave the way they do, not just improve their outputs.
  </p>
  <div class="research-interests">
    <span class="research-tag">Diffusion Models</span>
    <span class="research-tag">Video Generation</span>
    <span class="research-tag">Controllable Generation</span>
    <span class="research-tag">Mechanistic Interpretability</span>
    <span class="research-tag">Model Distillation</span>
    <span class="research-tag">Flow Matching</span>
  </div>
  <nav class="home-link-list" aria-label="Profile links">
    <a href="{{ '/resume/' | relative_url }}">Resume</a>
    <a href="https://scholar.google.com/citations?user={{ site.owner.google.scholar }}" target="_blank" rel="noopener">Google Scholar</a>
    <a href="https://linkedin.com/in/{{ site.owner.linkedin }}" target="_blank" rel="noopener">LinkedIn</a>
    <a href="mailto:{{ site.owner.email }}">Email</a>
    <a href="https://github.com/{{ site.owner.github }}" target="_blank" rel="noopener">GitHub</a>
  </nav>
</div>

<section class="recruiter-callout" aria-label="Recruiting availability">
  <div class="recruiter-callout__content">
    <p class="recruiter-callout__summary">
      I expect to complete my Ph.D. in Spring 2027 and am exploring full-time Research Scientist and Applied Scientist
      roles focused on generative vision, multimodal ML, and controllable image/video generation.
      You can view my <a href="{{ '/resume/' | relative_url }}">resume</a> or reach me by
      <a href="mailto:{{ site.owner.email }}">email</a> and
      <a href="https://linkedin.com/in/{{ site.owner.linkedin }}" target="_blank" rel="noopener">LinkedIn</a>.
    </p>
  </div>
  <a class="recruiter-callout__button" href="mailto:{{ site.owner.email }}?subject=Full-time%20Research%20Scientist%20%2F%20Applied%20Scientist%20role">
    Discuss opportunities
  </a>
</section>

## Highlights
<ul class="home-highlights">
  <li>Publications at CVPR 2026, AAAI 2026, ICCV 2025 (Highlight), ICML 2025 (Oral), and CVPR 2024.</li>
  <li>Research internships at Waymo (2026), Amazon (2025), and Adobe (2024).</li>
  <li>Industry deployment experience with diffusion-based services serving over 5 million daily requests.</li>
  <li>Co-organizer of the ICCV 2025 and CVPR 2026 Personalization in Generative AI (P13N) workshops.</li>
</ul>

## Experience
<ul class="home-highlights">
  <li><strong>Waymo</strong> - Research Intern (May 2026 - Present)</li>
  <li><strong>Virginia Tech</strong> - Research Assistant (Aug 2023 - Present)</li>
  <li><strong>Amazon</strong> - Applied Scientist Intern (May 2025 - Aug 2025)</li>
  <li><strong>Adobe</strong> - Research Intern, Video Generative AI (May 2024 - Aug 2024)</li>
  <li><strong>Lyrebird Studio</strong> - Machine Learning Engineer (Nov 2022 - Aug 2023)</li>
  <li><strong>Vispera</strong> - Computer Vision Research Engineer (Oct 2019 - Nov 2022)</li>
</ul>

{% include _news-recent.html %}

{% include _publications-featured.html %}
