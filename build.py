#!/usr/bin/env python3
"""
Zero-dependency static site builder for Tuna Meral's academic website.
Compiles data/*.json and _templates/ into static HTML files.
"""

import os
import json
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / "data"
TEMPLATES_DIR = ROOT_DIR / "_templates"
PAGES_DIR = TEMPLATES_DIR / "pages"
SITE_DIR = ROOT_DIR / "_site"

def load_json(filename):
    with open(DATA_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)

def render_profile_card(profile):
    social_html = []
    for s in profile.get("social", []):
        social_html.append(
            f'    <a href="{s["url"]}" class="author-social" title="{s["name"]}" aria-label="{s["name"]}">'
            f'<i class="{s["icon"]}"></i></a>'
        )
    social_links_str = "\n".join(social_html)

    return f"""<aside class="academic-author-card profile-card">
  <div class="profile-card__photo-wrap">
    <img src="{profile['photo']}" class="academic-bio-photo profile-card__photo"
      alt="{profile['name']} portrait">
  </div>

  <h3 class="academic-author-name profile-card__name">{profile['name']}</h3>
  <p class="profile-card__role">{profile['role']}</p>
  <p class="profile-card__affiliation">{profile['affiliation']}</p>

  <div class="social-icons profile-card__social">
{social_links_str}
  </div>
</aside>"""

def render_sidebar_news(news_items, count=3):
    items_html = []
    for item in news_items[:count]:
        links_html = ""
        if item.get("links"):
            link_tags = [f'<a href="{l["url"]}" class="sidebar-news__link" target="_blank" rel="noopener">[{l["text"]}]</a>' for l in item["links"]]
            links_html = f"""
            <span class="sidebar-news__links">
              {" ".join(link_tags)}
            </span>"""

        items_html.append(f"""      <li class="sidebar-news__item">
        <time class="sidebar-news__date">{item['date']}</time>
        <div class="sidebar-news__body">
          {item['content']}{links_html}
        </div>
      </li>""")

    list_str = "\n".join(items_html)
    return f"""<section class="sidebar-news" aria-label="Recent news">
  <h3 class="sidebar-news__title">Recent news</h3>
  <ul class="sidebar-news__list">
{list_str}
  </ul>
  <p class="sidebar-news__more">
    <a href="/news/">All news &rarr;</a>
  </p>
</section>"""

def render_featured_publications(publications):
    cards = []
    for pub in publications:
        if not pub.get("selected"):
            continue

        link = pub.get("link") or (next(iter(pub["links"].values())) if pub.get("links") else "#")
        img_src = pub.get("image", "")

        links_html = ""
        if pub.get("links"):
            link_tags = [f'<a href="{url}" target="_blank" rel="noopener">{name}</a>' for name, url in pub["links"].items()]
            links_html = f"""
            <div class="featured-pub-links">
              {" ".join(link_tags)}
            </div>"""

        cards.append(f"""      <article class="featured-pub-card">
        <figure class="featured-pub-image">
          <a class="featured-pub-image-frame" href="{link}" target="_blank" rel="noopener">
            <img src="{img_src}" onerror="this.style.display='none'; this.parentElement.classList.add('featured-pub-image-frame--empty');" alt="{pub['title']}" />
          </a>
        </figure>

        <div class="featured-pub-content">
          <h3>
            <a href="{link}" target="_blank" rel="noopener">{pub['title']}</a>
          </h3>
          <p class="featured-pub-venue">
            {pub['venue']} &middot; {pub['year']}
          </p>
          <p class="featured-pub-authors">{pub['authors']}</p>
{links_html}
        </div>
      </article>""")

    return "\n".join(cards)

def render_all_publications(publications):
    entries = []
    for pub in publications:
        link = pub.get("link") or (next(iter(pub["links"].values())) if pub.get("links") else "#")

        title_html = f'<a href="{link}" target="_blank" rel="noopener">{pub["title"]}</a>' if link else pub["title"]

        desc_html = f'<p class="publication-description-academic">{pub["description"]}</p>' if pub.get("description") else ""

        links_html = ""
        if pub.get("links"):
            link_tags = [f'<a href="{url}" class="academic-link" target="_blank" rel="noopener">{name}</a>' for name, url in pub["links"].items()]
            links_html = f"""
          <div class="publication-links-academic">
            {" ".join(link_tags)}
          </div>"""

        bibtex_html = ""
        if pub.get("bibtex"):
            bibtex_html = f"""
          <div class="citation-actions">
            <button type="button" class="academic-link citation-toggle" onclick="toggleCitation('{pub['id']}')">BibTeX</button>
            <button type="button" class="academic-link copy-citation" onclick="copyCitation('{pub['id']}', this)">Copy BibTeX</button>
          </div>

          <div id="{pub['id']}" class="citation-text-academic" hidden="">
            <pre>{pub['bibtex']}</pre>
          </div>"""

        entries.append(f"""      <article class="academic-publication">
        <h3 class="publication-title-academic">
          <span class="publication-year">{pub['year']}</span>
          {title_html}
        </h3>
        <p class="publication-venue-academic">{pub['venue']}</p>
        <p class="publication-authors-academic">{pub['authors']}</p>
        {desc_html}
{links_html}
{bibtex_html}
      </article>""")

    return "\n".join(entries)

def render_resume_publications(publications):
    blocks = []
    for pub in publications:
        links_parts = []
        if pub.get("links"):
            for name, url in pub["links"].items():
                links_parts.append(f'<a href="{url}">{name}</a>')
        links_str = f"\n{' &middot; '.join(links_parts)}" if links_parts else ""

        blocks.append(f"""<p><code>{pub['year']}</code>
<strong>{pub['title']}</strong>
{pub['authors']}
<em>{pub['venue']}</em>{links_str}</p>""")

    return "\n\n".join(blocks)

def render_news_timeline(news_items):
    by_year = {}
    for item in news_items:
        y = item["year"]
        by_year.setdefault(y, []).append(item)

    timeline_blocks = []
    for year in sorted(by_year.keys(), reverse=True):
        timeline_blocks.append(f'      <h3 class="timeline-year">{year}</h3>\n')
        for item in by_year[year]:
            links_html = ""
            if item.get("links"):
                link_tags = [f'<a href="{l["url"]}" class="timeline-link" target="_blank" rel="noopener">{l["text"]}</a>' for l in item["links"]]
                links_html = f"""
            <div class="timeline-links">
              {" ".join(link_tags)}
            </div>"""

            timeline_blocks.append(f"""        <article class="timeline-item" data-type="{item.get('type', 'milestone')}">
          <p class="timeline-date">{item['date']}</p>
          <div class="timeline-content">
            {item['content']}
          </div>
{links_html}
        </article>""")

    return "\n".join(timeline_blocks)

def build():
    profile = load_json("profile.json")
    publications = load_json("publications.json")
    news = load_json("news.json")

    base_template = open(TEMPLATES_DIR / "base.html", "r", encoding="utf-8").read()

    rendered_profile = render_profile_card(profile)
    rendered_sidebar_news = render_sidebar_news(news, count=3)
    rendered_featured_pubs = render_featured_publications(publications)
    rendered_all_pubs = render_all_publications(publications)
    rendered_resume_pubs = render_resume_publications(publications)
    rendered_timeline = render_news_timeline(news)

    pages = [
        {
            "template": "index.html",
            "output": "index.html",
            "title": "Tuna Meral &#8211; Ph.D. Candidate, Virginia Tech",
            "page_description": "Ph.D. candidate at Virginia Tech researching controllable and efficient generative vision models (image and video generation). On the job market for Spring 2027.",
            "page_url": "/",
            "current_nav": "home",
            "page_header": "",
            "sidebar_extra": f"\n{rendered_sidebar_news}",
            "substitutions": {
                "{{featured_publications}}": rendered_featured_pubs
            }
        },
        {
            "template": "publications.html",
            "output": "publications/index.html",
            "title": "Publications &#8211; Tuna Meral",
            "page_description": "Publications and preprints by Tuna Meral on generative AI, spherical flow matching, and video diffusion models.",
            "page_url": "/publications/",
            "current_nav": "publications",
            "page_header": "\n        <h1>Publications</h1>\n      ",
            "sidebar_extra": "",
            "substitutions": {
                "{{publications_list}}": rendered_all_pubs
            }
        },
        {
            "template": "news.html",
            "output": "news/index.html",
            "title": "News &#8211; Tuna Meral",
            "page_description": "Recent research news, paper releases, fellowship awards, and career milestones for Tuna Meral.",
            "page_url": "/news/",
            "current_nav": "news",
            "page_header": "\n        <h1>News</h1>\n      ",
            "sidebar_extra": "",
            "substitutions": {
                "{{news_timeline}}": rendered_timeline
            }
        },
        {
            "template": "resume.html",
            "output": "resume/index.html",
            "title": "Resume &#8211; Tuna Meral",
            "page_description": "Curriculum vitae and academic resume for Tuna Meral, Ph.D. Candidate at Virginia Tech.",
            "page_url": "/resume/",
            "current_nav": "resume",
            "page_header": "\n        <h1>Resume</h1>\n      ",
            "sidebar_extra": "",
            "substitutions": {
                "{{resume_publications}}": rendered_resume_pubs
            }
        },
        {
            "template": "404.html",
            "output": "404.html",
            "title": "Page Not Found &#8211; Tuna Meral",
            "page_description": "Page not found.",
            "page_url": "/404.html",
            "current_nav": "",
            "page_header": "\n        <h1>Page Not Found</h1>\n      ",
            "sidebar_extra": "",
            "substitutions": {}
        }
    ]

    # Ensure output directories exist
    (ROOT_DIR / "publications").mkdir(exist_ok=True)
    (ROOT_DIR / "news").mkdir(exist_ok=True)
    (ROOT_DIR / "resume").mkdir(exist_ok=True)
    SITE_DIR.mkdir(exist_ok=True)
    (SITE_DIR / "publications").mkdir(exist_ok=True)
    (SITE_DIR / "news").mkdir(exist_ok=True)
    (SITE_DIR / "resume").mkdir(exist_ok=True)

    for p in pages:
        page_body = open(PAGES_DIR / p["template"], "r", encoding="utf-8").read()

        # Guard: Strip accidental full-page wrapper tags if pasted into page templates
        if "<header" in page_body or "<footer" in page_body or "<body" in page_body:
            # Extract content from inside .article-wrap if present, or strip header/footer
            m_wrap = re.search(r'<div class="article-wrap">(.*?)</div>\s*</article>', page_body, re.DOTALL)
            if m_wrap:
                page_body = m_wrap.group(1).strip()
            else:
                page_body = re.sub(r'<header.*?</header>', '', page_body, flags=re.DOTALL)
                page_body = re.sub(r'<footer.*?</footer>', '', page_body, flags=re.DOTALL)
                page_body = re.sub(r'<head.*?</head>', '', page_body, flags=re.DOTALL)

        for placeholder, value in p["substitutions"].items():
            page_body = page_body.replace(placeholder, value)

        html = base_template
        html = html.replace("{{title}}", p["title"])
        html = html.replace("{{page_description}}", p["page_description"])
        html = html.replace("{{page_url}}", p["page_url"])
        html = html.replace("{{profile_card}}", rendered_profile)
        html = html.replace("{{sidebar_extra}}", p["sidebar_extra"])
        html = html.replace("{{page_header}}", p["page_header"])
        html = html.replace("{{content}}", page_body)

        # Active nav classes
        for nav in ["home", "publications", "news", "resume"]:
            active_str = " active" if nav == p["current_nav"] else ""
            html = html.replace(f"{{{{nav_active_{nav}}}}}", active_str)

        # Write to root and to _site
        out_root = ROOT_DIR / p["output"]
        out_site = SITE_DIR / p["output"]
        with open(out_root, "w", encoding="utf-8") as f:
            f.write(html)
        with open(out_site, "w", encoding="utf-8") as f:
            f.write(html)

    # Copy static assets into _site
    for item in ["assets", "images"]:
        src = ROOT_DIR / item
        dst = SITE_DIR / item
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    for single_file in ["CNAME", ".nojekyll", "favicon.ico", "favicon.png"]:
        src = ROOT_DIR / single_file
        dst = SITE_DIR / single_file
        if src.exists():
            shutil.copy2(src, dst)

    print("Site built successfully! HTML generated in root and _site/.")

if __name__ == "__main__":
    build()
