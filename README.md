# Tuna Meral — Academic Website

Personal academic website for Tuna Meral (Ph.D. Candidate in Computer Science at Virginia Tech).

Live site: [tmeral.com](https://tmeral.com)

## Local Development

No Ruby, Bundler, or external packages required. Uses standard Python:

```bash
python3 dev.py
```

This builds the site and starts a local server on `http://localhost:4000`.

To build without starting the server:

```bash
python3 build.py
```

## Structure

```
├── data/
│   ├── profile.json            # Name, role, affiliation, social links
│   ├── publications.json       # All papers, venues, links, abstracts, BibTeX
│   └── news.json               # All milestones and awards
├── _templates/
│   ├── base.html               # Shared HTML shell (head, nav, sidebar, footer)
│   └── pages/                  # Page body templates (index, resume, pubs, news, 404)
├── build.py                    # Zero-dependency site generator
├── dev.py                      # Build + local preview server
├── assets/
│   └── css/style.css           # Single shared stylesheet
├── images/                     # Photos and publication figures
├── CNAME                       # Custom domain configuration (tmeral.com)
└── .github/workflows/
    └── deploy.yml              # GitHub Actions automated build & deployment
```

## Deployment

Deployments are automated via GitHub Actions on every push to `main`.
GitHub Actions runs `python3 build.py` and deploys the static artifact directly to GitHub Pages.
