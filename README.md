# Tuna Meral — Academic Website

Personal academic website for Tuna Meral (Ph.D. Candidate in Computer Science at Virginia Tech).

Live site: [tmeral.com](https://tmeral.com)

## Local Preview

No build tools, package managers, or dependencies are required. Preview locally with Python:

```bash
python3 -m http.server 4000
```

Open `http://localhost:4000` in your browser.

## Structure

```
├── index.html              # Homepage (Bio, job market announcement, selected research)
├── resume/index.html       # Full CV / Resume
├── publications/index.html # Full publications list
├── news/index.html         # News and milestones timeline
├── 404.html                # Not found page
├── .nojekyll               # GitHub Pages static flag (bypasses build pipeline)
├── CNAME                   # Custom domain configuration (tmeral.com)
├── assets/
│   └── css/style.css       # Shared stylesheet
└── images/
    ├── bio-avatar.jpg      # Profile photo
    └── publications/       # Paper figures
```

## Deployment

The repository is hosted on GitHub Pages. Pushing to `main` serves the static HTML directly.
