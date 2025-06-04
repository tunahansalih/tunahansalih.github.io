# Personal Website

This is my personal website built with Jekyll. It includes my resume, blog posts, publications, and news.

## Local Development Setup

1. Install Ruby (if not already installed):
   ```bash
   # For macOS using Homebrew
   brew install ruby
   ```

2. Install Jekyll and Bundler:
   ```bash
   gem install jekyll bundler
   ```

3. Install dependencies:
   ```bash
   bundle install
   ```

4. Start the local server:
   ```bash
   bundle exec jekyll serve
   ```

5. Visit `http://localhost:4000` in your browser

## Project Structure

- `_posts/` - Blog posts
- `_data/` - Site data and configurations
- `_includes/` - Reusable HTML components
- `_layouts/` - Page layouts
- `assets/` - Static assets (CSS, JS, images)
- `images/` - Image files
- `_sass/` - SCSS stylesheets

## Adding Content

### Blog Posts
Create new posts in `_posts/` with the format: `YYYY-MM-DD-title.md`

### Jupyter Notebooks
1. Save your notebook as `.ipynb`
2. Convert to markdown using:
   ```bash
   jupyter nbconvert --to markdown path/to/notebook.ipynb
   ```
3. Move the generated `.md` file to `_posts/`

### Images and Media
- Place images in the `images/` directory
- Reference them in markdown using: `![alt text](/images/filename.jpg)`

## Deployment
The site is automatically deployed when changes are pushed to the main branch.
