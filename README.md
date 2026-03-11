# Instagram Sentiment Analyzer

This repository provides a simple script to scrape comments from an Instagram
post and perform basic sentiment analysis.  It uses `instaloader` for scraping
and either `textblob` or the Hugging‑Face `transformers` pipeline for
sentiment scoring.

## Setup

1. Create a Python virtual environment and activate it:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate      # Windows
   source venv/bin/activate       # macOS / Linux
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file lists all of the libraries the project uses.  A
   matching `torch`/`torchvision` install is recommended if you intend to use
   the Transformers pipeline; otherwise the script will silently fall back to
   `textblob`.

   After installation you may need to download the TextBlob corpora:
   ```bash
   python -m textblob.download_corpora
   ```

3. Update `project.py` with your Instagram username and password, and an
   example post URL.

## Usage

Run the main script:

```bash
python project.py
```

If your credentials are invalid or left as the placeholders the script will
print an error and exit.  When a cached session file is available, it will be
used instead of re‑logging in.

Results are written to `post_data/<shortcode>.csv` and a sentiment graph +
word cloud are displayed.

## Notes

* The example post URL in the repository is hard‑coded – you can change it or
  pass a different one by editing the script.
* The Transformers pipeline import can fail if the installed PyTorch/
  torchvision versions are incompatible; the code handles that gracefully.
