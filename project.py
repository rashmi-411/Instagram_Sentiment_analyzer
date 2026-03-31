import os.path
import csv
import pathlib
import pandas as pd
import instaloader
import emoji
import textblob
from instaloader import ConnectionException, Instaloader
from matplotlib import pyplot as plt
from wordcloud import WordCloud
# Transformers is used for sentiment analysis, but importing the pipeline can
# pull in vision dependencies (torchvision) which sometimes fail if the
# installed torch/torchvision versions are incompatible. Wrap import in a
# try/except and fall back to textblob if unavailable.
try:
    from transformers import pipeline
    sentiment_analyzer = pipeline(
        'sentiment-analysis',
        model='distilbert/distilbert-base-uncased-finetuned-sst-2-english'
    )
    _transformers_available = True
except Exception as _err:  # pragma: no cover
    # Keep the error message so the user can read why the pipeline failed.
    print("Warning: failed to initialize transformers pipeline:", _err)
    sentiment_analyzer = None
    _transformers_available = False


username = ""
password=""
url = "https://www.instagram.com/p/DCrpHvPMd2x/?utm_source=ig_web_copy_link&igsh=MzRlODBiNWFlZA=="
scrapFilepath = ""


def login(username, password):
    """Return an :class:`instaloader.Instaloader` that's logged in.

    The original project attempted to load a session file named after the user
    and fell back to credentials.  That code threw a rather opaque ``FileNotFound``
    or ``LoginException`` when things went wrong.  This version improves the
    behaviour:

    * If the session file is missing we try to log in with the supplied
      credentials.
    *  Placeholders in the global ``username``/``password`` variables now
      trigger an early error rather than attempting a bogus login.
    *  Login failures are caught and surfaced with a clear message.
    """
    if username == "user_id" or password == "your password":
        raise SystemExit("Please set a real Instagram username and password before running the script.")

    insta = instaloader.Instaloader()
    try:
        # instaloader.load_session_from_file will raise FileNotFoundError if the
        # session file doesn't exist, which we treat as 'no saved session'.
        insta.load_session_from_file(username)
        insta.context.username = username
        print("Logged in using existing session file")
    except FileNotFoundError:
        # no session; try logging in with credentials
        try:
            insta.login(username, password)
        except instaloader.exceptions.LoginException as exc:
            raise SystemExit(f"Login failed: {exc}")
        # update context username in case instaloader corrected it
        insta.context.username = insta.test_login() or username
        try:
            insta.save_session_to_file()
        except Exception:
            # non-fatal; we can continue even if saving failed
            pass
        print("Logged in using credentials and session saved")
    return insta


def scrapData(insta, url):
    #get the shortcode
    shortcode = str(url[28:39])
    #make the filename available
    post = instaloader.Post.from_shortcode(insta.context, shortcode)
    #make the post data directory if not exist
    csvName = shortcode + '.csv'
    pathlib.Path("post_data").mkdir(parents=True, exist_ok=True)
    output_path = pathlib.Path('post_data').absolute()
    post_file = output_path.joinpath(csvName)
    global scrapFilepath
    scrapFilepath = post_file

    field_names = [
        "post_shortcode",
        "commenter_username",
        "comment_text",
        "comment_likes"
    ]

    # open using a context manager so the file is closed automatically
    with post_file.open("w", encoding="utf-8") as fh:
        post_writer = csv.DictWriter(fh, fieldnames=field_names)
        post_writer.writeheader()

        ## get comments from post
        i = 0
        for x in post.get_comments():
            post_info = {
                "post_shortcode": post.shortcode,
                # x.owner is a Profile object; convert to username for CSV readability
                "commenter_username": getattr(x.owner, "username", str(x.owner)),
                "comment_text": (emoji.demojize(x.text)).encode('utf-8', errors='ignore').decode() if x.text else "",
                "comment_likes": x.likes_count
            }
            post_writer.writerow(post_info)
            i += 1
            if i == 80:
                break

    print("Done Scraping!")


def getPolarity(text, baseModel):
    """Return a polarity score for ``text``.

    If ``baseModel`` is True we always use :mod:`textblob`.  Otherwise we try to
    use the Transformers sentiment pipeline.  If the pipeline failed to import or
    initialize we log a warning once and fall back to textblob automatically.
    """
    # prefer textblob when requested or when transformers isn't available
    if baseModel or not _transformers_available:
        if not baseModel and not _transformers_available:
            # inform the user once that we fell back
            print("Note: transformers pipeline unavailable, using TextBlob instead")
        return textblob.TextBlob(text).sentiment.polarity

    # at this point we know sentiment_analyzer exists
    result = sentiment_analyzer(text)
    if result[0]["label"] == "NEGATIVE":
        return -1 * result[0]["score"]
    else:
        return result[0]["score"]


def readPrepData(filePath, showNegative=False, baseModel=True):
    df = pd.read_csv(filePath)
    df['comment_text'] = df['comment_text'].astype(str)
    df['text_polarity'] = df['comment_text'].apply(getPolarity, args=(baseModel,))
    if baseModel:
        df['sentiment'] = pd.cut(df['text_polarity'], [-1, -0.0000000001, 0.0000000001, 1],
                             labels=["Negative", "Neutral", "Positive"])
    else:
        df['sentiment'] = pd.cut(df['text_polarity'], [-1, -0.7, 0.7, 1],
                                 labels=["Negative", "Neutral", "Positive"])
    if showNegative:
        filter_df = df[df["sentiment"] == "Negative"]
        pd.set_option('display.max_columns', None)
        print(filter_df)
    return df


def makeGraph(df):
    if df.empty:
        print("DataFrame is empty, nothing to plot")
        return

    graph1 = df.groupby(['post_shortcode', 'sentiment']).count().reset_index()
    # select the first post shortcode we have; the table may contain multiple
    # posts if the input CSV was concatenated.  This mirrors the old behaviour
    # but avoids an index error when df is empty.
    first_shortcode = df['post_shortcode'].iloc[0]
    graph2 = graph1[graph1['post_shortcode'] == first_shortcode]

    colors = ["red", "blue", "green"]

    ## plot
    fig, ax = plt.subplots(ncols=1)

    for t, y, c in zip(graph2["sentiment"], graph2["comment_text"], colors):
        ax.plot([t, t], [0, y], color=c, marker="o", markevery=(1, 2))

    ## remove spines on right and top of plot
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_ylim(0, None)
    plt.title("Instagram Comment Sentiment", fontsize=15)
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=12)

    plt.show()

    text = " ".join(df['comment_text'])
    wordcloud = WordCloud(width=1400, height=1000, background_color='white').generate(text)

    # Display the WordCloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Hide the axes
    plt.show()


if __name__ == '__main__':
    # basic sanity checks before we start doing network I/O
    if username == "user_id" or password == "your password":
        print("ERROR: please edit project.py and provide your Instagram username/password")
        exit(1)

    insta = login(username, password)
    scrapData(insta, url)
    df = readPrepData(scrapFilepath, showNegative=True, baseModel=False)
    makeGraph(df)
    # df = readPrepData("post_data/DCH3zVQspkd.csv", showNegative=True, baseModel=True)
    # makeGraph(df)
