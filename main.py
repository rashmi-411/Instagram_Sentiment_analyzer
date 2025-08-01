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

username = "used_id"
password = "your password"
url = "https://www.instagram.com/p/DCrpHvPMd2x/?utm_source=ig_web_copy_link&igsh=MzRlODBiNWFlZA=="
scrapFilepath = ""
def login(username, password):
    try:
        insta = instaloader.Instaloader()
        file = os.path.abspath(f"{username}")
        insta.load_session_from_file(username, file)
        insta.context.username = username
        print("Logged in using session")
    except:
        insta = instaloader.Instaloader()
        insta.login(username, password)
        try:
            username = insta.test_login()
            if not username:
                raise ConnectionException()
        except ConnectionException:
            raise SystemExit("Some Issue with Login. Try again after some time")
        insta.context.username = username
        insta.save_session_to_file(f"{username}")
        print("Logged in using the Credentials")
    return insta


def scrapData(insta, url):
    shortcode = str(url[28:39])
    post = instaloader.Post.from_shortcode(insta.context, shortcode)
    csvName = shortcode + '.csv'
    pathlib.Path("post_data").mkdir(parents=True, exist_ok=True)
    output_path = pathlib.Path('post_data').absolute()
    post_file = output_path.joinpath(csvName)
    global scrapFilepath
    scrapFilepath = post_file
    post_file = post_file.open("w", encoding="utf-8")
    field_names = [
        "post_shortcode",
        "commenter_username",
        "comment_text",
        "comment_likes"
    ]

    post_writer = csv.DictWriter(post_file, fieldnames=field_names)
    post_writer.writeheader()

    ## get comments from post
    i = 0
    for x in post.get_comments():
        post_info = {
            "post_shortcode": post.shortcode,
            "commenter_username": x.owner,
            "comment_text": (emoji.demojize(x.text)).encode('utf-8', errors='ignore').decode() if x.text else "",
            "comment_likes": x.likes_count
        }
        post_writer.writerow(post_info)
        i += 1
        if i == 150:
            break

    print("Done Scraping!")


def getPolarity(text):
    return textblob.TextBlob(text).sentiment.polarity


def readPrepData(filePath, showNegative=False):
    df = pd.read_csv(filePath)
    df['comment_text'] = df['comment_text'].astype(str)
    df['text_polarity'] = df['comment_text'].apply(getPolarity)
    df['sentiment'] = pd.cut(df['text_polarity'], [-1, -0.0000000001, 0.0000000001, 1],
                             labels=["Negative", "Neutral", "Positive"])
    if showNegative:
        filter_df = df[df["sentiment"] == "Negative"]
        pd.set_option('display.max_columns', None)
        print(filter_df)
    return df


def makeGraph(df):
    graph1 = df.groupby(['post_shortcode', 'sentiment']).count().reset_index()
    graph2 = graph1[graph1['post_shortcode'] == df["post_shortcode"][0]]

    colors = colors = ["#FF0066", "gray", "#00FF00"]

    ## plot
    fig, (ax) = plt.subplots(ncols=1)

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
    insta = login(username, password)
    scrapData(insta, url)
    df = readPrepData(scrapFilepath, showNegative=True)
    makeGraph(df)
    # df = readPrepData("post_data/DCZzjW7SW6E.csv", showNegative=True)
    # makeGraph(df)


