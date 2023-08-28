from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger
from Scraper.google_news_source import GoogleNewsConfigV2
from Scraper.google_news_source import GoogleNewsSource
from Scraper.website_crawler_source import TrafilaturaCrawlerConfig
from Scraper.website_crawler_source import TrafilaturaCrawlerSource
from Scraper.youtube_scrapper import YoutubeScrapperConfig
from Scraper.youtube_scrapper import YoutubeScrapperSource


# Get google news comment
def get_google_news(query, max_pages, lookup_period, language, country):
    try:
        userName = []
        raw_text = []
        rating = []
        article_id = []
        title = []
        platform = []
        topic = []

        source_config_v2 = GoogleNewsConfigV2(
            query=query,
            max_pages=max_pages,
            period=lookup_period,
            language=language,
            country=country,
        )

        source = GoogleNewsSource()
        ggnews_payload = source.lookup(source_config_v2)

        for response_payload in ggnews_payload:
            response_payload_meta = dict(response_payload)['meta']

            if response_payload_meta['extracted_data'] is None:
                continue

            title.append(response_payload_meta['title'])
            platform.append(response_payload_meta['link'])
            rating.append(None)

            userName.append(response_payload_meta['extracted_data']['author'])
            article_id.append(response_payload_meta['extracted_data']['id'])

            # Add the short extract of article to the content
            excerpt_text = response_payload_meta['extracted_data']['excerpt']
            if not excerpt_text:
                excerpt_text = ''

            raw_text.append(
                excerpt_text +
                response_payload_meta['extracted_data']['raw_text'],
            )

            # Use the default category of the webste if available
            web_category = response_payload_meta['extracted_data']['categories']
            topic.append(web_category if web_category else query)

        df = pd.DataFrame(
            list(zip(article_id, userName, title, raw_text, rating, platform, topic)), columns=[
                'id', 'author', 'title', 'content', 'rating', 'source', 'topic',
            ],
        )
    except Exception:
        raise ValueError(
            'Your configuration is not valid!',
        )
    return df

# Get youtube comment


def get_youtube_comment(urls_dict: dict, max_count: Optional[int], lookup_period: Optional[str], fetch_replies: bool):

    try:
        text = []
        author = []
        yt_id = []
        title = []  # type: ignore[var-annotated]
        rating = []  # type: ignore[var-annotated]
        platform = []
        topics_df = []
        for url_tuple, topics in urls_dict.items():
            try:

                if url_tuple[1] == 0:
                    logger.info(
                        'Crawl up to max count as exact number of comments cannot be found',
                    )
                else:
                    poss_cmts = max_count if url_tuple[1] > max_count else url_tuple[1]

                    logger.info(
                        f'Based on max count: {max_count}, crawl a total of {poss_cmts}. '
                        f'Comments left: {0 if url_tuple[1] < max_count else abs(max_count- url_tuple[1]) }',
                    )

                ytb_config = YoutubeScrapperConfig(
                    video_url=url_tuple[0],
                    max_comments=max_count,
                    lookup_period=lookup_period,
                    fetch_replies=fetch_replies,
                )
                ytb = YoutubeScrapperSource()
                ytb_response_list = ytb.lookup(ytb_config)

                for response_payload in ytb_response_list:
                    # convert to dict
                    response = dict(response_payload)['meta']
                    text.append(response['text'])
                    author.append(response['author'])
                    yt_id.append(response['comment_id'])
                    title.append(None)  # type: ignore[union-attr]
                    rating.append(None)  # type: ignore[union-attr]
                    platform.append('Youtube')
                    topics_df.append(topics)
            except Exception:
                pass

        df = pd.DataFrame(
            list(zip(yt_id, author, title, text, rating, platform, topics_df)), columns=[
                'id', 'author', 'title', 'content', 'rating', 'source', 'topic',
            ],
        )
    except Exception:
        raise ValueError('Your configuration is not valid!')

    return df

# Get website comment


def get_website_comment(urls, topics):

    source_config = TrafilaturaCrawlerConfig(urls=urls)
    source = TrafilaturaCrawlerSource()
    source_response_list = source.lookup(source_config)

    id = []
    hostname = []
    title = []
    raw_text = []
    rating = []
    platform = []
    topic = []

    for i in range(0, len(source_response_list)):
        source_response_list[i] = dict(source_response_list[i])
        id.append(source_response_list[i]['meta']['id'])
        hostname.append(source_response_list[i]['meta']['author'])
        title.append(source_response_list[i]['meta']['title'])
        raw_text.append(source_response_list[i]['meta']['raw_text'])
        rating.append(None)
        platform.append(source_response_list[i]['meta']['source'])
        topic.append(topics)
    df = pd.DataFrame(
        list(zip(id, hostname, title, raw_text, rating, platform, topic)), columns=[
            'id', 'author', 'title', 'content', 'rating', 'source', 'topic',
        ],
    )

    return df
