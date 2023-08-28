from __future__ import annotations

from enum import Enum
from typing import Optional

import uvicorn
from fastapi import FastAPI
from loguru import logger
from Notifier.store_data_to_db import OutputDataType
from Notifier.store_data_to_db import store_data_to_db
from Scraper.get_comment import *
from Scraper.get_url import *
from Scraper.utils import convert_to_json

app = FastAPI()


class WebpageName(str, Enum):
    tinhte = 'tinhte'
    genk = 'genk'
    techz = 'techz'
    vietnamnet = 'vietnamnet'
    haibongio = 'haibongio'
    tuoitre = 'tuoitre'
    vnexpress = 'vnexpress'
    thanhnien = 'thanhnien'
    list_of_url = 'list_of_url'


class Playlist_or_list(str, Enum):
    playlist = 'playlist'
    urls = 'list_of_urls'


class YoutubeFetchReplies(str, Enum):
    yt_fetch_replies_true = 'True'
    yt_fetch_replies_false = 'False'


class SaveType(str, Enum):
    csv = 'csv'
    postgresql = 'postgreSql'
    json_file = 'json'


@app.get('/')
def home():
    return 'Congratulations! Your API is working as expected.'


@app.get('/Website')
def Website(
    webpage_name: WebpageName,
    save_type: SaveType,
    url: str,
    topics: Optional[str] = None,
    save_path: Optional[str] = None,
):

    if webpage_name != 'list_of_url':
        urls = WebsiteType[webpage_name].value(url)
    else:
        url = url.replace(' ', '')
        urls = list(url.split(','))

    data = get_website_comment(urls, topics)

    logger.success(f'Total crawled rows: {data.shape[0]}')

    if save_type != 'postgreSql':
        OutputDataType[save_type].value(data, save_path)
    else:
        store_data_to_db.store_to_db(data, 'website')

    return convert_to_json(data)


@app.get('/Youtube')
def Youtube(
    playlist_or_List_of_url: Playlist_or_list,
    url: str,
    save_type: SaveType,
    fetch_replies: YoutubeFetchReplies,
    max_count: Optional[int] = 100,
    lookup_period: Optional[str] = '5Y',
    save_path: Optional[str] = None,
):

    if playlist_or_List_of_url == 'playlist':
        urls_dict = youtube(url)
    else:
        urls = url.replace(' ', '').split(',')
        urls_dict = youtube(urls)

    data = get_youtube_comment(
        urls_dict, max_count, lookup_period, bool(fetch_replies),
    )

    logger.success(f'Total crawled rows: {data.shape[0]}')

    if save_type != 'postgreSql':
        OutputDataType[save_type].value(data, save_path)
    else:
        store_data_to_db.store_to_db(data, 'youtube')

    return convert_to_json(data)


@app.get('/Google_News')
def News(
    query: str,
    save_type: SaveType,
    max_pages: Optional[int] = 30,
    lookup_period: Optional[str] = '5Y',
    country: Optional[str] = 'VN',
    lang: Optional[str] = 'vi',
    save_path: Optional[str] = None,
):

    data = get_google_news(query, max_pages, lookup_period, lang, country)
    logger.success(f'Total crawled rows: {data.shape[0]}')

    if save_type != 'postgreSql':
        OutputDataType[save_type].value(data, save_path)
    else:
        store_data_to_db.store_to_db(data, 'ggnews')

    return convert_to_json(data)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
