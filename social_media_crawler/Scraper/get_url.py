from __future__ import annotations

from enum import Enum
from functools import partial
from typing import Union

import requests
import yt_dlp
from bs4 import BeautifulSoup
from loguru import logger

# 1. Get list of urls from a playlist on youtube

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36',
}


def youtube(yt_url: Union[str, list]):

    ydl = yt_dlp.YoutubeDL(
        {'outtmpl': '%(id)s%(ext)s', 'quiet': True, 'ignoreerrors': True},
    )

    # Create a dict with key/value pairs: tuple-list
    # tuple contains video url and number of comments
    # list contains topics of the video
    urls_dict = {}

    with ydl:

        if type(yt_url) == list:
            logger.info('Extracting info from list of video(s)...')

            for url in yt_url:
                result = ydl.extract_info(url, download=False)

                assert result[
                    'availability'
                ] == 'public', f'The video should be in `public` mode, but found {result["availability"]}'

                num_cmts = result['comment_count'] if result['comment_count'] is not None else 0

                if num_cmts == 0:
                    logger.warning(
                        f'Cannot extract the exact number of comments for '
                        f'video {result["title"]}',
                    )
                else:
                    logger.success(
                        f'Found {num_cmts} comments for video: {result["title"]}',
                    )

                urls_dict[(
                    result['webpage_url'],
                    num_cmts,
                )] = result['categories']
        else:
            logger.info('Extracting info from playlist...')
            result = ydl.extract_info(yt_url, download=False)

            assert result[
                'availability'
            ] == 'public', f'The playlist should be in `public` mode, but found {result["availability"]}'

            if result.get('entries') is not None:
                # Can be a playlist or a list of videos
                videos = result['entries']
                # loops entries to grab each video_url
                logger.info(f'Load {len(videos)} video(s) from playlist')
                for video in videos:
                    num_cmts = video['comment_count'] if video['comment_count'] is not None else 0

                    if num_cmts == 0:
                        logger.warning(
                            f'Cannot extract the exact number of comments for '
                            f'video {videos.index(video)+1}: {video["title"]}',
                        )
                    else:
                        logger.success(
                            f'Found {num_cmts} comments for '
                            f'video {videos.index(video)+1}: {video["title"]}',
                        )

                    urls_dict[(
                        video['webpage_url'],
                        num_cmts,
                    )] = video['categories']

    return urls_dict

# 2. Get list of urls from websites
# Tinhte.vn


def tinhte(webpage_url):
    # HTTP request
    response = requests.get(webpage_url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    output = []
    label_class_list = [
        # div tags
        ['div', 'jsx-2206250852 thread'],
        ['div', 'jsx-1673213166 item'],
        ['div', 'jsx-1462321135 slider tinhte'],
        ['div', 'jsx-1462321135 slider tinhte'], ['h3', 'searchResultTitle'],
        ['div', 'jsx-691990575 thread-comment__box'],
        ['div', 'jsx-3501530503 main  second'],
        ['div', 'jsx-3501530503 main'],
        ['div', 'jsx-2070971683 body'],
        ['div', 'jsx-4201240927 main reverse'],
        ['div', 'jsx-1462321135 slider tinhte'],
        # article tags
        ['article', 'jsx-2238569880'],
        ['artical', 'jsx-2206250852 item'],
        ['artical', 'jsx-810520461'],

        # a tags
        ['a', 'jsx-1560371842 thread'],

        # others
        ['h4', 'jsx-3501530503 thread-title'],
        ['span', 'xf-body-paragraph'],
    ]
    for label_class in label_class_list:
        try:
            titles = soup.findAll(label_class[0], class_=label_class[1])
            links = [link.find('a').attrs['href'] for link in titles]
            for link in links:
                if link.startswith('https://tinhte.vn/'):
                    output.append(link)
                else:
                    output.append('https://tinhte.vn/' + link)
            print()
        except Exception:
            pass
    return output

# Genk.vn


def genk(webpage_url):
    response = requests.get(webpage_url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    output = []
    label_class_list = [
        ['h3', ''], ['h2', ''], ['div', 'gknews_box gk_flx'],
        ['div', 'gknews_box'],
        ['h4', ''], ['h4', 'knswli-title'], [
            'li',
            'relate-news-li',
        ], ['div', 'total'],
    ]

    for label_class in label_class_list:
        try:
            titles = soup.findAll(label_class[0], class_=label_class[1])
            links = [link.find('a').attrs['href'] for link in titles]
            for link in links:
                if link.startswith('https://genk.vn/'):
                    output.append(link)
                else:
                    output.append('https://genk.vn/' + link)
        except Exception:
            pass
    return output

# Techz.vn


def techz(webpage_url):
    response = requests.get(webpage_url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    output = []
    label_class_list = [
        ['li', 'item'], ['div', 'item_slide_one slick-slide slick-current slick-active'], [
            'div',
            'item_slide_one slick-slide slick-active',
        ], ['a', 'thumb_box_news_right'],
        ['a', 'title_box_news_right'],
        ['div', 'item_slide_one slick-slide slick-cloned'],
        ['div', 'media-body'], ['div', 'card-body py-2'], ['div', 'card mb-2'], ]

    for label_class in label_class_list:
        try:
            titles = soup.findAll(label_class[0], class_=label_class[1])
            links = [link.find('a').attrs['href'] for link in titles]
            for link in links:
                if link.startswith('https://www.techz.vn/'):
                    output.append(link)
                else:
                    output.append('https://www.techz.vn/' + link)
        except Exception:
            pass
    return output

# VietnamNet.vn


def vietnamnet(webpage_url):
    response = requests.get(webpage_url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    output = []
    label_class_list = [
        ['h3', 'horizontalPost-content__title vnn-title'], [
            'h3',
            'verticalPost-content__title vnn-title',
        ],
        ['h3', 'slide-eventItem-content__title'], ['h3', 'title-box-photo'], [
            'h3',
            'horizontalRelatedPost-content__title vnn-title',
        ],
        ['h3', 'horizontalFeaturePost-content__title vnn-title'], ['h3', 'vnn-title'],
        ['h3', 'slide-horizontalItem-content__title'], [
            'h3',
            'subcate-bold__related-title vnn-title',
        ],
        ['h3', 'subcate-bold-s__title vnn-title'], [
            'h3',
            'subcate-bold-s__related-title vnn-title',
        ],
        ['h3', 'feature-box__content--title vnn-title'], ['h2', 'vnn-title'],

        # div tags
        ['div', 'verticalFeaturePost-wrapper'],
        ['div', 'verticalPost-wrapper'],
        ['div', 'horizontalPost-wrapper'],
        ['div', 'horizontalFeaturePost-wrapper'],
        ['div', 'horizontalRelatedPost-wrapper'],
        ['div', 'slideItem'],
        ['div', 'verticalPost-box'],
        ['div', 'feature-box'],
        ['div', 'verticalFeature-box'],
        ['div', 'verticalHighlight-box'],
        ['div', 'subcate-box'],
        ['div', 'subcate-box__suggest-news '],
        ['div', 'subcate-bold '],
        ['div', 'subcate-bold__content'],
        ['div', 'subcate-bold__related'],
        ['div', 'subcate-bold-s__related'],
        ['div', 'subcate-bold subcate-bold-s'],
        ['div', 'subcate-bold__related'],
        ['div', 'feature-box__content'],
        ['div', 'owned pb-15'],
    ]

    for label_class in label_class_list:
        try:
            titles = soup.findAll(label_class[0], class_=label_class[1])
            links = [link.find('a').attrs['href'] for link in titles]
            for link in links:
                if link.startswith('https://vietnamnet.vn'):
                    output.append(link)
                else:
                    output.append('https://vietnamnet.vn' + link)
        except Exception:
            pass
    return output

# 24h.vn


def haibongio(webpage_url):
    response = requests.get(webpage_url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    output = []
    label_class_list = [
        ['header', 'nwsTit cltitbxbdtt'], ['article', 'lstNml'],
        ['article', 'vdNml'], ['header', 'nwsTit titbxdoitrangchu'], [
            'article', 'lstRdStr',
        ],
        ['article', 'phtNml'], ['header', 'nwsTit titbxwhome'], [
            'article', 'icoInfo',
        ],
        ['h3', ''], ['p', 'cate-24h-car-news-rand__name margin-bottom-10'],
        ['header', 'nwsTit'], ['header', 'cate-24h-foot-box-adv-view-news__box--name'],
    ]

    for label_class in label_class_list:
        try:
            titles = soup.findAll(label_class[0], class_=label_class[1])
            links = [link.find('a').attrs['href'] for link in titles]
            for link in links:
                output.append(link)
        except Exception:
            pass
    return output

# TuoiTre.vn


def tuoitre(webpage_url):
    response = requests.get(webpage_url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    output = []
    label_class_list = [
        ['h2', 'title-focus-title'], ['h2', 'title-name'],
        ['article', 'vdNml'], ['header', 'nwsTit titbxdoitrangchu'], [
            'article', 'lstRdStr',
        ],
        ['article', 'phtNml'], ['header', 'nwsTit titbxwhome'], [
            'article', 'icoInfo',
        ], ['h3', 'title-news'],
        ['h3', ''], ['h3', 'title-name-newsvideo w156'], ['h3', 'name-title'],
    ]

    for label_class in label_class_list:
        try:
            titles = soup.findAll(label_class[0], class_=label_class[1])
            links = [link.find('a').attrs['href'] for link in titles]
            for link in links:
                if link.startswith('https://tuoitre.vn/'):
                    output.append(link)
                else:
                    output.append('https://tuoitre.vn/' + link)
        except Exception:
            pass
    return output

# VnExpress.vn


def vnexpress(webpage_url):
    response = requests.get(webpage_url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    output = []
    label_class_list = [['h3', 'title-news'], ['p', 'description']]

    for label_class in label_class_list:
        try:
            titles = soup.findAll(label_class[0], class_=label_class[1])
            links = [link.find('a').attrs['href'] for link in titles]
            for link in links:
                output.append(link)
        except Exception:
            pass
    return output

# Thanhnien.vn


def thanhnien(webpage_url):
    response = requests.get(webpage_url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    output = []
    label_class_list = [['h2', '']]

    for label_class in label_class_list:
        try:
            titles = soup.findAll(label_class[0], class_=label_class[1])
            links = [link.find('a').attrs['href'] for link in titles]
            for link in links:
                if link.startswith('https://thanhnien.vn/'):
                    output.append(link)
                else:
                    output.append('https://thanhnien.vn/' + link)
        except Exception:
            pass
    return output


class WebsiteType(Enum):
    tinhte = partial(tinhte)
    genk = partial(genk)
    techz = partial(techz)
    vietnamnet = partial(vietnamnet)
    haibongio = partial(haibongio)
    tuoitre = partial(tuoitre)
    vnexpress = partial(vnexpress)
    thanhnien = partial(thanhnien)
