from __future__ import annotations

from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from datetime import timezone
from typing import Any
from typing import List
from typing import Optional

from gnews import GNews
from GoogleNews import GoogleNews
from loguru import logger
from pydantic import PrivateAttr
from Scraper.base_source import BaseSource
from Scraper.base_source import BaseSourceConfig
from Scraper.payload import TextPayload
from Scraper.utils import convert_utc_time
from Scraper.website_crawler_source import BaseCrawlerConfig
from Scraper.website_crawler_source import TrafilaturaCrawlerConfig

GOOGLE_DATE_TIME_QUERY_PATTERN = '%Y-%m-%d'
GOOGLE_DATE_TIME_QUERY_PATTERNv2 = '%m-%d-%Y'


class GoogleNewsConfigV2(BaseSourceConfig):
    _google_news_client: GoogleNews = PrivateAttr()
    TYPE: str = 'GoogleNews'
    query: str
    country: Optional[str] = 'US'
    language: Optional[str] = 'en'
    max_pages: Optional[int] = 100
    period: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    fetch_article: Optional[bool] = True
    crawler_config: Optional[BaseCrawlerConfig] = None

    def __init__(self, **data: Any):
        super().__init__(**data)

        if self.period and self.start_date:
            raise AttributeError(
                "Can't use `period` and `start_date` both",
            )
        elif not self.start_date and self.end_date:
            raise AttributeError(
                "Can't use `end_date` without `start_date` or `period`",
            )

        if self.period:
            after_time = convert_utc_time(self.period)
            self.start_date = after_time.strftime(
                GOOGLE_DATE_TIME_QUERY_PATTERNv2,
            )

        if not self.end_date:
            before_time = datetime.combine(
                date.today(), time(
                    tzinfo=timezone.utc,
                ),
            ) + timedelta(days=1)
            self.end_date = before_time.strftime(
                GOOGLE_DATE_TIME_QUERY_PATTERNv2,
            )

        self._google_news_client = GoogleNews(
            lang=self.language,
            region=self.country,
            start=self.start_date,
            end=self.end_date,
            encode='utf-8',
        )

        self._google_news_client.enableException(True)

        if not self.crawler_config:
            self.crawler_config = TrafilaturaCrawlerConfig(urls=[])

    def get_client(self) -> GoogleNews:
        return self._google_news_client


class GoogleNewsConfig(BaseSourceConfig):
    _google_news_client: GNews = PrivateAttr()
    TYPE: str = 'GoogleNews'
    query: str
    country: Optional[str] = 'US'
    language: Optional[str] = 'en'
    max_results: Optional[int] = 100
    period: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    fetch_article: Optional[bool] = True
    crawler_config: Optional[BaseCrawlerConfig] = None

    def __init__(self, **data: Any):
        super().__init__(**data)

        if self.period and self.start_date:
            raise AttributeError(
                "Can't use `period` and `start_date` both",
            )
        elif not self.start_date and self.end_date:
            raise AttributeError(
                "Can't use `end_date` without `start_date` or `period`",
            )

        if self.period:
            after_time = convert_utc_time(self.period)
            self.start_date = after_time.strftime(
                GOOGLE_DATE_TIME_QUERY_PATTERN,
            )

        if not self.end_date:
            before_time = datetime.combine(
                date.today(), time(
                    tzinfo=timezone.utc,
                ),
            ) + timedelta(days=1)
            self.end_date = before_time.strftime(
                GOOGLE_DATE_TIME_QUERY_PATTERN,
            )

        self._google_news_client = GNews(
            language=self.language,
            country=self.country,
            max_results=self.max_results,
        )

        if not self.crawler_config:
            self.crawler_config = TrafilaturaCrawlerConfig(urls=[])

    def get_client(self) -> GNews:
        return self._google_news_client


class GoogleNewsSource(BaseSource):
    NAME: Optional[str] = 'GoogleNews'

    # type: ignore[override]
    def lookup(self, config: GoogleNewsConfigV2, **kwargs: Any) -> List[TextPayload]:
        source_responses: list[TextPayload] = []

        google_news_client = config.get_client()

        # articles = google_news_client.get_news(config.query)

        # google_news_client.get_news(config.query)
        google_news_client.search(config.query)
        logger.info('Crawled page 1.')

        try:
            for i in range(2, config.max_pages + 1):  # type: ignore[operator]
                google_news_client.get_page(i)
                logger.info(f'Crawled page {i}.')
        except BaseException:
            logger.error(
                f'Exceed available pages. Only has {i-1} pages available',
            )

        articles = google_news_client.results()

        for article in articles:

            if config.fetch_article and config.crawler_config:
                extracted_data = config.crawler_config.extract_url(
                    url=article['link'],
                )

                if extracted_data is not None and extracted_data.get('text', None) is not None:
                    article_text = extracted_data['text']
                    del extracted_data['text']
                else:
                    article_text = ''

                article['extracted_data'] = extracted_data
            else:
                article_text = article['description']

            source_responses.append(
                TextPayload(
                    processed_text=f"{article['title']}.\n\n {article_text}",
                    meta=vars(article) if hasattr(
                        article, '__dict__',
                    ) else article,
                    source_name=self.NAME,
                ),
            )

        return source_responses
