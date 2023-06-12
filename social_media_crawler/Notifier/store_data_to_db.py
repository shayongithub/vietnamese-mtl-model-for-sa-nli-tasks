from __future__ import annotations

import sys
import time
from enum import Enum
from functools import partial
from typing import Optional

import pandas as pd
import psycopg2
from config import settings
from loguru import logger
from Scraper.utils import increment_path
from sqlalchemy import create_engine

sys.path.append('.')


class store_data_to_db:
    def __init__(
        self, name=None, webpage_name=None, url=None, max_count=None,
        lookup_period=None, country=None, lang=None, query=None,
    ):
        self.name = name
        self.webpage_name = webpage_name
        self.url = url
        self.max_count = max_count
        self.lookup_period = lookup_period
        self.country = country
        self.lang = lang
        self.query = query

    def store_to_csv(data: pd.DataFrame, output_path: Optional[str] = 'output/crawled_data.csv'):
        if data.empty:
            logger.error('Crawl 0 rows. No data to store')
            return

        if output_path is None:
            output_path = 'output/crawled_data.csv'

        filepath = increment_path(output_path)

        filepath.parent.mkdir(exist_ok=True, parents=True)

        data.to_csv(filepath, mode='w', index=False, encoding='utf-8-sig')

        logger.success(
            f'Save data as CSV file successfully at: {filepath.resolve()}',
        )

        return filepath

    def store_to_json(data: pd.DataFrame, output_path: Optional[str] = 'output/crawled_data.json'):
        if data.empty:
            logger.error('Crawl 0 rows. No data to store')
            return

        if output_path is None:
            output_path = 'output/crawled_data.json'

        filepath = increment_path(output_path)

        filepath.parent.mkdir(exist_ok=True, parents=True)

        data.to_json(
            filepath, orient='index',
        )

        logger.success(
            f'Save data as JSON file successfully at: {filepath.resolve()}',
        )

        return filepath

    def store_to_db(data: pd.DataFrame, table: Optional[str] = 'crawled_data'):

        if data.empty:
            logger.error('Crawl 0 rows. No data to store')
            return

        try:
            logger.info('Connecting to database')
            # "postgresql://YourUserName:YourPassword@YourHostname:5432/YourDatabaseName"
            conn_string = (
                f'postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}'
                f'@{settings.POSTGRES_HOSTNAME}:{settings.DATABASE_PORT}/{settings.POSTGRES_DB}'
            )
            db = create_engine(conn_string)
            conn = db.connect()
            logger.success(f'Databse "{settings.POSTGRES_DB}" connected')
        except Exception:
            raise ValueError(
                'Can not connect to databse. Check your conn_string value!',
            )

        print('-' * 30)
        logger.info('Showing crawled data')
        print(data)

        data.to_sql(table, con=conn, if_exists='append', index=False)
        conn = psycopg2.connect(
            conn_string,
        )

        logger.success(f'Upload data successfully to table: {table}')

        conn.autocommit = True
        cursor = conn.cursor()

        sql1 = f'''select * from {table};'''
        cursor.execute(sql1)
        before_rows = cursor.rowcount

        # Delete duplicates
        logger.info(f'Rows count: {before_rows}')

        sql2 = f'''
        DELETE FROM {table} T1
            USING   {table} T2
        WHERE   T1.ctid < T2.ctid
            AND T1.id  = T2.id;
        '''
        cursor.execute(sql2)
        # for i in cursor.fetchall():
        #     print(i)
        del_rows = cursor.rowcount

        if del_rows < before_rows and del_rows != 0:
            logger.warning('Detect duplicates')
            time.sleep(1)
            logger.success(
                f'Deleted duplicates successfully. Deleted rows count: {del_rows}',
            )
        else:
            cursor.execute(sql1)
            logger.success(
                f'No duplicates found. Rows count: {cursor.rowcount}',
            )

        conn.commit()
        conn.close()


class OutputDataType(Enum):
    csv = partial(store_data_to_db.store_to_csv)
    json = partial(store_data_to_db.store_to_json)
    postgreSql = partial(store_data_to_db.store_to_db)
