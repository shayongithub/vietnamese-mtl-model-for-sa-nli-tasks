from __future__ import annotations

from abc import abstractmethod
from typing import Any
from typing import Optional

from pydantic import BaseSettings
from Scraper.base_store import BaseStore
from Scraper.payload import TextPayload


class BaseSourceConfig(BaseSettings):
    TYPE: str = 'Base'

    class Config:
        arbitrary_types_allowed = True


class BaseSource(BaseSettings):
    store: Optional[BaseStore] = None

    @abstractmethod
    def lookup(self, config: BaseSourceConfig, **kwargs: Any) -> list[TextPayload]:
        pass

    class Config:
        arbitrary_types_allowed = True
