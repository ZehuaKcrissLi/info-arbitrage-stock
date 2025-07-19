from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
import uuid

class EventType(str, Enum):
    EARNINGS = "earnings"
    NEWS = "news"
    MERGER = "merger"
    FDA_APPROVAL = "fda_approval"
    ANALYST_UPGRADE = "analyst_upgrade"
    ANALYST_DOWNGRADE = "analyst_downgrade"
    SEC_FILING = "sec_filing"
    MACRO_ECONOMIC = "macro_economic"
    SOCIAL_MEDIA = "social_media"

class MarketEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str
    tickers: List[str]
    event_type: EventType
    headline: str
    summary: str
    sentiment: float = Field(ge=-1.0, le=1.0)  # -1 to 1
    urgency: int = Field(ge=1, le=10)  # 1-10 scale
    raw_data: Dict[str, Any] = Field(default_factory=dict)

class EnhancedMarketEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_event: MarketEvent
    historical_context: str
    web_context: str
    knowledge_context: str
    enhanced_summary: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    processing_timestamp: datetime = Field(default_factory=datetime.utcnow)

class HistoricalEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime
    tickers: List[str]
    event_type: str
    description: str
    market_impact: float = Field(ge=-1.0, le=1.0)  # -1 to 1
    price_change_1h: float
    price_change_1d: float
    volume_ratio: float
    embedding: List[float] = Field(default_factory=list)

class NewsItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str
    timestamp: datetime
    headline: str
    content: str
    url: Optional[str] = None
    tickers: List[str] = Field(default_factory=list)
    sentiment: Optional[float] = None
    relevance_score: Optional[float] = None

class SocialMediaMention(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    platform: str  # twitter, reddit, etc.
    timestamp: datetime
    text: str
    author: str
    ticker: str
    sentiment: Optional[float] = None
    engagement_score: Optional[float] = None
    url: Optional[str] = None