from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
import uuid

class AnalysisType(str, Enum):
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    MACRO = "macro"

class FundamentalMetrics(BaseModel):
    pe_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    roe: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    free_cash_flow_yield: Optional[float] = None
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None

class TechnicalIndicators(BaseModel):
    # Trend indicators
    sma_20: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    
    # Momentum indicators
    rsi: Optional[float] = None
    macd: Optional[float] = None
    stoch: Optional[float] = None
    
    # Volatility indicators
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    atr: Optional[float] = None
    
    # Volume indicators
    volume_sma: Optional[float] = None
    obv: Optional[float] = None
    
    # Key levels
    vwap: Optional[float] = None
    pivot_points: Dict[str, float] = Field(default_factory=dict)

class SupportResistanceLevels(BaseModel):
    support_levels: List[float] = Field(default_factory=list)
    resistance_levels: List[float] = Field(default_factory=list)
    current_price: float
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None

class TechnicalSignals(BaseModel):
    trend_direction: str  # bullish, bearish, neutral
    momentum_signal: str  # strong_buy, buy, neutral, sell, strong_sell
    volume_confirmation: bool
    breakout_signal: Optional[str] = None
    pattern_detected: Optional[str] = None

class SocialSentimentData(BaseModel):
    twitter_sentiment: float = 0.0
    reddit_sentiment: float = 0.0
    twitter_volume: int = 0
    reddit_volume: int = 0
    trending_hashtags: List[str] = Field(default_factory=list)

class MacroIndicators(BaseModel):
    vix: Optional[float] = None
    treasury_10y: Optional[float] = None
    treasury_2y: Optional[float] = None
    dxy: Optional[float] = None
    oil_price: Optional[float] = None
    gold_price: Optional[float] = None
    spy_performance: Optional[float] = None
    qqq_performance: Optional[float] = None
    economic_calendar: List[Dict[str, Any]] = Field(default_factory=list)

class TickerFundamentalAnalysis(BaseModel):
    ticker: str
    metrics: FundamentalMetrics
    valuation: str  # undervalued, fair_value, overvalued
    narrative: str
    confidence: float = Field(ge=0.0, le=1.0)
    score: float = Field(ge=-1.0, le=1.0)  # -1 bearish to +1 bullish

class TickerTechnicalAnalysis(BaseModel):
    ticker: str
    indicators: TechnicalIndicators
    patterns: List[str] = Field(default_factory=list)
    support_resistance: SupportResistanceLevels
    signals: TechnicalSignals
    narrative: str
    confidence: float = Field(ge=0.0, le=1.0)
    score: float = Field(ge=-1.0, le=1.0)  # -1 bearish to +1 bullish

class FundamentalAnalysis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_id: str
    ticker_analyses: List[TickerFundamentalAnalysis]
    overall_sentiment: float = Field(ge=-1.0, le=1.0)

class TechnicalAnalysis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_id: str
    ticker_analyses: List[TickerTechnicalAnalysis]
    overall_direction: str  # bullish, bearish, neutral

class SentimentAnalysis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_id: str
    news_sentiment: float = Field(ge=-1.0, le=1.0)
    social_sentiment: Dict[str, SocialSentimentData] = Field(default_factory=dict)
    options_sentiment: Optional[float] = None
    aggregate_sentiment: float = Field(ge=-1.0, le=1.0)
    narrative: str
    confidence: float = Field(ge=0.0, le=1.0)

class MacroAnalysis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_id: str
    macro_indicators: MacroIndicators
    market_structure: Dict[str, Any] = Field(default_factory=dict)
    sector_analysis: Dict[str, float] = Field(default_factory=dict)
    narrative: str
    risk_assessment: str  # low, medium, high

class AnalysisBundle(BaseModel):
    event_id: str
    fundamental: Optional[FundamentalAnalysis] = None
    technical: Optional[TechnicalAnalysis] = None
    sentiment: Optional[SentimentAnalysis] = None
    macro: Optional[MacroAnalysis] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)