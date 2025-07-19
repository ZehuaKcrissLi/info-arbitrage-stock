from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
import uuid

class TradingAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    SELL_SHORT = "SELL_SHORT"
    HOLD = "HOLD"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    BRACKET = "bracket"

class OrderStatus(str, Enum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class DebateResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str
    debate_history: List[tuple] = Field(default_factory=list)
    final_recommendation: 'TradingRecommendation'
    confidence_score: float = Field(ge=0.0, le=1.0)
    consensus_level: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class TradingRecommendation(BaseModel):
    ticker: str
    action: TradingAction
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    timeframe: str = "intraday"  # intraday, swing, position
    target_price: Optional[float] = None
    stop_loss_price: Optional[float] = None

class TradeProposal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ticker: str
    action: TradingAction
    entry_price: float
    quantity: int
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    atr: float  # Average True Range for volatility calculation
    timeframe: str = "intraday"
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class RiskLevels(BaseModel):
    stop_loss: float
    take_profit: float
    risk_amount: float
    reward_amount: float
    risk_reward_ratio: float

class ValidationResult(BaseModel):
    valid: bool
    reason: Optional[str] = None
    suggested_quantity: Optional[int] = None

class RiskValidationResult(BaseModel):
    approved: bool
    adjusted_proposal: Optional[TradeProposal] = None
    risk_levels: Optional[RiskLevels] = None
    validation_details: Dict[str, ValidationResult] = Field(default_factory=dict)

class TradingSignal(BaseModel):
    signal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ticker: str
    action: TradingAction
    quantity: int
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    timeframe: str = "intraday"
    risk_reward_ratio: float
    risk_details: Optional[Dict[str, Any]] = None

class OrderRequest(BaseModel):
    symbol: str
    qty: int
    side: str  # buy, sell
    type: OrderType
    time_in_force: str = "day"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    order_class: Optional[str] = None  # simple, bracket, oco
    take_profit: Optional[Dict[str, float]] = None
    stop_loss: Optional[Dict[str, float]] = None

class TakeProfitRequest(BaseModel):
    limit_price: float

class StopLossRequest(BaseModel):
    stop_price: float
    limit_price: Optional[float] = None

class ExecutionResult(BaseModel):
    success: bool
    order_id: Optional[str] = None
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None
    reason: Optional[str] = None
    tracked_order: Optional[Dict[str, Any]] = None

class Position(BaseModel):
    symbol: str
    qty: int
    side: str  # long, short
    market_value: float
    cost_basis: float
    unrealized_pl: float
    unrealized_plpc: float
    current_price: float

class AccountStatus(BaseModel):
    total_equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    day_trade_count: int
    positions: List[Position] = Field(default_factory=list)
    daily_pnl: float = 0.0
    trades_today: int = 0

class Trade(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    signal_id: str
    ticker: str
    action: TradingAction
    quantity: int
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    stop_loss: float
    take_profit: float
    status: str = "open"  # open, closed
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    exit_reason: Optional[str] = None  # stop_loss, take_profit, manual

class PerformanceMetrics(BaseModel):
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    profit_factor: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0