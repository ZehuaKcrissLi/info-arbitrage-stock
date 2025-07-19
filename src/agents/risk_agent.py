import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import asyncio

from ..models.trading import (
    TradeProposal, RiskValidationResult, ValidationResult, RiskLevels,
    AccountStatus, Position, TradingAction
)
from ..config.settings import settings

logger = logging.getLogger(__name__)

class RiskManagementAgent:
    """Agent responsible for risk management and trade validation"""
    
    def __init__(self):
        self.position_limits = settings.risk.max_position_pct
        self.max_total_exposure = settings.risk.max_total_exposure
        self.max_sector_exposure = settings.risk.max_sector_exposure
        self.max_loss_pct = settings.risk.max_loss_pct
        self.risk_reward_ratio = settings.risk.risk_reward_ratio
        self.atr_stop_multiplier = settings.risk.atr_stop_multiplier
        self.max_daily_loss = settings.risk.max_daily_loss
        self.max_consecutive_losses = settings.risk.max_consecutive_losses
        self.cooldown_period = settings.risk.cooldown_period
        
        self.account_monitor = AccountMonitor()
        self.circuit_breaker = CircuitBreaker()
        
    async def start(self):
        """Initialize the risk management agent"""
        await self.account_monitor.start()
        await self.circuit_breaker.start()
        logger.info("Risk Management Agent started")
    
    async def stop(self):
        """Stop the risk management agent"""
        await self.account_monitor.stop()
        await self.circuit_breaker.stop()
        logger.info("Risk Management Agent stopped")
    
    async def validate_trade(self, trade_proposal: TradeProposal) -> RiskValidationResult:
        """Validate proposed trade against risk parameters"""
        
        try:
            # Check account status
            account_status = await self.account_monitor.get_current_status()
            
            # Validate position size
            position_validation = self._validate_position_size(trade_proposal, account_status)
            
            # Check daily limits
            daily_validation = self._validate_daily_limits(trade_proposal, account_status)
            
            # Assess concentration risk
            concentration_validation = self._validate_concentration(trade_proposal, account_status)
            
            # Check circuit breaker status
            circuit_breaker_status = self.circuit_breaker.check_status()
            
            # Validate market conditions
            market_validation = self._validate_market_conditions(trade_proposal)
            
            # Calculate risk-adjusted position size
            adjusted_proposal = self._adjust_position_size(
                trade_proposal, account_status, position_validation
            )
            
            # Generate stop loss and take profit levels
            risk_levels = self._calculate_risk_levels(adjusted_proposal)
            
            # Determine if trade is approved
            all_validations = [
                position_validation.valid,
                daily_validation.valid,
                concentration_validation.valid,
                circuit_breaker_status,
                market_validation.valid
            ]
            
            approved = all(all_validations)
            
            return RiskValidationResult(
                approved=approved,
                adjusted_proposal=adjusted_proposal if approved else None,
                risk_levels=risk_levels if approved else None,
                validation_details={
                    'position': position_validation,
                    'daily_limits': daily_validation,
                    'concentration': concentration_validation,
                    'circuit_breaker': ValidationResult(valid=circuit_breaker_status, reason="Circuit breaker check"),
                    'market_conditions': market_validation
                }
            )
            
        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return RiskValidationResult(
                approved=False,
                adjusted_proposal=None,
                risk_levels=None,
                validation_details={
                    'error': ValidationResult(valid=False, reason=f"Validation error: {str(e)}")
                }
            )
    
    def _validate_position_size(self, proposal: TradeProposal, 
                              account: AccountStatus) -> ValidationResult:
        """Validate position size against limits"""
        
        if account.total_equity <= 0:
            return ValidationResult(
                valid=False,
                reason="Invalid account equity"
            )
        
        max_position_value = account.total_equity * self.position_limits
        proposed_value = proposal.quantity * proposal.entry_price
        
        if proposed_value > max_position_value:
            suggested_quantity = int(max_position_value / proposal.entry_price)
            return ValidationResult(
                valid=False,
                reason=f"Position size ${proposed_value:,.2f} exceeds limit ${max_position_value:,.2f}",
                suggested_quantity=max(1, suggested_quantity)  # At least 1 share
            )
        
        # Check minimum position size
        if proposed_value < 100:  # Minimum $100 position
            return ValidationResult(
                valid=False,
                reason="Position size too small (minimum $100)"
            )
        
        return ValidationResult(valid=True)
    
    def _validate_daily_limits(self, proposal: TradeProposal, 
                             account: AccountStatus) -> ValidationResult:
        """Check daily trading limits"""
        
        # Check maximum daily trades
        if account.trades_today >= settings.trading.max_daily_trades:
            return ValidationResult(
                valid=False,
                reason=f"Daily trade limit reached ({account.trades_today}/{settings.trading.max_daily_trades})"
            )
        
        # Check daily P&L limits
        if account.daily_pnl <= -self.max_daily_loss * account.total_equity:
            return ValidationResult(
                valid=False,
                reason=f"Daily loss limit reached (${account.daily_pnl:,.2f})"
            )
        
        return ValidationResult(valid=True)
    
    def _validate_concentration(self, proposal: TradeProposal, 
                              account: AccountStatus) -> ValidationResult:
        """Assess concentration risk"""
        
        # Check total exposure
        current_exposure = sum(abs(pos.market_value) for pos in account.positions)
        proposed_exposure = proposal.quantity * proposal.entry_price
        total_exposure = current_exposure + proposed_exposure
        
        max_total_value = account.total_equity * self.max_total_exposure
        
        if total_exposure > max_total_value:
            return ValidationResult(
                valid=False,
                reason=f"Total exposure ${total_exposure:,.2f} exceeds limit ${max_total_value:,.2f}"
            )
        
        # Check position concentration in same ticker
        existing_position = next(
            (pos for pos in account.positions if pos.symbol == proposal.ticker), 
            None
        )
        
        if existing_position:
            combined_value = abs(existing_position.market_value) + proposed_exposure
            max_single_position = account.total_equity * self.position_limits * 2  # Allow 2x for additions
            
            if combined_value > max_single_position:
                return ValidationResult(
                    valid=False,
                    reason=f"Combined position in {proposal.ticker} would exceed concentration limit"
                )
        
        # Check sector concentration (simplified)
        sector_exposure = self._calculate_sector_exposure(proposal.ticker, account.positions)
        sector_exposure += proposed_exposure
        max_sector_value = account.total_equity * self.max_sector_exposure
        
        if sector_exposure > max_sector_value:
            return ValidationResult(
                valid=False,
                reason=f"Sector exposure would exceed limit"
            )
        
        return ValidationResult(valid=True)
    
    def _calculate_sector_exposure(self, ticker: str, positions: list) -> float:
        """Calculate current exposure to ticker's sector"""
        # Simplified sector mapping
        sector_map = {
            'AAPL': 'tech', 'MSFT': 'tech', 'GOOGL': 'tech', 'AMZN': 'tech',
            'TSLA': 'auto', 'JPM': 'finance', 'BAC': 'finance',
            'JNJ': 'healthcare', 'PFE': 'healthcare'
        }
        
        ticker_sector = sector_map.get(ticker, 'other')
        
        sector_exposure = 0.0
        for position in positions:
            position_sector = sector_map.get(position.symbol, 'other')
            if position_sector == ticker_sector:
                sector_exposure += abs(position.market_value)
        
        return sector_exposure
    
    def _validate_market_conditions(self, proposal: TradeProposal) -> ValidationResult:
        """Validate market conditions for trading"""
        
        # Check if it's market hours (simplified check)
        now = datetime.now()
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Allow trading only during market hours
        if not (market_open <= now <= market_close):
            return ValidationResult(
                valid=False,
                reason="Trading outside market hours"
            )
        
        # Check for weekend
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return ValidationResult(
                valid=False,
                reason="No trading on weekends"
            )
        
        return ValidationResult(valid=True)
    
    def _adjust_position_size(self, proposal: TradeProposal, account: AccountStatus,
                            position_validation: ValidationResult) -> TradeProposal:
        """Adjust position size based on validation results"""
        
        adjusted_proposal = TradeProposal(
            ticker=proposal.ticker,
            action=proposal.action,
            entry_price=proposal.entry_price,
            quantity=proposal.quantity,
            rationale=proposal.rationale,
            confidence=proposal.confidence,
            atr=proposal.atr,
            timeframe=proposal.timeframe
        )
        
        # Use suggested quantity if position size was invalid
        if not position_validation.valid and position_validation.suggested_quantity:
            adjusted_proposal.quantity = position_validation.suggested_quantity
            logger.info(f"Adjusted position size to {adjusted_proposal.quantity} shares")
        
        # Apply confidence-based sizing
        confidence_multiplier = min(1.0, proposal.confidence * 1.2)  # Max 120% of base size
        adjusted_proposal.quantity = int(adjusted_proposal.quantity * confidence_multiplier)
        adjusted_proposal.quantity = max(1, adjusted_proposal.quantity)  # At least 1 share
        
        return adjusted_proposal
    
    def _calculate_risk_levels(self, proposal: TradeProposal) -> RiskLevels:
        """Calculate stop loss and take profit levels"""
        
        # Base stop loss on ATR and percentage
        atr_stop = proposal.entry_price * (1 - self.atr_stop_multiplier * (proposal.atr / proposal.entry_price))
        percentage_stop = proposal.entry_price * (1 - self.max_loss_pct)
        
        # Use the less aggressive stop (higher price for long positions)
        if proposal.action in [TradingAction.BUY]:
            stop_loss = max(atr_stop, percentage_stop)
        else:  # For short positions
            stop_loss = min(atr_stop, percentage_stop)
        
        # Calculate take profit based on risk/reward ratio
        risk_amount = abs(proposal.entry_price - stop_loss)
        
        if proposal.action in [TradingAction.BUY]:
            take_profit = proposal.entry_price + (risk_amount * self.risk_reward_ratio)
        else:  # For short positions
            take_profit = proposal.entry_price - (risk_amount * self.risk_reward_ratio)
        
        reward_amount = abs(take_profit - proposal.entry_price)
        
        return RiskLevels(
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_amount=risk_amount,
            reward_amount=reward_amount,
            risk_reward_ratio=self.risk_reward_ratio
        )

class AccountMonitor:
    """Monitors account status and position information"""
    
    def __init__(self):
        self.is_running = False
        
    async def start(self):
        """Start account monitoring"""
        self.is_running = True
        logger.info("Account Monitor started")
    
    async def stop(self):
        """Stop account monitoring"""
        self.is_running = False
        logger.info("Account Monitor stopped")
    
    async def get_current_status(self) -> AccountStatus:
        """Get current account status"""
        # This would integrate with broker API in production
        # For now, return mock data
        
        return AccountStatus(
            total_equity=100000.0,
            cash=50000.0,
            buying_power=80000.0,
            portfolio_value=50000.0,
            day_trade_count=0,
            positions=[],
            daily_pnl=0.0,
            trades_today=0
        )

class CircuitBreaker:
    """Implements circuit breaker functionality"""
    
    def __init__(self):
        self.consecutive_losses = 0
        self.last_loss_time = None
        self.is_triggered = False
        self.trigger_time = None
        
    async def start(self):
        """Start circuit breaker"""
        logger.info("Circuit Breaker started")
    
    async def stop(self):
        """Stop circuit breaker"""
        logger.info("Circuit Breaker stopped")
    
    def check_status(self) -> bool:
        """Check if circuit breaker allows trading"""
        
        # Check if breaker is triggered and still in cooldown
        if self.is_triggered and self.trigger_time:
            cooldown_end = self.trigger_time + timedelta(seconds=settings.risk.cooldown_period)
            if datetime.utcnow() < cooldown_end:
                return False
            else:
                # Reset circuit breaker
                self.reset()
        
        return not self.is_triggered
    
    def record_loss(self):
        """Record a losing trade"""
        self.consecutive_losses += 1
        self.last_loss_time = datetime.utcnow()
        
        logger.info(f"Recorded loss #{self.consecutive_losses}")
        
        # Check if we should trigger circuit breaker
        if self.consecutive_losses >= settings.risk.max_consecutive_losses:
            self.trigger()
    
    def record_win(self):
        """Record a winning trade"""
        if self.consecutive_losses > 0:
            logger.info(f"Win recorded, resetting consecutive losses from {self.consecutive_losses}")
        self.consecutive_losses = 0
        self.last_loss_time = None
    
    def trigger(self):
        """Trigger circuit breaker"""
        self.is_triggered = True
        self.trigger_time = datetime.utcnow()
        logger.warning(f"Circuit breaker TRIGGERED after {self.consecutive_losses} consecutive losses")
    
    def reset(self):
        """Reset circuit breaker"""
        self.is_triggered = False
        self.trigger_time = None
        self.consecutive_losses = 0
        self.last_loss_time = None
        logger.info("Circuit breaker RESET")
    
    def force_reset(self):
        """Force reset circuit breaker (manual override)"""
        self.reset()
        logger.warning("Circuit breaker FORCE RESET (manual override)")
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed circuit breaker status"""
        status = {
            'is_triggered': self.is_triggered,
            'consecutive_losses': self.consecutive_losses,
            'trigger_time': self.trigger_time.isoformat() if self.trigger_time else None,
            'last_loss_time': self.last_loss_time.isoformat() if self.last_loss_time else None
        }
        
        if self.is_triggered and self.trigger_time:
            cooldown_end = self.trigger_time + timedelta(seconds=settings.risk.cooldown_period)
            status['cooldown_ends'] = cooldown_end.isoformat()
            status['time_remaining'] = max(0, (cooldown_end - datetime.utcnow()).total_seconds())
        
        return status