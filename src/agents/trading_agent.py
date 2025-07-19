import logging
from datetime import datetime
from typing import Optional

from ..models.trading import (
    TradingSignal, TradeProposal, DebateResult, TradingAction,
    TradingRecommendation
)
from ..models.analysis import AnalysisBundle
from ..agents.risk_agent import RiskManagementAgent
from ..config.settings import settings

logger = logging.getLogger(__name__)

class TradingAgent:
    """Main trading agent that synthesizes analysis and generates trading signals"""
    
    def __init__(self):
        self.risk_manager = RiskManagementAgent()
        
    async def start(self):
        """Initialize the trading agent"""
        await self.risk_manager.start()
        logger.info("Trading Agent started")
    
    async def stop(self):
        """Stop the trading agent"""
        await self.risk_manager.stop()
        logger.info("Trading Agent stopped")
    
    async def generate_trading_signal(self, debate_result: DebateResult,
                                    analyses: AnalysisBundle) -> TradingSignal:
        """Generate final trading signal from debate and analysis"""
        
        try:
            # Extract recommendation from debate
            recommendation = debate_result.final_recommendation
            
            # Create initial trade proposal
            initial_proposal = self._create_trade_proposal(recommendation, analyses)
            
            # Validate with risk management
            risk_validation = await self.risk_manager.validate_trade(initial_proposal)
            
            if not risk_validation.approved:
                return self._create_hold_signal(
                    recommendation.ticker,
                    "Trade rejected by risk management",
                    risk_validation.validation_details
                )
            
            # Generate final signal using approved proposal and risk levels
            final_signal = self._create_trading_signal(
                risk_validation.adjusted_proposal,
                risk_validation.risk_levels,
                debate_result.confidence_score
            )
            
            logger.info(f"Generated trading signal: {final_signal.action} {final_signal.quantity} {final_signal.ticker}")
            return final_signal
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            # Return safe default signal
            ticker = recommendation.ticker if recommendation else "UNKNOWN"
            return self._create_hold_signal(
                ticker,
                f"Error in signal generation: {str(e)}",
                {}
            )
    
    def _create_trade_proposal(self, recommendation: TradingRecommendation,
                             analyses: AnalysisBundle) -> TradeProposal:
        """Create initial trade proposal from recommendation"""
        
        # Get current price from technical analysis
        current_price = self._extract_current_price(analyses, recommendation.ticker)
        
        # Calculate initial quantity based on recommendation confidence
        initial_quantity = self._calculate_initial_quantity(recommendation, current_price)
        
        # Get ATR for volatility calculation
        atr = self._extract_atr(analyses, recommendation.ticker)
        
        return TradeProposal(
            ticker=recommendation.ticker,
            action=recommendation.action,
            entry_price=current_price,
            quantity=initial_quantity,
            rationale=recommendation.rationale,
            confidence=recommendation.confidence,
            atr=atr,
            timeframe=recommendation.timeframe
        )
    
    def _extract_current_price(self, analyses: AnalysisBundle, ticker: str) -> float:
        """Extract current price from analyses"""
        
        # Try to get from technical analysis first
        if analyses.technical:
            for analysis in analyses.technical.ticker_analyses:
                if analysis.ticker == ticker:
                    if analysis.support_resistance.current_price > 0:
                        return analysis.support_resistance.current_price
        
        # Fallback to a default price (in production, would fetch from market data)
        default_prices = {
            'AAPL': 175.0,
            'MSFT': 380.0,
            'GOOGL': 140.0,
            'AMZN': 155.0,
            'TSLA': 200.0,
            'META': 350.0,
            'NVDA': 800.0
        }
        
        return default_prices.get(ticker, 100.0)
    
    def _calculate_initial_quantity(self, recommendation: TradingRecommendation, 
                                  current_price: float) -> int:
        """Calculate initial quantity based on confidence and price"""
        
        # Base position size as percentage of account (from risk management)
        base_position_pct = settings.risk.max_position_pct
        
        # Adjust based on confidence (50% to 100% of max position)
        confidence_multiplier = 0.5 + (recommendation.confidence * 0.5)
        position_pct = base_position_pct * confidence_multiplier
        
        # Assume $100k account for calculation (will be adjusted by risk management)
        assumed_account_value = 100000.0
        position_value = assumed_account_value * position_pct
        
        # Calculate quantity
        quantity = int(position_value / current_price)
        
        # Ensure minimum 1 share
        return max(1, quantity)
    
    def _extract_atr(self, analyses: AnalysisBundle, ticker: str) -> float:
        """Extract ATR (Average True Range) from technical analysis"""
        
        if analyses.technical:
            for analysis in analyses.technical.ticker_analyses:
                if analysis.ticker == ticker:
                    if analysis.indicators.atr:
                        return analysis.indicators.atr
        
        # Default ATR as percentage of price (2% default volatility)
        current_price = self._extract_current_price(analyses, ticker)
        return current_price * 0.02
    
    def _create_trading_signal(self, proposal: TradeProposal, risk_levels,
                             confidence: float) -> TradingSignal:
        """Create final trading signal from approved proposal"""
        
        return TradingSignal(
            ticker=proposal.ticker,
            action=proposal.action,
            quantity=proposal.quantity,
            entry_price=proposal.entry_price,
            stop_loss=risk_levels.stop_loss,
            take_profit=risk_levels.take_profit,
            confidence=confidence,
            timeframe=proposal.timeframe,
            rationale=proposal.rationale,
            risk_reward_ratio=risk_levels.risk_reward_ratio
        )
    
    def _create_hold_signal(self, ticker: str, reason: str, 
                          risk_details: dict) -> TradingSignal:
        """Create a HOLD signal when trade is not approved"""
        
        return TradingSignal(
            ticker=ticker,
            action=TradingAction.HOLD,
            quantity=0,
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            confidence=0.0,
            timeframe="intraday",
            rationale=reason,
            risk_reward_ratio=0.0,
            risk_details=risk_details
        )

class SignalGenerator:
    """Helper class for generating trading signals"""
    
    def __init__(self):
        pass
    
    def generate_signal_from_analyses(self, analyses: AnalysisBundle) -> Optional[TradingSignal]:
        """Generate signal directly from analyses (without debate)"""
        
        if not analyses or not analyses.fundamental:
            return None
        
        # Get primary ticker
        primary_ticker = analyses.fundamental.ticker_analyses[0].ticker
        
        # Calculate overall score from all analyses
        overall_score = self._calculate_overall_score(analyses)
        
        # Determine action based on score
        if overall_score > 0.3:
            action = TradingAction.BUY
        elif overall_score < -0.3:
            action = TradingAction.SELL
        else:
            action = TradingAction.HOLD
        
        # Create trading signal
        if action != TradingAction.HOLD:
            return TradingSignal(
                ticker=primary_ticker,
                action=action,
                quantity=100,  # Default quantity
                entry_price=self._get_current_price(primary_ticker),
                stop_loss=0.0,  # To be calculated by risk management
                take_profit=0.0,  # To be calculated by risk management
                confidence=abs(overall_score),
                timeframe="intraday",
                rationale=f"Generated from analysis score: {overall_score:.2f}",
                risk_reward_ratio=2.0
            )
        
        return None
    
    def _calculate_overall_score(self, analyses: AnalysisBundle) -> float:
        """Calculate overall score from all analyses"""
        
        scores = []
        weights = []
        
        # Fundamental score
        if analyses.fundamental:
            scores.append(analyses.fundamental.overall_sentiment)
            weights.append(0.3)
        
        # Technical score
        if analyses.technical:
            technical_score = self._technical_direction_to_score(
                analyses.technical.overall_direction
            )
            scores.append(technical_score)
            weights.append(0.3)
        
        # Sentiment score
        if analyses.sentiment:
            scores.append(analyses.sentiment.aggregate_sentiment)
            weights.append(0.2)
        
        # Macro score
        if analyses.macro:
            macro_score = self._risk_assessment_to_score(
                analyses.macro.risk_assessment
            )
            scores.append(macro_score)
            weights.append(0.2)
        
        # Calculate weighted average
        if scores:
            total_weight = sum(weights[:len(scores)])
            weighted_sum = sum(
                score * weight 
                for score, weight in zip(scores, weights[:len(scores)])
            )
            return weighted_sum / total_weight
        
        return 0.0
    
    def _technical_direction_to_score(self, direction: str) -> float:
        """Convert technical direction to numeric score"""
        direction_scores = {
            'bullish': 0.7,
            'bearish': -0.7,
            'neutral': 0.0
        }
        return direction_scores.get(direction.lower(), 0.0)
    
    def _risk_assessment_to_score(self, risk_assessment: str) -> float:
        """Convert risk assessment to numeric score"""
        risk_scores = {
            'low': 0.3,
            'medium': 0.0,
            'high': -0.3
        }
        return risk_scores.get(risk_assessment.lower(), 0.0)
    
    def _get_current_price(self, ticker: str) -> float:
        """Get current price for ticker (placeholder)"""
        # This would fetch from market data in production
        default_prices = {
            'AAPL': 175.0,
            'MSFT': 380.0,
            'GOOGL': 140.0,
            'AMZN': 155.0,
            'TSLA': 200.0
        }
        return default_prices.get(ticker, 100.0)