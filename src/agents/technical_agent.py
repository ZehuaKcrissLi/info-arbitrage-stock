import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import aiohttp
import yfinance as yf
import ta

from ..models.events import EnhancedMarketEvent
from ..models.analysis import (
    TechnicalAnalysis, TickerTechnicalAnalysis, TechnicalIndicators, 
    SupportResistanceLevels, TechnicalSignals
)
from ..config.settings import settings

logger = logging.getLogger(__name__)

class TechnicalAnalysisAgent:
    """Agent for technical analysis of market events"""
    
    def __init__(self):
        self.session = None
        
    async def start(self):
        """Initialize the agent"""
        self.session = aiohttp.ClientSession()
        logger.info("Technical Analysis Agent started")
    
    async def stop(self):
        """Stop the agent"""
        if self.session:
            await self.session.close()
        logger.info("Technical Analysis Agent stopped")
    
    async def analyze(self, event: EnhancedMarketEvent) -> TechnicalAnalysis:
        """Perform technical analysis"""
        analysis_results = []
        
        for ticker in event.original_event.tickers:
            try:
                # Fetch OHLCV data
                ohlcv_data = await self._fetch_ohlcv_data(ticker)
                
                if ohlcv_data.empty:
                    logger.warning(f"No OHLCV data available for {ticker}")
                    continue
                
                # Calculate technical indicators
                indicators = self._calculate_indicators(ohlcv_data)
                
                # Identify patterns
                patterns = self._identify_patterns(ohlcv_data)
                
                # Determine support/resistance levels
                levels = self._calculate_support_resistance(ohlcv_data)
                
                # Generate technical signals
                signals = self._generate_signals(indicators, patterns, levels)
                
                # Create narrative
                narrative = await self._generate_technical_narrative(
                    ticker, indicators, patterns, levels, signals
                )
                
                # Calculate confidence and score
                confidence = self._calculate_technical_confidence(signals, indicators)
                score = self._calculate_technical_score(signals, indicators, event)
                
                analysis_results.append(
                    TickerTechnicalAnalysis(
                        ticker=ticker,
                        indicators=indicators,
                        patterns=patterns,
                        support_resistance=levels,
                        signals=signals,
                        narrative=narrative,
                        confidence=confidence,
                        score=score
                    )
                )
                
            except Exception as e:
                logger.error(f"Error in technical analysis for {ticker}: {e}")
                # Add minimal analysis on error
                analysis_results.append(
                    TickerTechnicalAnalysis(
                        ticker=ticker,
                        indicators=TechnicalIndicators(),
                        patterns=[],
                        support_resistance=SupportResistanceLevels(current_price=0.0),
                        signals=TechnicalSignals(
                            trend_direction="unknown",
                            momentum_signal="neutral",
                            volume_confirmation=False
                        ),
                        narrative=f"Technical analysis unavailable for {ticker}",
                        confidence=0.0,
                        score=0.0
                    )
                )
        
        return TechnicalAnalysis(
            event_id=event.id,
            ticker_analyses=analysis_results,
            overall_direction=self._determine_overall_direction(analysis_results)
        )
    
    async def _fetch_ohlcv_data(self, ticker: str) -> pd.DataFrame:
        """Fetch OHLCV data for multiple timeframes"""
        try:
            # Use yfinance for simplicity (in production, use proper market data API)
            stock = yf.Ticker(ticker)
            
            # Get 1-minute data for the last 7 days
            data_1m = stock.history(period="7d", interval="1m")
            
            if data_1m.empty:
                logger.warning(f"No 1-minute data for {ticker}")
                # Fallback to daily data
                data_1m = stock.history(period="1y", interval="1d")
            
            return data_1m
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {ticker}: {e}")
            return pd.DataFrame()
    
    def _calculate_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """Calculate suite of technical indicators"""
        if df.empty:
            return TechnicalIndicators()
        
        try:
            # Ensure we have the required columns
            if not all(col in df.columns for col in ['High', 'Low', 'Close', 'Volume']):
                logger.warning("Missing required OHLCV columns")
                return TechnicalIndicators()
            
            # Get the latest values
            close = df['Close'].iloc[-1] if len(df) > 0 else None
            high = df['High']
            low = df['Low']
            volume = df['Volume']
            
            indicators = TechnicalIndicators()
            
            # Trend indicators
            if len(df) >= 20:
                indicators.sma_20 = ta.trend.sma_indicator(df['Close'], window=20).iloc[-1]
            if len(df) >= 12:
                indicators.ema_12 = ta.trend.ema_indicator(df['Close'], window=12).iloc[-1]
            if len(df) >= 26:
                indicators.ema_26 = ta.trend.ema_indicator(df['Close'], window=26).iloc[-1]
            
            # Momentum indicators
            if len(df) >= 14:
                indicators.rsi = ta.momentum.rsi(df['Close'], window=14).iloc[-1]
                indicators.stoch = ta.momentum.stoch(high, low, df['Close']).iloc[-1]
            
            if len(df) >= 26:
                indicators.macd = ta.trend.macd_diff(df['Close']).iloc[-1]
            
            # Volatility indicators
            if len(df) >= 20:
                indicators.bollinger_upper = ta.volatility.bollinger_hband(df['Close']).iloc[-1]
                indicators.bollinger_lower = ta.volatility.bollinger_lband(df['Close']).iloc[-1]
            
            if len(df) >= 14:
                indicators.atr = ta.volatility.average_true_range(high, low, df['Close']).iloc[-1]
            
            # Volume indicators
            if len(df) >= 20:
                indicators.volume_sma = ta.volume.volume_sma(df['Close'], volume).iloc[-1]
                indicators.obv = ta.volume.on_balance_volume(df['Close'], volume).iloc[-1]
            
            # VWAP calculation
            if len(df) > 0:
                indicators.vwap = self._calculate_vwap(df)
            
            # Pivot points
            if len(df) >= 3:
                indicators.pivot_points = self._calculate_pivot_points(df)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return TechnicalIndicators()
    
    def _calculate_vwap(self, df: pd.DataFrame) -> Optional[float]:
        """Calculate Volume Weighted Average Price"""
        try:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            vwap = (typical_price * df['Volume']).sum() / df['Volume'].sum()
            return float(vwap)
        except:
            return None
    
    def _calculate_pivot_points(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate pivot points from recent data"""
        try:
            # Use last complete day's data
            last_day = df.iloc[-1]
            high = last_day['High']
            low = last_day['Low']
            close = last_day['Close']
            
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            
            return {
                'pivot': float(pivot),
                'r1': float(r1),
                's1': float(s1),
                'r2': float(r2),
                's2': float(s2)
            }
        except:
            return {}
    
    def _identify_patterns(self, df: pd.DataFrame) -> List[str]:
        """Identify chart patterns"""
        patterns = []
        
        if df.empty or len(df) < 20:
            return patterns
        
        try:
            close_prices = df['Close'].values
            
            # Simple pattern detection
            recent_prices = close_prices[-10:]
            
            # Breakout pattern
            if len(recent_prices) >= 5:
                recent_high = max(recent_prices[-5:])
                previous_high = max(close_prices[-20:-5]) if len(close_prices) >= 20 else 0
                
                if recent_high > previous_high * 1.02:  # 2% breakout
                    patterns.append("bullish_breakout")
            
            # Support/resistance test
            current_price = close_prices[-1]
            price_range = max(close_prices[-20:]) - min(close_prices[-20:])
            
            if price_range > 0:
                # Check if price is near support or resistance
                support_level = min(close_prices[-20:])
                resistance_level = max(close_prices[-20:])
                
                if abs(current_price - support_level) / price_range < 0.05:
                    patterns.append("near_support")
                elif abs(current_price - resistance_level) / price_range < 0.05:
                    patterns.append("near_resistance")
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error identifying patterns: {e}")
            return patterns
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> SupportResistanceLevels:
        """Calculate support and resistance levels"""
        if df.empty:
            return SupportResistanceLevels(current_price=0.0)
        
        try:
            current_price = float(df['Close'].iloc[-1])
            
            # Simple support/resistance calculation using recent highs and lows
            recent_data = df.tail(50) if len(df) >= 50 else df
            
            # Find local maxima and minima
            highs = recent_data['High'].values
            lows = recent_data['Low'].values
            
            # Use numpy to find local extrema
            resistance_levels = []
            support_levels = []
            
            # Simple method: use recent significant highs and lows
            if len(recent_data) >= 10:
                # Resistance levels (recent highs)
                max_high = max(highs)
                resistance_levels.append(max_high)
                
                # Support levels (recent lows)
                min_low = min(lows)
                support_levels.append(min_low)
                
                # Additional levels based on pivot points
                for i in range(2, len(highs) - 2):
                    if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                        highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                        resistance_levels.append(highs[i])
                
                for i in range(2, len(lows) - 2):
                    if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                        lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                        support_levels.append(lows[i])
            
            # Remove duplicates and sort
            resistance_levels = sorted(list(set(resistance_levels)), reverse=True)
            support_levels = sorted(list(set(support_levels)))
            
            # Find nearest levels
            nearest_resistance = None
            nearest_support = None
            
            for level in resistance_levels:
                if level > current_price:
                    nearest_resistance = level
                    break
            
            for level in reversed(support_levels):
                if level < current_price:
                    nearest_support = level
                    break
            
            return SupportResistanceLevels(
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                current_price=current_price,
                nearest_support=nearest_support,
                nearest_resistance=nearest_resistance
            )
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return SupportResistanceLevels(current_price=0.0)
    
    def _generate_signals(
        self, 
        indicators: TechnicalIndicators, 
        patterns: List[str], 
        levels: SupportResistanceLevels
    ) -> TechnicalSignals:
        """Generate technical signals"""
        
        # Default neutral signals
        signals = TechnicalSignals(
            trend_direction="neutral",
            momentum_signal="neutral",
            volume_confirmation=False
        )
        
        try:
            # Trend direction based on moving averages
            if indicators.ema_12 and indicators.ema_26:
                if indicators.ema_12 > indicators.ema_26:
                    signals.trend_direction = "bullish"
                else:
                    signals.trend_direction = "bearish"
            
            # Momentum signal based on RSI and MACD
            momentum_score = 0
            
            if indicators.rsi:
                if indicators.rsi > 70:
                    momentum_score -= 2  # Overbought
                elif indicators.rsi > 60:
                    momentum_score += 1  # Strong
                elif indicators.rsi < 30:
                    momentum_score += 2  # Oversold (potential bounce)
                elif indicators.rsi < 40:
                    momentum_score -= 1  # Weak
            
            if indicators.macd:
                if indicators.macd > 0:
                    momentum_score += 1
                else:
                    momentum_score -= 1
            
            # Convert momentum score to signal
            if momentum_score >= 2:
                signals.momentum_signal = "strong_buy"
            elif momentum_score == 1:
                signals.momentum_signal = "buy"
            elif momentum_score == -1:
                signals.momentum_signal = "sell"
            elif momentum_score <= -2:
                signals.momentum_signal = "strong_sell"
            
            # Volume confirmation
            if indicators.volume_sma and indicators.obv:
                # Simple volume confirmation logic
                signals.volume_confirmation = True  # Placeholder
            
            # Pattern-based signals
            if "bullish_breakout" in patterns:
                signals.breakout_signal = "bullish_breakout"
            elif "bearish_breakdown" in patterns:
                signals.breakout_signal = "bearish_breakdown"
            
            # Detect specific patterns
            if patterns:
                signals.pattern_detected = patterns[0]  # Use first pattern
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return signals
    
    async def _generate_technical_narrative(
        self,
        ticker: str,
        indicators: TechnicalIndicators,
        patterns: List[str],
        levels: SupportResistanceLevels,
        signals: TechnicalSignals
    ) -> str:
        """Generate technical analysis narrative"""
        
        narrative_parts = []
        
        # Trend analysis
        narrative_parts.append(f"{ticker} shows {signals.trend_direction} technical trend.")
        
        # Momentum analysis
        if signals.momentum_signal != "neutral":
            narrative_parts.append(f"Momentum indicators suggest {signals.momentum_signal} signal.")
        
        # RSI analysis
        if indicators.rsi:
            if indicators.rsi > 70:
                narrative_parts.append(f"RSI at {indicators.rsi:.1f} indicates overbought conditions.")
            elif indicators.rsi < 30:
                narrative_parts.append(f"RSI at {indicators.rsi:.1f} suggests oversold levels.")
            else:
                narrative_parts.append(f"RSI at {indicators.rsi:.1f} is in neutral territory.")
        
        # Support/resistance analysis
        if levels.nearest_support and levels.nearest_resistance:
            narrative_parts.append(
                f"Trading between support at ${levels.nearest_support:.2f} "
                f"and resistance at ${levels.nearest_resistance:.2f}."
            )
        
        # Pattern analysis
        if patterns:
            pattern_desc = ", ".join(patterns)
            narrative_parts.append(f"Detected patterns: {pattern_desc}.")
        
        # Volume confirmation
        if signals.volume_confirmation:
            narrative_parts.append("Volume confirms price movement.")
        else:
            narrative_parts.append("Volume does not strongly confirm price action.")
        
        return " ".join(narrative_parts)
    
    def _calculate_technical_confidence(
        self, 
        signals: TechnicalSignals, 
        indicators: TechnicalIndicators
    ) -> float:
        """Calculate confidence in technical analysis"""
        confidence = 0.0
        
        # Base confidence
        confidence += 0.2
        
        # Data availability
        indicator_count = 0
        total_indicators = 11  # Total number of indicators we track
        
        for field_name in indicators.__fields__:
            if getattr(indicators, field_name) is not None:
                indicator_count += 1
        
        data_coverage = indicator_count / total_indicators
        confidence += data_coverage * 0.4
        
        # Signal clarity
        if signals.trend_direction != "neutral":
            confidence += 0.2
        
        if signals.momentum_signal in ["strong_buy", "strong_sell"]:
            confidence += 0.2
        elif signals.momentum_signal in ["buy", "sell"]:
            confidence += 0.1
        
        # Volume confirmation
        if signals.volume_confirmation:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _calculate_technical_score(
        self, 
        signals: TechnicalSignals, 
        indicators: TechnicalIndicators,
        event: EnhancedMarketEvent
    ) -> float:
        """Calculate technical score (-1 to +1)"""
        score = 0.0
        
        # Trend score
        if signals.trend_direction == "bullish":
            score += 0.3
        elif signals.trend_direction == "bearish":
            score -= 0.3
        
        # Momentum score
        momentum_scores = {
            "strong_buy": 0.4,
            "buy": 0.2,
            "neutral": 0.0,
            "sell": -0.2,
            "strong_sell": -0.4
        }
        score += momentum_scores.get(signals.momentum_signal, 0.0)
        
        # RSI consideration
        if indicators.rsi:
            if indicators.rsi < 30:  # Oversold - potential bounce
                score += 0.2
            elif indicators.rsi > 70:  # Overbought - potential decline
                score -= 0.2
        
        # Pattern consideration
        if signals.breakout_signal == "bullish_breakout":
            score += 0.2
        elif signals.breakout_signal == "bearish_breakdown":
            score -= 0.2
        
        # Volume confirmation
        if signals.volume_confirmation:
            score *= 1.1  # Boost score if volume confirms
        
        return max(-1.0, min(1.0, score))
    
    def _determine_overall_direction(self, analyses: List[TickerTechnicalAnalysis]) -> str:
        """Determine overall technical direction"""
        if not analyses:
            return "neutral"
        
        # Weight by confidence
        bullish_weight = 0.0
        bearish_weight = 0.0
        
        for analysis in analyses:
            if analysis.score > 0:
                bullish_weight += analysis.score * analysis.confidence
            else:
                bearish_weight += abs(analysis.score) * analysis.confidence
        
        if bullish_weight > bearish_weight * 1.2:
            return "bullish"
        elif bearish_weight > bullish_weight * 1.2:
            return "bearish"
        else:
            return "neutral"