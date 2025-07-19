import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import aiohttp

from ..models.events import EnhancedMarketEvent
from ..models.analysis import (
    FundamentalAnalysis, TickerFundamentalAnalysis, FundamentalMetrics
)
from ..config.settings import settings

logger = logging.getLogger(__name__)

class FundamentalAnalysisAgent:
    """Agent for fundamental analysis of market events"""
    
    def __init__(self):
        self.finnhub_api_key = settings.finnhub_api_key
        self.session = None
        
    async def start(self):
        """Initialize the agent"""
        self.session = aiohttp.ClientSession()
        logger.info("Fundamental Analysis Agent started")
    
    async def stop(self):
        """Stop the agent"""
        if self.session:
            await self.session.close()
        logger.info("Fundamental Analysis Agent stopped")
    
    async def analyze(self, event: EnhancedMarketEvent) -> FundamentalAnalysis:
        """Perform fundamental analysis on the event"""
        analysis_results = []
        
        for ticker in event.original_event.tickers:
            try:
                # Fetch financial data
                financial_data = await self._fetch_financial_data(ticker)
                
                # Calculate key metrics
                metrics = self._calculate_key_metrics(financial_data)
                
                # Assess valuation
                valuation = await self._assess_valuation(ticker, metrics, event)
                
                # Generate narrative analysis
                narrative = await self._generate_analysis_narrative(
                    ticker, financial_data, metrics, valuation, event
                )
                
                # Calculate confidence and score
                confidence = self._calculate_confidence(metrics, valuation)
                score = self._calculate_fundamental_score(metrics, valuation, event)
                
                analysis_results.append(
                    TickerFundamentalAnalysis(
                        ticker=ticker,
                        metrics=metrics,
                        valuation=valuation,
                        narrative=narrative,
                        confidence=confidence,
                        score=score
                    )
                )
                
            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {e}")
                # Add error case
                analysis_results.append(
                    TickerFundamentalAnalysis(
                        ticker=ticker,
                        metrics=FundamentalMetrics(),
                        valuation="unknown",
                        narrative=f"Unable to analyze {ticker} due to data unavailability",
                        confidence=0.0,
                        score=0.0
                    )
                )
        
        return FundamentalAnalysis(
            event_id=event.id,
            ticker_analyses=analysis_results,
            overall_sentiment=self._calculate_overall_sentiment(analysis_results)
        )
    
    async def _fetch_financial_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch financial data for a ticker"""
        try:
            # Fetch basic financials from Finnhub
            url = "https://finnhub.io/api/v1/stock/metric"
            params = {
                "symbol": ticker,
                "metric": "all",
                "token": self.finnhub_api_key
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("metric", {})
                else:
                    logger.warning(f"Failed to fetch data for {ticker}: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error fetching financial data for {ticker}: {e}")
            return {}
    
    def _calculate_key_metrics(self, financial_data: Dict[str, Any]) -> FundamentalMetrics:
        """Calculate fundamental metrics from financial data"""
        def safe_get(key: str, default: Optional[float] = None) -> Optional[float]:
            value = financial_data.get(key)
            if value is not None and isinstance(value, (int, float)):
                return float(value)
            return default
        
        return FundamentalMetrics(
            pe_ratio=safe_get("peBasicExclExtraTTM"),
            peg_ratio=safe_get("pegBasicExclExtraTTM"),
            debt_to_equity=safe_get("totalDebt/totalEquityQuarterly"),
            roe=safe_get("roeRfy"),
            revenue_growth=safe_get("revenueGrowthTTMYoy"),
            earnings_growth=safe_get("epsGrowthTTMYoy"),
            free_cash_flow_yield=safe_get("freeCashFlowTTM"),
            market_cap=safe_get("marketCapitalization"),
            enterprise_value=safe_get("enterpriseValue")
        )
    
    async def _assess_valuation(
        self, 
        ticker: str, 
        metrics: FundamentalMetrics, 
        event: EnhancedMarketEvent
    ) -> str:
        """Assess if the stock is undervalued, fair value, or overvalued"""
        
        # Simple valuation logic based on PE ratio and growth
        if metrics.pe_ratio is None:
            return "unknown"
        
        # Basic valuation thresholds
        pe_threshold_low = 15
        pe_threshold_high = 25
        
        # Adjust thresholds based on growth
        if metrics.earnings_growth is not None:
            if metrics.earnings_growth > 20:  # High growth
                pe_threshold_high = 35
            elif metrics.earnings_growth < 5:  # Low growth
                pe_threshold_high = 20
        
        if metrics.pe_ratio < pe_threshold_low:
            return "undervalued"
        elif metrics.pe_ratio > pe_threshold_high:
            return "overvalued"
        else:
            return "fair_value"
    
    async def _generate_analysis_narrative(
        self,
        ticker: str,
        financial_data: Dict[str, Any],
        metrics: FundamentalMetrics,
        valuation: str,
        event: EnhancedMarketEvent
    ) -> str:
        """Generate narrative analysis"""
        
        narrative_parts = []
        
        # Company valuation assessment
        narrative_parts.append(f"{ticker} appears {valuation} based on current metrics.")
        
        # PE ratio analysis
        if metrics.pe_ratio is not None:
            if metrics.pe_ratio < 10:
                narrative_parts.append(f"The PE ratio of {metrics.pe_ratio:.1f} suggests the stock may be undervalued.")
            elif metrics.pe_ratio > 30:
                narrative_parts.append(f"The PE ratio of {metrics.pe_ratio:.1f} indicates high market expectations.")
            else:
                narrative_parts.append(f"The PE ratio of {metrics.pe_ratio:.1f} is within reasonable range.")
        
        # Growth analysis
        if metrics.revenue_growth is not None and metrics.earnings_growth is not None:
            if metrics.revenue_growth > 15 and metrics.earnings_growth > 15:
                narrative_parts.append("Strong revenue and earnings growth support positive fundamentals.")
            elif metrics.revenue_growth < 0 or metrics.earnings_growth < 0:
                narrative_parts.append("Declining revenue or earnings growth raises concerns.")
        
        # Financial health
        if metrics.debt_to_equity is not None:
            if metrics.debt_to_equity > 1.0:
                narrative_parts.append(f"High debt-to-equity ratio ({metrics.debt_to_equity:.1f}) may indicate financial risk.")
            elif metrics.debt_to_equity < 0.3:
                narrative_parts.append("Low debt levels indicate strong financial position.")
        
        # Event context
        event_sentiment = event.original_event.sentiment
        if event_sentiment > 0.5:
            narrative_parts.append("Positive market event supports bullish fundamental outlook.")
        elif event_sentiment < -0.5:
            narrative_parts.append("Negative market event raises fundamental concerns.")
        
        return " ".join(narrative_parts)
    
    def _calculate_confidence(self, metrics: FundamentalMetrics, valuation: str) -> float:
        """Calculate confidence in the analysis"""
        confidence = 0.0
        
        # Base confidence
        confidence += 0.2
        
        # Data availability boosts confidence
        available_metrics = 0
        total_metrics = 9  # Total number of metrics we track
        
        for field_name in metrics.__fields__:
            if getattr(metrics, field_name) is not None:
                available_metrics += 1
        
        data_coverage = available_metrics / total_metrics
        confidence += data_coverage * 0.5
        
        # Valuation certainty
        if valuation != "unknown":
            confidence += 0.3
        
        return min(1.0, confidence)
    
    def _calculate_fundamental_score(
        self, 
        metrics: FundamentalMetrics, 
        valuation: str,
        event: EnhancedMarketEvent
    ) -> float:
        """Calculate fundamental score (-1 to +1)"""
        score = 0.0
        
        # Valuation score
        if valuation == "undervalued":
            score += 0.4
        elif valuation == "overvalued":
            score -= 0.4
        
        # Growth score
        if metrics.revenue_growth is not None:
            if metrics.revenue_growth > 15:
                score += 0.2
            elif metrics.revenue_growth < 0:
                score -= 0.2
        
        if metrics.earnings_growth is not None:
            if metrics.earnings_growth > 15:
                score += 0.2
            elif metrics.earnings_growth < 0:
                score -= 0.2
        
        # Financial health score
        if metrics.debt_to_equity is not None:
            if metrics.debt_to_equity < 0.3:
                score += 0.1
            elif metrics.debt_to_equity > 1.5:
                score -= 0.2
        
        if metrics.roe is not None:
            if metrics.roe > 15:
                score += 0.1
            elif metrics.roe < 5:
                score -= 0.1
        
        # Event impact
        event_sentiment = event.original_event.sentiment
        score += event_sentiment * 0.3
        
        return max(-1.0, min(1.0, score))
    
    def _calculate_overall_sentiment(self, analyses: List[TickerFundamentalAnalysis]) -> float:
        """Calculate overall sentiment from individual analyses"""
        if not analyses:
            return 0.0
        
        # Weight by confidence
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for analysis in analyses:
            weight = analysis.confidence
            total_weighted_score += analysis.score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return total_weighted_score / total_weight