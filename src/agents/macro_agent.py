import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import aiohttp
import yfinance as yf

from ..models.events import EnhancedMarketEvent
from ..models.analysis import MacroAnalysis, MacroIndicators
from ..config.settings import settings

logger = logging.getLogger(__name__)

class MacroAnalysisAgent:
    """Agent for macro economic analysis"""
    
    def __init__(self):
        self.session = None
        
    async def start(self):
        """Initialize the agent"""
        self.session = aiohttp.ClientSession()
        logger.info("Macro Analysis Agent started")
    
    async def stop(self):
        """Stop the agent"""
        if self.session:
            await self.session.close()
        logger.info("Macro Analysis Agent stopped")
    
    async def analyze(self, event: EnhancedMarketEvent) -> MacroAnalysis:
        """Analyze macro environment and its impact"""
        
        # Fetch current macro indicators
        macro_indicators = await self._fetch_macro_indicators()
        
        # Analyze market structure
        market_structure = await self._analyze_market_structure()
        
        # Assess sector rotation
        sector_analysis = await self._analyze_sector_rotation(event.original_event.tickers)
        
        # Generate macro narrative
        narrative = await self._generate_macro_narrative(
            event, macro_indicators, market_structure, sector_analysis
        )
        
        # Assess macro risk
        risk_assessment = self._assess_macro_risk(macro_indicators, market_structure)
        
        return MacroAnalysis(
            event_id=event.id,
            macro_indicators=macro_indicators,
            market_structure=market_structure,
            sector_analysis=sector_analysis,
            narrative=narrative,
            risk_assessment=risk_assessment
        )
    
    async def _fetch_macro_indicators(self) -> MacroIndicators:
        """Fetch current macro economic indicators"""
        indicators = MacroIndicators()
        
        try:
            # Use yfinance to get market data
            # VIX
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="1d")
            if not vix_data.empty:
                indicators.vix = float(vix_data['Close'].iloc[-1])
            
            # Treasury yields (using ETF proxies)
            try:
                # 10-year treasury (^TNX)
                tnx = yf.Ticker("^TNX")
                tnx_data = tnx.history(period="1d")
                if not tnx_data.empty:
                    indicators.treasury_10y = float(tnx_data['Close'].iloc[-1])
                
                # 2-year treasury (^IRX for 13-week, using as proxy)
                irx = yf.Ticker("^IRX")
                irx_data = irx.history(period="1d")
                if not irx_data.empty:
                    indicators.treasury_2y = float(irx_data['Close'].iloc[-1])
            except:
                logger.warning("Could not fetch treasury yields")
            
            # Dollar Index
            try:
                dxy = yf.Ticker("DX-Y.NYB")
                dxy_data = dxy.history(period="1d")
                if not dxy_data.empty:
                    indicators.dxy = float(dxy_data['Close'].iloc[-1])
            except:
                logger.warning("Could not fetch DXY")
            
            # Oil price (WTI)
            try:
                oil = yf.Ticker("CL=F")
                oil_data = oil.history(period="1d")
                if not oil_data.empty:
                    indicators.oil_price = float(oil_data['Close'].iloc[-1])
            except:
                logger.warning("Could not fetch oil price")
            
            # Gold price
            try:
                gold = yf.Ticker("GC=F")
                gold_data = gold.history(period="1d")
                if not gold_data.empty:
                    indicators.gold_price = float(gold_data['Close'].iloc[-1])
            except:
                logger.warning("Could not fetch gold price")
            
            # SPY performance
            spy = yf.Ticker("SPY")
            spy_data = spy.history(period="5d")
            if len(spy_data) >= 2:
                today_close = spy_data['Close'].iloc[-1]
                yesterday_close = spy_data['Close'].iloc[-2]
                indicators.spy_performance = float((today_close - yesterday_close) / yesterday_close * 100)
            
            # QQQ performance
            qqq = yf.Ticker("QQQ")
            qqq_data = qqq.history(period="5d")
            if len(qqq_data) >= 2:
                today_close = qqq_data['Close'].iloc[-1]
                yesterday_close = qqq_data['Close'].iloc[-2]
                indicators.qqq_performance = float((today_close - yesterday_close) / yesterday_close * 100)
            
            # Economic calendar (placeholder)
            indicators.economic_calendar = await self._get_today_economic_events()
            
        except Exception as e:
            logger.error(f"Error fetching macro indicators: {e}")
        
        return indicators
    
    async def _get_today_economic_events(self) -> List[Dict[str, Any]]:
        """Get today's economic calendar events (placeholder)"""
        # This would integrate with an economic calendar API
        # For now, return mock events
        
        mock_events = [
            {
                "time": "08:30",
                "event": "Initial Jobless Claims",
                "importance": "medium",
                "forecast": "220K",
                "previous": "225K"
            },
            {
                "time": "14:00",
                "event": "Fed Speaker",
                "importance": "high",
                "forecast": "N/A",
                "previous": "N/A"
            }
        ]
        
        return mock_events
    
    async def _analyze_market_structure(self) -> Dict[str, Any]:
        """Analyze current market structure"""
        market_structure = {}
        
        try:
            # Analyze sector performance
            sector_etfs = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financials': 'XLF',
                'Energy': 'XLE',
                'Utilities': 'XLU',
                'Consumer Discretionary': 'XLY',
                'Consumer Staples': 'XLP',
                'Industrials': 'XLI',
                'Materials': 'XLB'
            }
            
            sector_performance = {}
            for sector, etf in sector_etfs.items():
                try:
                    ticker = yf.Ticker(etf)
                    data = ticker.history(period="5d")
                    if len(data) >= 2:
                        today_close = data['Close'].iloc[-1]
                        yesterday_close = data['Close'].iloc[-2]
                        performance = (today_close - yesterday_close) / yesterday_close * 100
                        sector_performance[sector] = float(performance)
                except:
                    continue
            
            market_structure['sector_performance'] = sector_performance
            
            # Market breadth indicators
            market_structure['market_breadth'] = await self._calculate_market_breadth()
            
            # Risk-on vs Risk-off sentiment
            market_structure['risk_sentiment'] = self._assess_risk_sentiment()
            
        except Exception as e:
            logger.error(f"Error analyzing market structure: {e}")
        
        return market_structure
    
    async def _calculate_market_breadth(self) -> Dict[str, float]:
        """Calculate market breadth indicators"""
        breadth = {}
        
        try:
            # Simple breadth calculation using major indices
            indices = {
                'SPY': 'S&P 500',
                'QQQ': 'NASDAQ',
                'IWM': 'Russell 2000'
            }
            
            performances = []
            for symbol, name in indices.items():
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="5d")
                    if len(data) >= 2:
                        today_close = data['Close'].iloc[-1]
                        yesterday_close = data['Close'].iloc[-2]
                        performance = (today_close - yesterday_close) / yesterday_close * 100
                        performances.append(performance)
                        breadth[f'{name}_performance'] = float(performance)
                except:
                    continue
            
            # Calculate breadth score
            if performances:
                positive_count = sum(1 for p in performances if p > 0)
                breadth['positive_ratio'] = positive_count / len(performances)
                breadth['average_performance'] = sum(performances) / len(performances)
        
        except Exception as e:
            logger.error(f"Error calculating market breadth: {e}")
        
        return breadth
    
    def _assess_risk_sentiment(self) -> str:
        """Assess current risk-on vs risk-off sentiment"""
        # This would analyze various risk indicators
        # For now, return a placeholder assessment
        return "neutral"
    
    async def _analyze_sector_rotation(self, tickers: List[str]) -> Dict[str, float]:
        """Analyze sector rotation and its impact on specific tickers"""
        sector_analysis = {}
        
        # Map tickers to sectors (simplified)
        ticker_sectors = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'AMZN': 'Consumer Discretionary',
            'TSLA': 'Consumer Discretionary',
            'JNJ': 'Healthcare',
            'PFE': 'Healthcare',
            'XOM': 'Energy',
            'JPM': 'Financials',
            'BAC': 'Financials'
        }
        
        try:
            for ticker in tickers:
                sector = ticker_sectors.get(ticker, 'Unknown')
                
                if sector != 'Unknown':
                    # Get sector ETF performance
                    sector_etf_map = {
                        'Technology': 'XLK',
                        'Healthcare': 'XLV',
                        'Financials': 'XLF',
                        'Energy': 'XLE',
                        'Consumer Discretionary': 'XLY',
                        'Consumer Staples': 'XLP'
                    }
                    
                    etf_symbol = sector_etf_map.get(sector)
                    if etf_symbol:
                        try:
                            etf = yf.Ticker(etf_symbol)
                            data = etf.history(period="5d")
                            if len(data) >= 2:
                                today_close = data['Close'].iloc[-1]
                                yesterday_close = data['Close'].iloc[-2]
                                performance = (today_close - yesterday_close) / yesterday_close * 100
                                sector_analysis[f'{ticker}_sector_performance'] = float(performance)
                        except:
                            sector_analysis[f'{ticker}_sector_performance'] = 0.0
                else:
                    sector_analysis[f'{ticker}_sector_performance'] = 0.0
                    
        except Exception as e:
            logger.error(f"Error analyzing sector rotation: {e}")
        
        return sector_analysis
    
    async def _generate_macro_narrative(
        self,
        event: EnhancedMarketEvent,
        macro_indicators: MacroIndicators,
        market_structure: Dict[str, Any],
        sector_analysis: Dict[str, float]
    ) -> str:
        """Generate macro analysis narrative"""
        
        narrative_parts = []
        
        # VIX analysis
        if macro_indicators.vix is not None:
            if macro_indicators.vix > 25:
                narrative_parts.append(f"High volatility (VIX: {macro_indicators.vix:.1f}) indicates market uncertainty.")
            elif macro_indicators.vix < 15:
                narrative_parts.append(f"Low volatility (VIX: {macro_indicators.vix:.1f}) suggests market complacency.")
            else:
                narrative_parts.append(f"Moderate volatility (VIX: {macro_indicators.vix:.1f}) shows balanced market conditions.")
        
        # Treasury yield analysis
        if macro_indicators.treasury_10y is not None and macro_indicators.treasury_2y is not None:
            yield_spread = macro_indicators.treasury_10y - macro_indicators.treasury_2y
            if yield_spread < 0:
                narrative_parts.append("Inverted yield curve suggests economic concerns.")
            elif yield_spread > 2.0:
                narrative_parts.append("Steep yield curve indicates strong growth expectations.")
            else:
                narrative_parts.append("Normal yield curve shape supports stable outlook.")
        
        # Market performance analysis
        if macro_indicators.spy_performance is not None and macro_indicators.qqq_performance is not None:
            if macro_indicators.spy_performance > 1 and macro_indicators.qqq_performance > 1:
                narrative_parts.append("Strong broad market performance supports risk-on sentiment.")
            elif macro_indicators.spy_performance < -1 and macro_indicators.qqq_performance < -1:
                narrative_parts.append("Weak market performance indicates risk-off conditions.")
            
            # Tech vs broad market comparison
            if macro_indicators.qqq_performance > macro_indicators.spy_performance + 0.5:
                narrative_parts.append("Technology sector outperforming broad market.")
            elif macro_indicators.spy_performance > macro_indicators.qqq_performance + 0.5:
                narrative_parts.append("Broad market outperforming technology sector.")
        
        # Sector rotation analysis
        if sector_analysis:
            strong_sectors = []
            weak_sectors = []
            
            for key, performance in sector_analysis.items():
                if 'sector_performance' in key:
                    if performance > 1:
                        strong_sectors.append(key.replace('_sector_performance', ''))
                    elif performance < -1:
                        weak_sectors.append(key.replace('_sector_performance', ''))
            
            if strong_sectors:
                narrative_parts.append(f"Strong sector performance in: {', '.join(strong_sectors)}.")
            if weak_sectors:
                narrative_parts.append(f"Weak sector performance in: {', '.join(weak_sectors)}.")
        
        # Dollar and commodities
        if macro_indicators.dxy is not None:
            narrative_parts.append(f"Dollar index at {macro_indicators.dxy:.1f}.")
        
        if macro_indicators.oil_price is not None:
            narrative_parts.append(f"Oil trading at ${macro_indicators.oil_price:.1f}.")
        
        # Economic calendar impact
        if macro_indicators.economic_calendar:
            high_impact_events = [
                event for event in macro_indicators.economic_calendar 
                if event.get('importance') == 'high'
            ]
            if high_impact_events:
                narrative_parts.append(f"High-impact economic events today: {len(high_impact_events)} scheduled.")
        
        # Event context
        event_type = event.original_event.event_type.value
        if event_type in ['earnings', 'merger']:
            narrative_parts.append(f"Macro environment generally {'supportive' if macro_indicators.spy_performance and macro_indicators.spy_performance > 0 else 'challenging'} for {event_type} events.")
        
        return " ".join(narrative_parts)
    
    def _assess_macro_risk(
        self, 
        macro_indicators: MacroIndicators, 
        market_structure: Dict[str, Any]
    ) -> str:
        """Assess macro risk level"""
        
        risk_score = 0
        
        # VIX risk
        if macro_indicators.vix is not None:
            if macro_indicators.vix > 30:
                risk_score += 2
            elif macro_indicators.vix > 20:
                risk_score += 1
        
        # Yield curve risk
        if macro_indicators.treasury_10y is not None and macro_indicators.treasury_2y is not None:
            yield_spread = macro_indicators.treasury_10y - macro_indicators.treasury_2y
            if yield_spread < 0:
                risk_score += 2  # Inverted curve
            elif yield_spread < 0.5:
                risk_score += 1  # Flattening curve
        
        # Market performance risk
        if macro_indicators.spy_performance is not None:
            if macro_indicators.spy_performance < -2:
                risk_score += 2
            elif macro_indicators.spy_performance < -1:
                risk_score += 1
        
        # Determine risk level
        if risk_score >= 4:
            return "high"
        elif risk_score >= 2:
            return "medium"
        else:
            return "low"