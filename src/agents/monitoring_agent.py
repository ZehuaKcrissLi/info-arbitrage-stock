import asyncio
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import aiohttp
import feedparser
from bs4 import BeautifulSoup

from ..models.events import MarketEvent, EventType, NewsItem, SocialMediaMention
from ..config.settings import settings

logger = logging.getLogger(__name__)

class InformationMonitoringAgent:
    def __init__(self):
        self.finnhub_api_key = settings.finnhub_api_key
        self.newsapi_key = settings.newsapi_key
        self.session = None
        self.is_running = False
        
        # Tracking keywords for different event types
        self.event_keywords = {
            EventType.EARNINGS: ["earnings", "quarterly results", "q1", "q2", "q3", "q4", "revenue", "eps"],
            EventType.MERGER: ["merger", "acquisition", "buyout", "takeover", "deal"],
            EventType.FDA_APPROVAL: ["fda approval", "drug approval", "clinical trial", "phase"],
            EventType.ANALYST_UPGRADE: ["upgrade", "buy rating", "outperform"],
            EventType.ANALYST_DOWNGRADE: ["downgrade", "sell rating", "underperform"]
        }
        
        # Ticker extraction pattern
        self.ticker_pattern = re.compile(r'\$([A-Z]{1,5})')
        
    async def start(self):
        """Start the monitoring agent"""
        self.is_running = True
        self.session = aiohttp.ClientSession()
        logger.info("Information Monitoring Agent started")
        
        # Start monitoring tasks
        await self.monitor_continuous()
    
    async def stop(self):
        """Stop the monitoring agent"""
        self.is_running = False
        if self.session:
            await self.session.close()
        logger.info("Information Monitoring Agent stopped")
    
    async def monitor_continuous(self):
        """Main monitoring loop"""
        tasks = [
            self.monitor_news(),
            self.monitor_sec_filings(),
            self.monitor_macro_events()
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
    
    async def monitor_news(self):
        """Monitor news sources for relevant updates"""
        while self.is_running:
            try:
                # Monitor Finnhub news
                finnhub_news = await self._fetch_finnhub_news()
                for news_item in finnhub_news:
                    if self._is_relevant_news(news_item):
                        event = self._structure_news_event(news_item)
                        await self._publish_event(event)
                
                # Monitor NewsAPI
                newsapi_news = await self._fetch_newsapi_news()
                for news_item in newsapi_news:
                    if self._is_relevant_news(news_item):
                        event = self._structure_news_event(news_item)
                        await self._publish_event(event)
                
                await asyncio.sleep(settings.trading.news_check_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring news: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def monitor_sec_filings(self):
        """Monitor SEC filings for relevant updates"""
        while self.is_running:
            try:
                # This is a simplified implementation
                # In production, you'd want to use SEC EDGAR RSS feeds
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring SEC filings: {e}")
                await asyncio.sleep(300)
    
    async def monitor_macro_events(self):
        """Monitor macro economic events"""
        while self.is_running:
            try:
                # This would monitor economic calendars, Fed announcements, etc.
                await asyncio.sleep(600)  # Check every 10 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring macro events: {e}")
                await asyncio.sleep(600)
    
    async def _fetch_finnhub_news(self) -> List[NewsItem]:
        """Fetch news from Finnhub API"""
        news_items = []
        try:
            url = "https://finnhub.io/api/v1/news"
            params = {
                "category": "general",
                "token": self.finnhub_api_key
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    for item in data:
                        news_item = NewsItem(
                            source="finnhub",
                            timestamp=datetime.fromtimestamp(item.get("datetime", 0)),
                            headline=item.get("headline", ""),
                            content=item.get("summary", ""),
                            url=item.get("url"),
                            tickers=self._extract_tickers(item.get("headline", "") + " " + item.get("summary", ""))
                        )
                        news_items.append(news_item)
                        
        except Exception as e:
            logger.error(f"Error fetching Finnhub news: {e}")
            
        return news_items
    
    async def _fetch_newsapi_news(self) -> List[NewsItem]:
        """Fetch news from NewsAPI"""
        news_items = []
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": "stocks OR earnings OR merger OR acquisition",
                "language": "en",
                "sortBy": "publishedAt",
                "apiKey": self.newsapi_key,
                "pageSize": 20
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    for article in data.get("articles", []):
                        news_item = NewsItem(
                            source="newsapi",
                            timestamp=datetime.fromisoformat(
                                article.get("publishedAt", "").replace("Z", "+00:00")
                            ),
                            headline=article.get("title", ""),
                            content=article.get("description", "") or article.get("content", ""),
                            url=article.get("url"),
                            tickers=self._extract_tickers(
                                article.get("title", "") + " " + 
                                (article.get("description", "") or "")
                            )
                        )
                        news_items.append(news_item)
                        
        except Exception as e:
            logger.error(f"Error fetching NewsAPI news: {e}")
            
        return news_items
    
    def _extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from text"""
        # Look for $TICKER format
        tickers = self.ticker_pattern.findall(text.upper())
        
        # Also look for common company names and convert to tickers
        company_ticker_map = {
            "APPLE": "AAPL",
            "MICROSOFT": "MSFT",
            "GOOGLE": "GOOGL",
            "ALPHABET": "GOOGL",
            "AMAZON": "AMZN",
            "TESLA": "TSLA",
            "META": "META",
            "FACEBOOK": "META",
            "NVIDIA": "NVDA"
        }
        
        text_upper = text.upper()
        for company, ticker in company_ticker_map.items():
            if company in text_upper:
                tickers.append(ticker)
        
        return list(set(tickers))  # Remove duplicates
    
    def _is_relevant_news(self, news_item: NewsItem) -> bool:
        """Check if news item is relevant for trading"""
        # Must have tickers
        if not news_item.tickers:
            return False
        
        # Check for relevant keywords
        text = (news_item.headline + " " + news_item.content).lower()
        
        for event_type, keywords in self.event_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return True
        
        return False
    
    def _determine_event_type(self, text: str) -> EventType:
        """Determine the type of event based on text content"""
        text_lower = text.lower()
        
        for event_type, keywords in self.event_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return event_type
        
        return EventType.NEWS  # Default
    
    def _calculate_sentiment(self, text: str) -> float:
        """Basic sentiment calculation"""
        positive_words = [
            "beat", "exceeds", "strong", "growth", "profit", "gain", "surge", 
            "rally", "upgrade", "bullish", "positive", "up", "rise", "boost"
        ]
        negative_words = [
            "miss", "below", "weak", "decline", "loss", "drop", "fall", 
            "downgrade", "bearish", "negative", "down", "crash", "plunge"
        ]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text_lower.split())
        if total_words == 0:
            return 0.0
        
        sentiment_score = (positive_count - negative_count) / max(total_words / 10, 1)
        return max(-1.0, min(1.0, sentiment_score))
    
    def _assess_urgency(self, news_item: NewsItem) -> int:
        """Assess urgency of news item (1-10 scale)"""
        urgency = 5  # Default medium urgency
        
        text = (news_item.headline + " " + news_item.content).lower()
        
        # High urgency keywords
        high_urgency_keywords = [
            "breaking", "urgent", "halt", "suspended", "emergency", 
            "bankruptcy", "lawsuit", "fda approval", "merger"
        ]
        
        # Market hours consideration
        now = datetime.now()
        market_hours = 9 <= now.hour <= 16
        
        for keyword in high_urgency_keywords:
            if keyword in text:
                urgency = 8 if market_hours else 7
                break
        
        # Earnings during market hours
        if any(word in text for word in ["earnings", "quarterly"]):
            urgency = 7 if market_hours else 6
        
        return urgency
    
    def _structure_news_event(self, news_item: NewsItem) -> MarketEvent:
        """Convert news item to structured market event"""
        text = news_item.headline + " " + news_item.content
        
        return MarketEvent(
            source=news_item.source,
            timestamp=news_item.timestamp,
            tickers=news_item.tickers,
            event_type=self._determine_event_type(text),
            headline=news_item.headline,
            summary=news_item.content[:500],  # Limit summary length
            sentiment=self._calculate_sentiment(text),
            urgency=self._assess_urgency(news_item),
            raw_data={
                "url": news_item.url,
                "full_content": news_item.content
            }
        )
    
    async def _publish_event(self, event: MarketEvent):
        """Publish event to event bus"""
        # This would publish to Redis/event bus in real implementation
        logger.info(f"Market event detected: {event.headline} - Tickers: {event.tickers}")
        
        # For now, we'll just log the event
        # In the full implementation, this would publish to the event bus
        # await self.event_bus.publish("market_events", event)

class SECFilingMonitor:
    """Monitor SEC filings for relevant updates"""
    
    def __init__(self):
        self.rss_feeds = [
            "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&CIK=&type=8-K&company=&dateb=&owner=include&start=0&count=40&output=atom",
            "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&CIK=&type=10-K&company=&dateb=&owner=include&start=0&count=40&output=atom",
            "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&CIK=&type=10-Q&company=&dateb=&owner=include&start=0&count=40&output=atom"
        ]
    
    async def monitor_filings(self):
        """Monitor SEC RSS feeds for new filings"""
        while True:
            try:
                for feed_url in self.rss_feeds:
                    await self._check_feed(feed_url)
                
                await asyncio.sleep(1800)  # Check every 30 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring SEC filings: {e}")
                await asyncio.sleep(1800)
    
    async def _check_feed(self, feed_url: str):
        """Check a specific RSS feed for new filings"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(feed_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        for entry in feed.entries[:5]:  # Check latest 5 entries
                            # Process SEC filing entry
                            logger.info(f"New SEC filing: {entry.title}")
                            
        except Exception as e:
            logger.error(f"Error checking SEC feed {feed_url}: {e}")

class SocialMediaAggregator:
    """Aggregate social media mentions and sentiment"""
    
    def __init__(self):
        # This would require proper API keys and setup
        pass
    
    async def fetch_twitter_mentions(self, ticker: str) -> List[SocialMediaMention]:
        """Fetch Twitter mentions for a ticker"""
        # Placeholder implementation
        # In production, this would use Twitter API v2
        return []
    
    async def fetch_reddit_mentions(self, ticker: str) -> List[SocialMediaMention]:
        """Fetch Reddit mentions for a ticker"""
        # Placeholder implementation  
        # In production, this would use Reddit API (PRAW)
        return []