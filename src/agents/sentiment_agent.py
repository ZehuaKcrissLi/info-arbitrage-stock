import logging
import asyncio
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import aiohttp
import numpy as np
from textblob import TextBlob

from ..models.events import EnhancedMarketEvent
from ..models.analysis import SentimentAnalysis, SocialSentimentData
from ..config.settings import settings

logger = logging.getLogger(__name__)

class SentimentAnalysisAgent:
    """Agent for sentiment analysis from multiple sources"""
    
    def __init__(self):
        self.session = None
        self.sentiment_keywords = {
            'positive': [
                'bullish', 'buy', 'long', 'pump', 'moon', 'rally', 'surge', 'soar',
                'breakout', 'strong', 'good', 'great', 'excellent', 'amazing',
                'beat', 'exceed', 'outperform', 'upgrade', 'positive', 'gain'
            ],
            'negative': [
                'bearish', 'sell', 'short', 'dump', 'crash', 'tank', 'drop', 'fall',
                'breakdown', 'weak', 'bad', 'terrible', 'awful', 'horrible',
                'miss', 'underperform', 'downgrade', 'negative', 'loss', 'decline'
            ]
        }
        
    async def start(self):
        """Initialize the agent"""
        self.session = aiohttp.ClientSession()
        logger.info("Sentiment Analysis Agent started")
    
    async def stop(self):
        """Stop the agent"""
        if self.session:
            await self.session.close()
        logger.info("Sentiment Analysis Agent stopped")
    
    async def analyze(self, event: EnhancedMarketEvent) -> SentimentAnalysis:
        """Analyze market sentiment from multiple sources"""
        
        # Analyze news sentiment
        news_sentiment = await self._analyze_news_sentiment(event)
        
        # Analyze social media sentiment (placeholder for now)
        social_sentiment = await self._analyze_social_sentiment(event)
        
        # Analyze options sentiment (placeholder)
        options_sentiment = await self._analyze_options_sentiment(event)
        
        # Calculate aggregate sentiment
        aggregate_sentiment = self._calculate_aggregate_sentiment(
            news_sentiment, social_sentiment, options_sentiment
        )
        
        # Generate sentiment narrative
        narrative = await self._generate_sentiment_narrative(
            event, news_sentiment, social_sentiment, aggregate_sentiment
        )
        
        # Calculate confidence
        confidence = self._calculate_sentiment_confidence(
            news_sentiment, social_sentiment, aggregate_sentiment
        )
        
        return SentimentAnalysis(
            event_id=event.id,
            news_sentiment=news_sentiment,
            social_sentiment=social_sentiment,
            options_sentiment=options_sentiment,
            aggregate_sentiment=aggregate_sentiment,
            narrative=narrative,
            confidence=confidence
        )
    
    async def _analyze_news_sentiment(self, event: EnhancedMarketEvent) -> float:
        """Analyze sentiment from news content"""
        try:
            # Combine headline and summary for analysis
            text = f"{event.original_event.headline} {event.original_event.summary}"
            
            # Use multiple sentiment analysis methods
            
            # 1. TextBlob sentiment
            blob = TextBlob(text)
            textblob_sentiment = blob.sentiment.polarity
            
            # 2. Keyword-based sentiment
            keyword_sentiment = self._calculate_keyword_sentiment(text)
            
            # 3. Use the event's original sentiment as baseline
            event_sentiment = event.original_event.sentiment
            
            # Combine sentiments with weights
            combined_sentiment = (
                textblob_sentiment * 0.4 + 
                keyword_sentiment * 0.3 + 
                event_sentiment * 0.3
            )
            
            # Ensure within bounds
            return max(-1.0, min(1.0, combined_sentiment))
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            return event.original_event.sentiment  # Fallback
    
    def _calculate_keyword_sentiment(self, text: str) -> float:
        """Calculate sentiment based on keyword analysis"""
        text_lower = text.lower()
        
        positive_count = 0
        negative_count = 0
        
        # Count positive keywords
        for keyword in self.sentiment_keywords['positive']:
            positive_count += len(re.findall(r'\b' + keyword + r'\b', text_lower))
        
        # Count negative keywords
        for keyword in self.sentiment_keywords['negative']:
            negative_count += len(re.findall(r'\b' + keyword + r'\b', text_lower))
        
        # Calculate sentiment score
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return 0.0
        
        sentiment_score = (positive_count - negative_count) / total_sentiment_words
        
        # Scale to [-1, 1] range
        return max(-1.0, min(1.0, sentiment_score))
    
    async def _analyze_social_sentiment(self, event: EnhancedMarketEvent) -> Dict[str, SocialSentimentData]:
        """Analyze sentiment from social media sources"""
        social_sentiment = {}
        
        for ticker in event.original_event.tickers:
            try:
                # Fetch social media mentions (placeholder implementation)
                twitter_mentions = await self._fetch_twitter_mentions(ticker)
                reddit_mentions = await self._fetch_reddit_mentions(ticker)
                
                # Analyze sentiment
                twitter_scores = []
                for mention in twitter_mentions:
                    score = self._analyze_text_sentiment(mention.get('text', ''))
                    twitter_scores.append(score)
                
                reddit_scores = []
                for mention in reddit_mentions:
                    score = self._analyze_text_sentiment(mention.get('text', ''))
                    reddit_scores.append(score)
                
                # Calculate averages
                twitter_sentiment = np.mean(twitter_scores) if twitter_scores else 0.0
                reddit_sentiment = np.mean(reddit_scores) if reddit_scores else 0.0
                
                # Extract trending hashtags (simplified)
                trending_hashtags = self._extract_trending_hashtags(twitter_mentions)
                
                social_sentiment[ticker] = SocialSentimentData(
                    twitter_sentiment=twitter_sentiment,
                    reddit_sentiment=reddit_sentiment,
                    twitter_volume=len(twitter_mentions),
                    reddit_volume=len(reddit_mentions),
                    trending_hashtags=trending_hashtags
                )
                
            except Exception as e:
                logger.error(f"Error analyzing social sentiment for {ticker}: {e}")
                social_sentiment[ticker] = SocialSentimentData()
        
        return social_sentiment
    
    async def _fetch_twitter_mentions(self, ticker: str) -> List[Dict[str, Any]]:
        """Fetch Twitter mentions for a ticker (placeholder)"""
        # This is a placeholder implementation
        # In production, you'd use Twitter API v2
        
        # Return mock data for demonstration
        mock_mentions = [
            {"text": f"${ticker} is looking bullish today! Great earnings beat.", "engagement": 100},
            {"text": f"Thinking of buying more ${ticker} on this dip", "engagement": 50},
            {"text": f"${ticker} to the moon! 🚀", "engagement": 200}
        ]
        
        return mock_mentions
    
    async def _fetch_reddit_mentions(self, ticker: str) -> List[Dict[str, Any]]:
        """Fetch Reddit mentions for a ticker (placeholder)"""
        # This is a placeholder implementation
        # In production, you'd use Reddit API (PRAW)
        
        # Return mock data for demonstration
        mock_mentions = [
            {"text": f"DD on {ticker}: Strong fundamentals and great technicals", "upvotes": 500},
            {"text": f"Should I hold my {ticker} position?", "upvotes": 20},
            {"text": f"{ticker} earnings call was impressive", "upvotes": 150}
        ]
        
        return mock_mentions
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of a single text"""
        try:
            # Use TextBlob for simplicity
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            
            # Enhance with keyword analysis
            keyword_sentiment = self._calculate_keyword_sentiment(text)
            
            # Combine with equal weights
            combined = (sentiment + keyword_sentiment) / 2
            
            return max(-1.0, min(1.0, combined))
            
        except:
            return 0.0
    
    def _extract_trending_hashtags(self, mentions: List[Dict[str, Any]]) -> List[str]:
        """Extract trending hashtags from mentions"""
        hashtag_counts = {}
        
        for mention in mentions:
            text = mention.get('text', '')
            hashtags = re.findall(r'#(\w+)', text)
            
            for hashtag in hashtags:
                hashtag_lower = hashtag.lower()
                hashtag_counts[hashtag_lower] = hashtag_counts.get(hashtag_lower, 0) + 1
        
        # Return top 5 hashtags
        sorted_hashtags = sorted(hashtag_counts.items(), key=lambda x: x[1], reverse=True)
        return [hashtag for hashtag, count in sorted_hashtags[:5]]
    
    async def _analyze_options_sentiment(self, event: EnhancedMarketEvent) -> Optional[float]:
        """Analyze options flow sentiment (placeholder)"""
        # This would analyze options flow data if available
        # For now, return None as placeholder
        return None
    
    def _calculate_aggregate_sentiment(
        self, 
        news_sentiment: float, 
        social_sentiment: Dict[str, SocialSentimentData], 
        options_sentiment: Optional[float]
    ) -> float:
        """Calculate aggregate sentiment from all sources"""
        
        sentiment_components = []
        weights = []
        
        # News sentiment (high weight)
        sentiment_components.append(news_sentiment)
        weights.append(0.5)
        
        # Social sentiment (medium weight)
        if social_sentiment:
            social_scores = []
            for ticker_data in social_sentiment.values():
                # Combine Twitter and Reddit sentiment
                if ticker_data.twitter_volume > 0 and ticker_data.reddit_volume > 0:
                    combined_social = (
                        ticker_data.twitter_sentiment * 0.6 + 
                        ticker_data.reddit_sentiment * 0.4
                    )
                    social_scores.append(combined_social)
                elif ticker_data.twitter_volume > 0:
                    social_scores.append(ticker_data.twitter_sentiment)
                elif ticker_data.reddit_volume > 0:
                    social_scores.append(ticker_data.reddit_sentiment)
            
            if social_scores:
                avg_social_sentiment = np.mean(social_scores)
                sentiment_components.append(avg_social_sentiment)
                weights.append(0.3)
        
        # Options sentiment (low weight)
        if options_sentiment is not None:
            sentiment_components.append(options_sentiment)
            weights.append(0.2)
        
        # Calculate weighted average
        if sentiment_components:
            total_weight = sum(weights[:len(sentiment_components)])
            weighted_sum = sum(
                sentiment * weight 
                for sentiment, weight in zip(sentiment_components, weights[:len(sentiment_components)])
            )
            return weighted_sum / total_weight
        
        return 0.0
    
    async def _generate_sentiment_narrative(
        self,
        event: EnhancedMarketEvent,
        news_sentiment: float,
        social_sentiment: Dict[str, SocialSentimentData],
        aggregate_sentiment: float
    ) -> str:
        """Generate sentiment analysis narrative"""
        
        narrative_parts = []
        
        # Overall sentiment
        if aggregate_sentiment > 0.3:
            narrative_parts.append("Market sentiment is strongly positive.")
        elif aggregate_sentiment > 0.1:
            narrative_parts.append("Market sentiment is mildly positive.")
        elif aggregate_sentiment < -0.3:
            narrative_parts.append("Market sentiment is strongly negative.")
        elif aggregate_sentiment < -0.1:
            narrative_parts.append("Market sentiment is mildly negative.")
        else:
            narrative_parts.append("Market sentiment is neutral.")
        
        # News sentiment analysis
        if news_sentiment > 0.2:
            narrative_parts.append("News coverage is predominantly positive.")
        elif news_sentiment < -0.2:
            narrative_parts.append("News coverage shows negative tone.")
        else:
            narrative_parts.append("News coverage appears balanced.")
        
        # Social media analysis
        if social_sentiment:
            total_volume = sum(
                data.twitter_volume + data.reddit_volume 
                for data in social_sentiment.values()
            )
            
            if total_volume > 50:
                narrative_parts.append("High social media engagement detected.")
            elif total_volume > 10:
                narrative_parts.append("Moderate social media discussion observed.")
            else:
                narrative_parts.append("Limited social media activity.")
            
            # Check for trending hashtags
            all_hashtags = []
            for data in social_sentiment.values():
                all_hashtags.extend(data.trending_hashtags)
            
            if all_hashtags:
                hashtag_text = ", ".join(all_hashtags[:3])
                narrative_parts.append(f"Trending hashtags: {hashtag_text}.")
        
        # Event context
        event_type = event.original_event.event_type.value
        if event_type in ["earnings", "merger", "fda_approval"]:
            if aggregate_sentiment > 0:
                narrative_parts.append(f"Positive sentiment aligns well with {event_type} announcement.")
            else:
                narrative_parts.append(f"Market reaction to {event_type} appears cautious.")
        
        return " ".join(narrative_parts)
    
    def _calculate_sentiment_confidence(
        self,
        news_sentiment: float,
        social_sentiment: Dict[str, SocialSentimentData],
        aggregate_sentiment: float
    ) -> float:
        """Calculate confidence in sentiment analysis"""
        
        confidence = 0.0
        
        # Base confidence
        confidence += 0.3
        
        # News sentiment clarity (strong sentiment = higher confidence)
        news_strength = abs(news_sentiment)
        confidence += news_strength * 0.3
        
        # Social media volume and consensus
        if social_sentiment:
            total_volume = sum(
                data.twitter_volume + data.reddit_volume 
                for data in social_sentiment.values()
            )
            
            # Volume boost
            if total_volume > 100:
                confidence += 0.2
            elif total_volume > 20:
                confidence += 0.1
            
            # Consensus boost (if social sentiment aligns with news)
            social_scores = []
            for data in social_sentiment.values():
                if data.twitter_volume > 0 or data.reddit_volume > 0:
                    avg_social = (data.twitter_sentiment + data.reddit_sentiment) / 2
                    social_scores.append(avg_social)
            
            if social_scores:
                avg_social = np.mean(social_scores)
                # If social and news sentiment agree, boost confidence
                if (news_sentiment > 0 and avg_social > 0) or (news_sentiment < 0 and avg_social < 0):
                    confidence += 0.2
        
        # Aggregate sentiment strength
        aggregate_strength = abs(aggregate_sentiment)
        confidence += aggregate_strength * 0.2
        
        return min(1.0, confidence)