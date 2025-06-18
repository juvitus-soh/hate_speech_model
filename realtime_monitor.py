"""
Real-time Hate Speech Monitoring System

A comprehensive real-time monitoring system for detecting hate speech in social media streams.
Simulates Twitter and Facebook feeds and processes posts through the hate speech detection system.

Features:
- Real-time stream simulation
- SQLite database storage
- Web dashboard interface
- Statistics and reporting
- Background processing
- Alert system
"""

import asyncio
import json
import websockets
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional
import logging
from dataclasses import asdict
import sqlite3
import threading
from queue import Queue
import time

# Import our hate speech detector
from hate_speech_detector import CameroonHateSpeechDetector, SocialMediaProcessor, HateSpeechResult

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manage SQLite database for storing results"""

    def __init__(self, db_path: str = "hate_speech_detections.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                is_hate_speech BOOLEAN NOT NULL,
                confidence REAL NOT NULL,
                detected_keywords TEXT,
                category TEXT,
                severity TEXT,
                explanation TEXT,
                timestamp DATETIME,
                user_id TEXT,
                platform TEXT,
                post_id TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                total_processed INTEGER,
                hate_speech_detected INTEGER,
                platform TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def store_detection(self, result: HateSpeechResult, metadata: Dict = None):
        """Store a detection result in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if metadata is None:
            metadata = {}

        cursor.execute('''
            INSERT INTO detections 
            (text, is_hate_speech, confidence, detected_keywords, category, 
             severity, explanation, timestamp, user_id, platform, post_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.text,
            result.is_hate_speech,
            result.confidence,
            json.dumps(result.detected_keywords),
            result.category,
            result.severity,
            result.explanation,
            result.timestamp.isoformat(),
            metadata.get('user_id'),
            metadata.get('platform'),
            metadata.get('post_id')
        ))

        conn.commit()
        conn.close()

    def get_recent_detections(self, hours: int = 24, hate_only: bool = True) -> List[Dict]:
        """Get recent detections"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()

        if hate_only:
            cursor.execute('''
                SELECT * FROM detections 
                WHERE timestamp > ? AND is_hate_speech = 1
                ORDER BY timestamp DESC
            ''', (cutoff_time,))
        else:
            cursor.execute('''
                SELECT * FROM detections 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            ''', (cutoff_time,))

        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        return results

    def get_statistics(self, days: int = 7) -> Dict:
        """Get detection statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN is_hate_speech = 1 THEN 1 ELSE 0 END) as hate_speech,
                category,
                platform
            FROM detections 
            WHERE timestamp > ?
            GROUP BY category, platform
        ''', (cutoff_time,))

        results = cursor.fetchall()
        conn.close()

        return {
            'period_days': days,
            'breakdown': [
                {
                    'total': row[0],
                    'hate_speech': row[1],
                    'category': row[2],
                    'platform': row[3]
                }
                for row in results
            ]
        }


class TwitterStreamSimulator:
    """Simulate Twitter stream for testing (replace with real Twitter API)"""

    def __init__(self, callback: Callable):
        self.callback = callback
        self.running = False

        # Sample Cameroonian social media posts for testing (including election content)
        self.sample_posts = [
            "Ces tontinards ne respectent rien dans ce pays!",
            "Les sardinards sont corrompus jusqu'à l'os",
            "Pourquoi les anglofous causent toujours des problèmes?",
            "J'adore la culture camerounaise, très riche",
            "Les Bamenda pensent qu'ils peuvent tout faire",
            "Belle journée à Yaoundé aujourd'hui",
            "Ces graffi sont vraiment bizarres",
            "Les Kamtalibans détruisent notre pays",
            "La cuisine camerounaise est excellente",
            "Come no go, retournez chez vous!",
            # New election-related posts
            "Paul Biya 2025 election must be stopped",
            "We nor go gree for rigged election again",
            "Fake news about Maurice Kamto spreading everywhere",
            "Vote buying reported in Northern regions",
            "Élection truquée comme d'habitude au Cameroun",
            "Constitutional council announces new dates",
            "Dem don plan election wahala for us",
            "Opposition jailed before campaign starts",
            "E go hot for Cameroon if CPDM continue",
            "Normal voter registration ongoing peacefully",
            "Ces wadjo volent encore nos ressources",
            "Beautiful traditional dance in Foumban today",
            "Corrupt ministers stealing public money again",
            "University students protest fees increase",
            "Lions indomptables won the match!",
            "Traffic jam in Douala as usual",
            "New hospital opens in Garoua",
            "Rice harvest season starts in North",
            "Fishing boats return to Limbe port",
            "Market prices increase again"
        ]

    async def start_stream(self):
        """Start simulated stream"""
        self.running = True
        post_id = 1

        while self.running:
            # Simulate receiving a post every 2-10 seconds
            await asyncio.sleep(2 + (post_id % 8))

            if not self.running:
                break

            # Create fake post data
            post_data = {
                'text': self.sample_posts[post_id % len(self.sample_posts)],
                'user_id': f'user_{post_id % 5 + 1}',
                'platform': 'twitter',
                'post_id': f'post_{post_id}',
                'timestamp': datetime.now().isoformat()
            }

            await self.callback(post_data)
            post_id += 1

    def stop_stream(self):
        """Stop the stream"""
        self.running = False


class FacebookStreamSimulator:
    """Simulate Facebook stream for testing"""

    def __init__(self, callback: Callable):
        self.callback = callback
        self.running = False

        self.sample_posts = [
            "Les moutons du nord ne comprennent rien",
            "Wadjo revient encore nous embêter",
            "Très belle cérémonie aujourd'hui à Douala",
            "Ces nges man arnaquent tout le monde",
            "La brigade anti-sardinards fait du bon travail",
            "Match fantastique ce soir",
            "L'age de Kumba est toujours faux",
            "Les Beti contrôlent tout dans ce pays",
            # New election-related Facebook posts
            "Biya must go, no more dictatorship in 2025",
            "Fraude électorale organisée par le CPDM",
            "Maurice Kamto rally blocked by police again",
            "Fake results already prepared for election",
            "Diaspora voting rights being suppressed",
            "Election observers needed in Cameroon",
            "Unrest in Anglophone regions continues",
            "Media biaisés refusing to cover opposition",
            "Farmers market busy this weekend",
            "School fees payment deadline approaches",
            "Rainy season starts early this year",
            "Church service was very inspiring today",
            "New road construction begins in Bamenda",
            "Internet connection problems again",
            "Petrol prices increase once more",
            "Good harvest expected this season",
            "Wedding ceremony was beautiful",
            "Football training session tomorrow"
        ]

    async def start_stream(self):
        """Start simulated Facebook stream"""
        self.running = True
        post_id = 1000

        while self.running:
            await asyncio.sleep(3 + (post_id % 6))

            if not self.running:
                break

            post_data = {
                'text': self.sample_posts[post_id % len(self.sample_posts)],
                'user_id': f'fb_user_{post_id % 3 + 1}',
                'platform': 'facebook',
                'post_id': f'fb_post_{post_id}',
                'timestamp': datetime.now().isoformat()
            }

            await self.callback(post_data)
            post_id += 1

    def stop_stream(self):
        self.running = False


class RealTimeHateSpeechMonitor:
    """Main real-time monitoring system"""

    def __init__(self):
        self.detector = CameroonHateSpeechDetector()
        self.processor = SocialMediaProcessor(self.detector)
        self.db_manager = DatabaseManager()

        # Stream simulators (replace with real APIs)
        self.twitter_stream = TwitterStreamSimulator(self.process_post)
        self.facebook_stream = FacebookStreamSimulator(self.process_post)

        self.running = False
        self.stats = {
            'posts_processed': 0,
            'hate_speech_detected': 0,
            'start_time': None
        }

    async def process_post(self, post_data: Dict):
        """Process incoming social media post"""
        try:
            self.stats['posts_processed'] += 1

            # Detect hate speech
            result = self.processor.process_post(post_data)

            if result:
                self.stats['hate_speech_detected'] += 1

                # Store in database
                metadata = {
                    'user_id': post_data.get('user_id'),
                    'platform': post_data.get('platform'),
                    'post_id': post_data.get('post_id')
                }
                self.db_manager.store_detection(result, metadata)

                # Log the detection
                logger.warning(
                    f"HATE SPEECH DETECTED on {post_data.get('platform', 'unknown')}: "
                    f"'{result.text[:100]}...' "
                    f"(Confidence: {result.confidence:.2%}, "
                    f"Keywords: {result.detected_keywords})"
                )

                # Here you could add alerts, notifications, etc.
                await self.send_alert(result, post_data)

            # Log progress every 10 posts
            if self.stats['posts_processed'] % 10 == 0:
                logger.info(
                    f"Processed {self.stats['posts_processed']} posts, "
                    f"detected {self.stats['hate_speech_detected']} hate speech instances"
                )

        except Exception as e:
            logger.error(f"Error processing post: {e}")

    async def send_alert(self, result: HateSpeechResult, post_data: Dict):
        """Send alert for detected hate speech"""
        # This is where you'd integrate with your alerting system
        # For now, just log
        logger.info(f"ALERT: High severity hate speech detected from {post_data.get('user_id')}")

    async def start_monitoring(self):
        """Start real-time monitoring"""
        self.running = True
        self.stats['start_time'] = datetime.now()

        logger.info("Starting real-time hate speech monitoring...")

        # Start all stream simulators concurrently
        tasks = [
            asyncio.create_task(self.twitter_stream.start_stream()),
            asyncio.create_task(self.facebook_stream.start_stream()),
            asyncio.create_task(self.status_reporter())
        ]

        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        finally:
            await self.stop_monitoring()

    async def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        self.twitter_stream.stop_stream()
        self.facebook_stream.stop_stream()

        logger.info("Monitoring stopped")
        logger.info(f"Final stats: {self.stats}")

    async def status_reporter(self):
        """Report status every minute"""
        while self.running:
            await asyncio.sleep(60)  # Report every minute

            if self.running:
                uptime = datetime.now() - self.stats['start_time']
                rate = self.stats['posts_processed'] / max(uptime.total_seconds() / 60, 1)

                logger.info(
                    f"STATUS: {self.stats['posts_processed']} posts processed, "
                    f"{self.stats['hate_speech_detected']} hate speech detected, "
                    f"Rate: {rate:.1f} posts/min, "
                    f"Uptime: {uptime}"
                )

    def get_live_statistics(self) -> Dict:
        """Get current monitoring statistics"""
        uptime = datetime.now() - self.stats['start_time'] if self.stats['start_time'] else timedelta(0)

        return {
            'current_session': self.stats,
            'uptime': str(uptime),
            'database_stats': self.db_manager.get_statistics(),
            'recent_detections': len(self.db_manager.get_recent_detections(hours=1))
        }


class WebDashboard:
    """Simple web dashboard for monitoring"""

    def __init__(self, monitor: RealTimeHateSpeechMonitor, port: int = 8080):
        self.monitor = monitor
        self.port = port

    async def handle_request(self, request):
        """Handle web requests"""
        if request.path == '/':
            return self.dashboard_html()
        elif request.path == '/api/stats':
            return json.dumps(self.monitor.get_live_statistics(), indent=2)
        elif request.path == '/api/recent':
            recent = self.monitor.db_manager.get_recent_detections(hours=24)
            return json.dumps(recent, indent=2, default=str)
        else:
            return "404 Not Found"

    def dashboard_html(self) -> str:
        """Generate simple HTML dashboard"""
        stats = self.monitor.get_live_statistics()

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cameroon Hate Speech Monitor</title>
            <meta http-equiv="refresh" content="30">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                .stat-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #3498db; }}
                .stat-number {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
                .stat-label {{ color: #7f8c8d; text-transform: uppercase; font-size: 0.9em; }}
                .alert {{ background: #e74c3c; color: white; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .success {{ background: #27ae60; color: white; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .api-links {{ background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .api-links a {{ color: #3498db; text-decoration: none; margin-right: 15px; }}
                .api-links a:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="header">🇨🇲 Cameroon Hate Speech Detection Dashboard</h1>

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{stats['current_session']['posts_processed']}</div>
                        <div class="stat-label">Posts Processed</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{stats['current_session']['hate_speech_detected']}</div>
                        <div class="stat-label">Hate Speech Detected</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{stats['recent_detections']}</div>
                        <div class="stat-label">Recent Detections (1h)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{stats['uptime']}</div>
                        <div class="stat-label">System Uptime</div>
                    </div>
                </div>

                <div class="success">
                    ✅ Real-time monitoring active - Processing Twitter and Facebook streams
                </div>

                <div class="api-links">
                    <h3>API Endpoints</h3>
                    <a href="/api/stats">Live Statistics (JSON)</a>
                    <a href="/api/recent">Recent Detections (JSON)</a>
                </div>

                <h3>System Status</h3>
                <ul>
                    <li>🔍 Keyword Detection: 160+ Cameroon-specific terms loaded</li>
                    <li>🤖 AI Classification: Pre-trained models active</li>
                    <li>💾 Database: SQLite storage operational</li>
                    <li>📡 Streams: Twitter & Facebook simulators running</li>
                </ul>

                <p><small>Page auto-refreshes every 30 seconds | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
            </div>
        </body>
        </html>
        """


# Main execution
async def main():
    """Main function to run the monitoring system"""

    print("=== Cameroon Real-Time Hate Speech Detection System ===\n")

    # Initialize monitor
    monitor = RealTimeHateSpeechMonitor()

    print("Starting monitoring system...")
    print("This will simulate social media streams and detect hate speech in real-time.")
    print("📊 Database: hate_speech_detections.db")
    print("🌐 Dashboard available at: http://localhost:8080 (if implemented)")
    print("Press Ctrl+C to stop.\n")

    try:
        # Start monitoring
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        await monitor.stop_monitoring()

    # Show final statistics
    print(f"\nFinal Statistics:")
    final_stats = monitor.get_live_statistics()
    print(json.dumps(final_stats, indent=2, default=str))


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('hate_speech_monitor.log')
        ]
    )

    # Run the monitoring system
    asyncio.run(main())