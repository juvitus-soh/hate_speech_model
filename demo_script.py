"""
Interactive Demo for Cameroon Hate Speech Detection System

A comprehensive demonstration script that showcases all features of the hate speech detection system.
Includes various test scenarios, performance metrics, and interactive testing capabilities.

Features:
- Basic detection demos
- Keyword-triggered AI demonstration
- Real-time monitoring simulation
- Database features showcase
- Batch processing examples
- Interactive testing interface
"""

import time
import json
import asyncio
from datetime import datetime
from typing import List, Dict
import random

# Import our detection system
from hate_speech_detector import CameroonHateSpeechDetector, SocialMediaProcessor
from realtime_monitor import DatabaseManager, RealTimeHateSpeechMonitor


class DemoRunner:
    """Main demo runner class"""

    def __init__(self):
        print("🇨🇲 Initializing Cameroon Hate Speech Detection Demo...")
        self.detector = CameroonHateSpeechDetector()
        self.processor = SocialMediaProcessor(self.detector)
        self.db_manager = DatabaseManager()
        print("✅ Demo system initialized!\n")

    def run_demo(self):
        """Run the interactive demo"""
        print("=" * 60)
        print("🇨🇲 CAMEROON HATE SPEECH DETECTION SYSTEM DEMO")
        print("=" * 60)
        print("This demo showcases the keyword-triggered AI detection system")
        print("specifically designed for Cameroonian social media content.\n")

        demos = [
            ("Basic Detection", self.demo_basic_detection),
            ("Keyword-Triggered AI", self.demo_keyword_triggered_ai),
            ("Realtime Monitoring", self.demo_realtime_monitoring),
            ("Database Features", self.demo_database_features),
            ("Batch Processing", self.demo_batch_processing),
            ("Custom Scenarios", self.demo_custom_scenarios),
            ("Interactive Testing", self.interactive_test)
        ]

        while True:
            print("\n📋 Available Demos:")
            for i, (name, _) in enumerate(demos, 1):
                print(f"  {i}. {name}")
            print("  0. Exit Demo")

            try:
                choice = input("\n🎯 Select demo (0-7): ").strip()

                if choice == "0":
                    print("\n👋 Thank you for trying the Cameroon Hate Speech Detection Demo!")
                    break
                elif choice.isdigit() and 1 <= int(choice) <= len(demos):
                    demo_name, demo_func = demos[int(choice) - 1]
                    print(f"\n🚀 Running: {demo_name}")
                    demo_func()
                    input("\n⏸️  Press Enter to continue...")
                else:
                    print("❌ Invalid choice. Please select 0-7.")
            except KeyboardInterrupt:
                print("\n\n👋 Demo interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def demo_basic_detection(self):
        """Demonstrate basic hate speech detection"""
        print("\n🎯 DEMO: Basic Hate Speech Detection")
        print("-" * 40)
        print("Testing various types of Cameroonian social media content:\n")

        test_cases = [
            {
                "text": "Ces tontinards volent notre argent encore une fois",
                "expected": "High confidence ethnic hate speech"
            },
            {
                "text": "Beautiful sunset in Douala today, very peaceful",
                "expected": "Clean content, no hate speech detected"
            },
            {
                "text": "Paul Biya must go from power immediately",
                "expected": "Political dissent, medium severity"
            },
            {
                "text": "We nor go gree for this rigged election",
                "expected": "Pidgin political expression"
            },
            {
                "text": "Les anglofous causent toujours des problèmes",
                "expected": "Linguistic discrimination"
            },
            {
                "text": "The traditional dance was amazing at the festival",
                "expected": "Cultural content, clean"
            }
        ]

        total_time = 0

        for i, case in enumerate(test_cases, 1):
            print(f"Test {i}: '{case['text']}'")
            print(f"Expected: {case['expected']}")

            start_time = time.time()
            result = self.detector.detect_hate_speech(case['text'])
            processing_time = (time.time() - start_time) * 1000
            total_time += processing_time

            status = "🚨 HATE SPEECH" if result.is_hate_speech else "✅ CLEAN"
            print(f"Result: {status} (Confidence: {result.confidence:.1%})")
            print(f"Category: {result.category}, Severity: {result.severity}")

            if result.detected_keywords:
                print(f"Keywords detected: {result.detected_keywords}")

            print(f"Processing time: {processing_time:.1f}ms")
            print(f"Explanation: {result.explanation}")
            print("-" * 40)

        print(f"📊 Summary:")
        print(f"  Average processing time: {total_time / len(test_cases):.1f}ms")
        print(f"  Total test cases: {len(test_cases)}")

    def demo_keyword_triggered_ai(self):
        """
        Demonstrate the keyword-triggered AI analysis system
        """
        print("\n🎯 DEMO: Keyword-Triggered AI Analysis")
        print("-" * 40)
        print("This shows how keywords trigger AI analysis for context verification.\n")

        test_cases = [
            {
                "text": "Ces tontinards volent notre argent encore",
                "expected": "Keywords trigger AI analysis"
            },
            {
                "text": "Cette entreprise utilise le système de tontine",
                "expected": "Keywords trigger AI, but context shows normal usage"
            },
            {
                "text": "Paul Biya must go from power immediately",
                "expected": "Keywords trigger AI for political context"
            },
            {
                "text": "Beautiful sunset in Douala today",
                "expected": "No keywords, no AI analysis needed"
            },
            {
                "text": "We nor go gree for this rigged election",
                "expected": "Keywords trigger AI for Pidgin analysis"
            },
            {
                "text": "You are such an idiot and worthless person",
                "expected": "No keywords but AI may detect general toxicity"
            }
        ]

        for i, case in enumerate(test_cases, 1):
            print(f"Test {i}: {case['text']}")
            print(f"Expected: {case['expected']}")

            result = self.detector.detect_hate_speech(case['text'])

            # Show the process
            keywords = self.detector.keywords_detector.detect_keywords(case['text'])

            if keywords:
                print(f"🔍 Keywords found: {[k['keyword'] for k in keywords]}")
                print(f"🤖 AI analysis triggered: {result.explanation}")
            else:
                print(f"✅ No keywords detected")
                if "AI detected" in result.explanation:
                    print(f"🤖 AI analysis performed anyway: {result.explanation}")
                else:
                    print(f"⚡ Fast screening - no AI needed")

            status = "🚨 HATE SPEECH" if result.is_hate_speech else "✅ CLEAN"
            print(f"Result: {status} (Confidence: {result.confidence:.1%})")
            print(f"Category: {result.category}, Severity: {result.severity}")
            print("-" * 40)

        # Show efficiency statistics
        stats = self.detector.get_statistics()
        print(f"\n📊 Processing Efficiency:")
        print(f"  Total processed: {stats.get('total_processed', 0)}")
        print(f"  Keyword-triggered AI: {stats.get('keyword_triggered', 0)}")
        print(f"  AI-only detections: {stats.get('ai_only_detected', 0)}")
        print(f"  Clean content (fast): {stats.get('clean_content', 0)}")

        if stats.get('total_processed', 0) > 0:
            ai_usage = (stats.get('keyword_triggered', 0) + stats.get('ai_only_detected', 0)) / stats.get(
                'total_processed', 1)
            print(f"  AI usage rate: {ai_usage:.1%} (efficient!)")

    def demo_realtime_monitoring(self):
        """Demonstrate real-time monitoring capabilities"""
        print("\n🎯 DEMO: Real-time Monitoring Simulation")
        print("-" * 40)
        print("Simulating live social media stream processing...\n")

        # Sample stream data
        sample_posts = [
            {"text": "Ces sardinards sont vraiment corrompus", "platform": "twitter", "user_id": "user1"},
            {"text": "Beautiful ceremony in Bamenda today", "platform": "facebook", "user_id": "user2"},
            {"text": "Election truquée comme d'habitude", "platform": "twitter", "user_id": "user3"},
            {"text": "We nor go gree for dem lies", "platform": "facebook", "user_id": "user4"},
            {"text": "Amazing football match yesterday", "platform": "twitter", "user_id": "user5"},
            {"text": "Les anglofous toujours en train de causer des problèmes", "platform": "facebook",
             "user_id": "user6"},
            {"text": "University graduation ceremony was inspiring", "platform": "twitter", "user_id": "user7"},
            {"text": "Ces wadjo volent nos resources naturelles", "platform": "facebook", "user_id": "user8"}
        ]

        processed_count = 0
        hate_detected = 0

        print("📡 Processing incoming posts...")

        for post in sample_posts:
            processed_count += 1

            # Simulate real-time delay
            time.sleep(0.5)

            print(f"\n📩 Post {processed_count}: '{post['text'][:50]}...'")
            print(f"   Platform: {post['platform']}, User: {post['user_id']}")

            # Process with hate speech detector
            result = self.processor.process_post(post)

            if result:
                hate_detected += 1
                print(f"   🚨 HATE SPEECH DETECTED!")
                print(f"   Confidence: {result.confidence:.1%}")
                print(f"   Category: {result.category}")
                if result.detected_keywords:
                    print(f"   Keywords: {result.detected_keywords}")

                # Store in database
                metadata = {
                    'user_id': post['user_id'],
                    'platform': post['platform'],
                    'post_id': f'demo_post_{processed_count}'
                }
                self.db_manager.store_detection(result, metadata)
            else:
                print(f"   ✅ Clean content")

        print(f"\n📊 Stream Processing Summary:")
        print(f"  Posts processed: {processed_count}")
        print(f"  Hate speech detected: {hate_detected}")
        print(f"  Detection rate: {hate_detected / processed_count:.1%}")
        print(f"  Results stored in database: hate_speech_detections.db")

    def demo_database_features(self):
        """Demonstrate database storage and retrieval features"""
        print("\n🎯 DEMO: Database Features")
        print("-" * 40)

        # Get recent detections
        recent = self.db_manager.get_recent_detections(hours=24, hate_only=True)
        print(f"📊 Recent hate speech detections (24h): {len(recent)}")

        if recent:
            print("\nMost recent detections:")
            for i, detection in enumerate(recent[:5], 1):
                print(f"{i}. '{detection['text'][:60]}...'")
                print(f"   Confidence: {detection['confidence']:.1%}, Category: {detection['category']}")
                print(f"   Platform: {detection['platform']}, Time: {detection['timestamp']}")
        else:
            print("No recent detections found. Run real-time monitoring demo first.")

        # Get statistics
        stats = self.db_manager.get_statistics(days=7)
        print(f"\n📈 Database Statistics (7 days):")
        print(f"Period: {stats['period_days']} days")

        if stats['breakdown']:
            print("Breakdown by category and platform:")
            for item in stats['breakdown']:
                if item['total'] > 0:
                    print(
                        f"  {item['category']} ({item['platform']}): {item['hate_speech']}/{item['total']} hate speech")
        else:
            print("No statistics available yet.")

    def demo_batch_processing(self):
        """Demonstrate batch processing capabilities"""
        print("\n🎯 DEMO: Batch Processing")
        print("-" * 40)
        print("Processing multiple texts simultaneously...\n")

        batch_texts = [
            "Ces tontinards ne respectent vraiment rien",
            "Beautiful traditional music at the festival",
            "Élection encore truquée par le CPDM",
            "We nor go gree for dis government wahala",
            "Amazing sunset over Mount Cameroon",
            "Les sardinards corrompus volent tout",
            "University students study hard for exams",
            "Come no go back to your region",
            "Peaceful protest in downtown Yaoundé",
            "Ces anglofous causent toujours des problèmes"
        ]

        print(f"📦 Processing batch of {len(batch_texts)} texts...")

        start_time = time.time()
        results = self.detector.batch_detect(batch_texts)
        processing_time = time.time() - start_time

        # Analyze results
        hate_count = sum(1 for r in results if r.is_hate_speech)
        categories = {}
        total_keywords = 0

        print("\n📋 Results:")
        for i, result in enumerate(results, 1):
            status = "🚨 HATE" if result.is_hate_speech else "✅ CLEAN"
            print(f"{i:2d}. {status} ({result.confidence:.1%}) - '{batch_texts[i - 1][:40]}...'")

            if result.is_hate_speech:
                categories[result.category] = categories.get(result.category, 0) + 1
                total_keywords += len(result.detected_keywords)

        print(f"\n📊 Batch Processing Summary:")
        print(f"  Total texts processed: {len(batch_texts)}")
        print(f"  Hate speech detected: {hate_count}")
        print(f"  Detection rate: {hate_count / len(batch_texts):.1%}")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Average time per text: {processing_time / len(batch_texts) * 1000:.1f}ms")
        print(f"  Total keywords detected: {total_keywords}")

        if categories:
            print("  Categories detected:")
            for category, count in categories.items():
                print(f"    {category}: {count}")

    def demo_custom_scenarios(self):
        """Demonstrate custom scenarios and edge cases"""
        print("\n🎯 DEMO: Custom Scenarios & Edge Cases")
        print("-" * 40)
        print("Testing various challenging scenarios...\n")

        scenarios = {
            "Mixed Languages": [
                "Ces tontinards and these anglofous always fighting",
                "Biya must go mais les elections sont truquées",
                "We nor go gree but nous devons voter"
            ],
            "Misspellings": [
                "Ces tontonards sont corrompues",  # misspelled tontinards
                "Anglofou causent problemes",  # missing 's'
                "Sardinars volent argent"  # misspelled sardinards
            ],
            "Context Matters": [
                "The tontine system helps our community save money",  # legitimate use
                "Ces tontinards organize community development",  # could be neutral
                "Historical tensions between groups are documented"  # academic context
            ],
            "Borderline Cases": [
                "Government policies are questionable",
                "Some politicians are not effective",
                "There are regional differences in culture"
            ],
            "Election Period": [
                "Vote dem don buy with money and rice",
                "Opposition candidates arrested again",
                "International observers not allowed everywhere",
                "Normal voting process continues peacefully"
            ]
        }

        for scenario_name, texts in scenarios.items():
            print(f"🔍 Scenario: {scenario_name}")

            for text in texts:
                result = self.detector.detect_hate_speech(text)
                status = "🚨 HATE" if result.is_hate_speech else "✅ CLEAN"
                print(f"  {status} ({result.confidence:.1%}) - '{text}'")

                if result.detected_keywords:
                    print(f"    Keywords: {result.detected_keywords}")
                if result.explanation and "AI" in result.explanation:
                    print(f"    AI Analysis: {result.explanation.split(':')[-1].strip()}")

            print()

    def interactive_test(self):
        """Interactive testing interface"""
        print("\n🎯 DEMO: Interactive Testing")
        print("-" * 40)
        print("Enter your own text to test the hate speech detection system.")
        print("Type 'quit' to return to main menu.\n")

        while True:
            try:
                text = input("📝 Enter text to analyze: ").strip()

                if text.lower() in ['quit', 'exit', 'q']:
                    break

                if not text:
                    print("❌ Please enter some text.")
                    continue

                print(f"\n🔍 Analyzing: '{text}'")

                start_time = time.time()
                result = self.detector.detect_hate_speech(text)
                processing_time = (time.time() - start_time) * 1000

                # Show detailed results
                status = "🚨 HATE SPEECH DETECTED" if result.is_hate_speech else "✅ CLEAN CONTENT"
                print(f"\n{status}")
                print(f"Confidence: {result.confidence:.1%}")
                print(f"Category: {result.category}")
                print(f"Severity: {result.severity}")
                print(f"Processing time: {processing_time:.1f}ms")

                if result.detected_keywords:
                    print(f"Keywords detected: {result.detected_keywords}")

                print(f"Explanation: {result.explanation}")

                # Show keyword analysis
                keywords = self.detector.keywords_detector.detect_keywords(text)
                if keywords:
                    print(f"\nKeyword analysis:")
                    for kw in keywords:
                        print(f"  - '{kw['keyword']}' ({kw['category']}, {kw['severity']} severity)")
                else:
                    print("\nNo keywords detected in this text.")

                print("-" * 40)

            except KeyboardInterrupt:
                print("\n\nReturning to main menu...")
                break
            except Exception as e:
                print(f"❌ Error analyzing text: {e}")


def main():
    """Main function to run the demo"""
    try:
        demo = DemoRunner()
        demo.run_demo()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        print("Make sure all required dependencies are installed:")
        print("pip install torch transformers")


if __name__ == "__main__":
    main()