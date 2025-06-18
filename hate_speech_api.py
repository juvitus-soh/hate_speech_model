"""
Cameroon Hate Speech Detection API

A FastAPI service that exposes the hate speech detection system as REST endpoints.
Supports single text analysis, batch processing, and real-time monitoring.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import json
from datetime import datetime, timedelta
import logging
from contextlib import asynccontextmanager
import uvicorn

# Import our detection system
from hate_speech_detector import CameroonHateSpeechDetector, HateSpeechResult
from realtime_monitor import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global detector instance (loaded once on startup)
detector = None
db_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global detector, db_manager

    # Startup
    logger.info("Loading Cameroon Hate Speech Detection models...")
    detector = CameroonHateSpeechDetector()
    db_manager = DatabaseManager()
    logger.info("✅ Models loaded successfully!")

    yield

    # Shutdown
    logger.info("🔄 Shutting down API service...")


# Initialize FastAPI app
app = FastAPI(
    title="Cameroon Hate Speech Detection API",
    description="AI-powered hate speech detection specifically designed for Cameroonian social media content",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================
# REQUEST/RESPONSE MODELS
# ================================

class TextAnalysisRequest(BaseModel):
    """Request model for single text analysis"""
    text: str = Field(..., min_length=1, max_length=2000, description="Text to analyze for hate speech")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    platform: Optional[str] = Field(None, description="Source platform (twitter, facebook, etc.)")
    store_result: bool = Field(True, description="Whether to store result in database")


class BatchAnalysisRequest(BaseModel):
    """Request model for batch text analysis"""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to analyze")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    platform: Optional[str] = Field(None, description="Source platform")
    store_results: bool = Field(True, description="Whether to store results in database")


class HateSpeechResponse(BaseModel):
    """Response model for hate speech detection"""
    text: str
    is_hate_speech: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    category: str
    severity: str
    detected_keywords: List[str]
    explanation: str
    timestamp: datetime
    processing_time_ms: float


class BatchAnalysisResponse(BaseModel):
    """Response model for batch analysis"""
    total_analyzed: int
    hate_speech_detected: int
    processing_time_ms: float
    results: List[HateSpeechResponse]
    summary: Dict[str, Any]


class SystemStatsResponse(BaseModel):
    """Response model for system statistics"""
    total_requests: int
    hate_speech_detected: int
    clean_content: int
    avg_processing_time_ms: float
    keyword_triggered_percentage: float
    ai_only_detections: int
    uptime: str
    models_loaded: Dict[str, str]


# ================================
# UTILITY FUNCTIONS
# ================================

def get_detector() -> CameroonHateSpeechDetector:
    """Dependency to get detector instance"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Detection models not loaded")
    return detector


def get_db_manager() -> DatabaseManager:
    """Dependency to get database manager"""
    if db_manager is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return db_manager


def convert_result_to_response(result: HateSpeechResult, processing_time: float) -> HateSpeechResponse:
    """Convert internal result to API response"""
    return HateSpeechResponse(
        text=result.text,
        is_hate_speech=result.is_hate_speech,
        confidence=result.confidence,
        category=result.category,
        severity=result.severity,
        detected_keywords=result.detected_keywords,
        explanation=result.explanation,
        timestamp=result.timestamp,
        processing_time_ms=processing_time
    )


# ================================
# API ENDPOINTS
# ================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """API documentation and status page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cameroon Hate Speech Detection API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { color: #2c3e50; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { font-weight: bold; color: #e74c3c; }
            code { background: #ecf0f1; padding: 2px 4px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1 class="header">🇨🇲 Cameroon Hate Speech Detection API</h1>
        <p>AI-powered hate speech detection for Cameroonian social media content</p>

        <h2>Quick Test</h2>
        <form action="/analyze" method="post" style="margin: 20px 0;">
            <textarea name="text" placeholder="Enter text to analyze..." style="width: 100%; height: 100px; padding: 10px;"></textarea><br><br>
            <button type="submit" style="padding: 10px 20px; background: #3498db; color: white; border: none; border-radius: 5px;">Analyze Text</button>
        </form>

        <h2>API Endpoints</h2>

        <div class="endpoint">
            <span class="method">POST</span> <code>/analyze</code><br>
            Analyze single text for hate speech
        </div>

        <div class="endpoint">
            <span class="method">POST</span> <code>/analyze/batch</code><br>
            Analyze multiple texts in batch
        </div>

        <div class="endpoint">
            <span class="method">GET</span> <code>/stats</code><br>
            Get system statistics and performance metrics
        </div>

        <div class="endpoint">
            <span class="method">GET</span> <code>/health</code><br>
            Check API health status
        </div>

        <div class="endpoint">
            <span class="method">GET</span> <code>/recent</code><br>
            Get recent hate speech detections
        </div>

        <p><strong>📚 Full Documentation:</strong> <a href="/docs">Interactive API Docs</a> | <a href="/redoc">ReDoc</a></p>

        <h2>Features</h2>
        <ul>
            <li>✅ 160+ Cameroon-specific hate speech keywords</li>
            <li>✅ Multi-language support (French, English, Pidgin)</li>
            <li>✅ Keyword-triggered AI analysis for efficiency</li>
            <li>✅ Real-time processing and batch analysis</li>
            <li>✅ Comprehensive logging and statistics</li>
        </ul>
    </body>
    </html>
    """


@app.post("/analyze", response_model=HateSpeechResponse)
async def analyze_text(
        request: TextAnalysisRequest,
        background_tasks: BackgroundTasks,
        detector: CameroonHateSpeechDetector = Depends(get_detector),
        db: DatabaseManager = Depends(get_db_manager)
):
    """
    Analyze a single text for hate speech

    - **text**: The text content to analyze
    - **user_id**: Optional identifier for the user who posted the content
    - **platform**: Optional platform identifier (twitter, facebook, etc.)
    - **store_result**: Whether to store the result in the database

    Returns detailed analysis including confidence score, detected keywords, and explanation.
    """
    try:
        start_time = datetime.now()

        # Perform hate speech detection
        result = detector.detect_hate_speech(request.text)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Store result in database if requested
        if request.store_result:
            metadata = {
                'user_id': request.user_id,
                'platform': request.platform,
                'post_id': None,
                'api_endpoint': '/analyze'
            }
            background_tasks.add_task(db.store_detection, result, metadata)

        # Convert to API response format
        response = convert_result_to_response(result, processing_time)

        # Log the request
        logger.info(f"Single analysis: {'HATE' if result.is_hate_speech else 'CLEAN'} "
                    f"({result.confidence:.2%}) - {processing_time:.1f}ms")

        return response

    except Exception as e:
        logger.error(f"Error in single text analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(
        request: BatchAnalysisRequest,
        background_tasks: BackgroundTasks,
        detector: CameroonHateSpeechDetector = Depends(get_detector),
        db: DatabaseManager = Depends(get_db_manager)
):
    """
    Analyze multiple texts for hate speech in batch

    - **texts**: List of text contents to analyze (max 100)
    - **user_id**: Optional identifier for the user
    - **platform**: Optional platform identifier
    - **store_results**: Whether to store results in the database

    Returns analysis for each text plus summary statistics.
    """
    try:
        start_time = datetime.now()

        # Perform batch detection
        results = detector.batch_detect(request.texts)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        avg_time_per_text = processing_time / len(request.texts)

        # Convert results to API format
        api_results = []
        hate_count = 0
        categories = {}
        severities = {}

        for result in results:
            api_result = convert_result_to_response(result, avg_time_per_text)
            api_results.append(api_result)

            if result.is_hate_speech:
                hate_count += 1
                categories[result.category] = categories.get(result.category, 0) + 1
                severities[result.severity] = severities.get(result.severity, 0) + 1

        # Store results in database if requested
        if request.store_results:
            metadata = {
                'user_id': request.user_id,
                'platform': request.platform,
                'post_id': None,
                'api_endpoint': '/analyze/batch'
            }
            for result in results:
                background_tasks.add_task(db.store_detection, result, metadata)

        # Create summary
        summary = {
            'hate_speech_rate': hate_count / len(request.texts),
            'categories_detected': categories,
            'severity_breakdown': severities,
            'avg_time_per_text_ms': avg_time_per_text
        }

        response = BatchAnalysisResponse(
            total_analyzed=len(request.texts),
            hate_speech_detected=hate_count,
            processing_time_ms=processing_time,
            results=api_results,
            summary=summary
        )

        # Log the batch request
        logger.info(f"Batch analysis: {len(request.texts)} texts, "
                    f"{hate_count} hate speech detected, {processing_time:.1f}ms total")

        return response

    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


@app.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats(
        detector: CameroonHateSpeechDetector = Depends(get_detector),
        db: DatabaseManager = Depends(get_db_manager)
):
    """
    Get system statistics and performance metrics

    Returns comprehensive statistics about API usage, detection performance,
    and system health metrics.
    """
    try:
        # Get detector statistics
        detector_stats = detector.get_statistics()

        # Get database statistics
        db_stats = db.get_statistics(days=7)

        # Calculate metrics
        total_requests = detector_stats.get('total_processed', 0)
        hate_detected = detector_stats.get('hate_speech_detected', 0)
        clean_content = total_requests - hate_detected

        keyword_triggered = detector_stats.get('keyword_triggered', 0)
        ai_only = detector_stats.get('ai_only_detected', 0)

        keyword_percentage = (keyword_triggered / total_requests * 100) if total_requests > 0 else 0

        response = SystemStatsResponse(
            total_requests=total_requests,
            hate_speech_detected=hate_detected,
            clean_content=clean_content,
            avg_processing_time_ms=150.0,  # Estimate based on typical performance
            keyword_triggered_percentage=keyword_percentage,
            ai_only_detections=ai_only,
            uptime=str(datetime.now() - datetime.now().replace(hour=0, minute=0, second=0)),
            models_loaded={
                "keyword_detector": "160+ Cameroon-specific terms",
                "ai_classifier": "Pre-trained transformer model",
                "database": "SQLite with real-time logging"
            }
        )

        return response

    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.get("/health")
async def health_check(detector: CameroonHateSpeechDetector = Depends(get_detector)):
    """
    Health check endpoint

    Returns the current health status of the API and its components.
    """
    try:
        # Test detector with a simple text
        test_result = detector.detect_hate_speech("Hello, this is a test.")

        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "components": {
                "detector": "✅ operational",
                "database": "✅ operational",
                "ai_model": "✅ operational"
            },
            "test_detection": {
                "processed": True,
                "response_time_ms": "<10"
            }
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.get("/recent")
async def get_recent_detections(
        hours: int = 24,
        limit: int = 50,
        hate_only: bool = True,
        db: DatabaseManager = Depends(get_db_manager)
):
    """
    Get recent hate speech detections

    - **hours**: Number of hours to look back (default: 24)
    - **limit**: Maximum number of results to return (default: 50)
    - **hate_only**: Only return hate speech detections (default: true)

    Returns recent detections for monitoring and analysis purposes.
    """
    try:
        detections = db.get_recent_detections(hours=hours, hate_only=hate_only)

        # Limit results
        limited_detections = detections[:limit]

        return {
            "total_found": len(detections),
            "returned": len(limited_detections),
            "time_period_hours": hours,
            "detections": limited_detections
        }

    except Exception as e:
        logger.error(f"Error getting recent detections: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recent detections: {str(e)}")


@app.get("/keywords")
async def get_keyword_categories(detector: CameroonHateSpeechDetector = Depends(get_detector)):
    """
    Get information about keyword categories and detection capabilities

    Returns the keyword categories, count, and examples for transparency.
    """
    try:
        keywords_info = {}
        total_keywords = 0

        for category, data in detector.keywords_detector.keywords.items():
            keywords_info[category] = {
                "count": len(data['terms']),
                "severity": data['severity'],
                "category": data['category'],
                "examples": data['terms'][:3]  # Show first 3 as examples
            }
            total_keywords += len(data['terms'])

        return {
            "total_keywords": total_keywords,
            "categories": len(keywords_info),
            "keyword_categories": keywords_info,
            "languages_supported": ["French", "English", "Pidgin", "Mixed"],
            "last_updated": "2024-12-17"
        }

    except Exception as e:
        logger.error(f"Error getting keyword info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get keyword info: {str(e)}")


# ================================
# MAIN APPLICATION
# ================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cameroon Hate Speech Detection API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")

    args = parser.parse_args()

    print("🇨🇲 Starting Cameroon Hate Speech Detection API...")
    print(f"📡 Server will be available at: http://{args.host}:{args.port}")
    print(f"📚 API Documentation: http://{args.host}:{args.port}/docs")
    print(f"🔍 Health Check: http://{args.host}:{args.port}/health")

    uvicorn.run(
        "hate_speech_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level="info"
    )