import uvicorn
import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file
load_dotenv()

# 🔧 Prevent repeated HuggingFace Hub HTTP checks on every startup.
# The model is already cached locally after first download.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Setup logging — show only important messages
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("unified_server")

# 🔇 Silence noisy third-party libraries and verbose internal modules
_silenced_libs = [
    # Third-party
    'alembic', 'apscheduler', 'faiss', 'faiss.loader',
    'sentence_transformers', 'sentence_transformers.models.Transformer',
    'httpx', 'urllib3', 'transformers', 'transformers.modeling_utils', 'safetensors',
    'accelerate',
    # AI Recommender internals
    'ai_book_recommender.models.ensemble',
    'ai_book_recommender.models.neural_reranker',
    'ai_book_recommender.models.context_ranker',
    'ai_book_recommender.models.transformer_encoder',
    'ai_book_recommender.models.graph_recommender',
    'ai_book_recommender.feature_store',
    'ai_book_recommender.retrieval.vector_index',
    'ai_book_recommender.retrieval.hybrid_retrieval',
    'ai_book_recommender.user_intelligence.online_learning',
    'ai_book_recommender.retrieval.cache_manager',
    'ai_book_recommender.interest_fetcher_service',
    'ai_book_recommender.engine',
    'ai_book_recommender.explainability.explainer',
    'ai_book_recommender.evaluation.metrics',
    'ai_book_recommender.user_intelligence.user_model',
    # Flask app internals
    'recommendation_pipeline',
    'flask_book_recommendation.recommender.pipeline',
    'flask_book_recommendation.recommender.content',
    'flask_book_recommendation.recommender.trending',
    'flask_book_recommendation.recommender.helpers',
    'flask_book_recommendation.utils',
    'flask_book_recommendation.routes.main',
    'flask_book_recommendation.routes.public',
    'ai_book_recommender.user_intelligence.behavior_sequence',
    'flask_book_recommendation.app',
    'flask.app',
]

for _lib in _silenced_libs:
    if _lib in ['alembic', 'apscheduler', 'faiss', 'faiss.loader', 'sentence_transformers', 'sentence_transformers.models.Transformer', 'httpx', 'urllib3', 'transformers', 'transformers.modeling_utils', 'safetensors', 'accelerate']:
        logging.getLogger(_lib).setLevel(logging.ERROR)
    else:
        logging.getLogger(_lib).setLevel(logging.WARNING)

# Suppress PyTorch UserWarnings (tensor creation warnings)
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

# Import Flask App
try:
    from flask_book_recommendation.app import create_app
    flask_app = create_app()
    logger.info("✅ Flask application created successfully")
except Exception as e:
    logger.error(f"❌ Failed to create Flask application: {e}")
    flask_app = None

# Import AI Recommender App
try:
    from ai_book_recommender.api import app as engine_app
    logger.info("✅ AI Recommender API imported successfully")
except Exception as e:
    logger.error(f"❌ Failed to import AI Recommender API: {e}")
    engine_app = FastAPI() # Fallback empty app

# Create Root App
root_app = FastAPI(title="Unified Book Platform")

# CORS middleware for the root app
root_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@root_app.get("/health")
async def health_check():
    return {
        "status": "online",
        "flask_ready": flask_app is not None,
        "engine_ready": engine_app is not None
    }

# Mount AI Engine API
root_app.mount("/api/engine", engine_app)

# Mount Flask at Root
if flask_app:
    root_app.mount("/", WSGIMiddleware(flask_app))

# Configuration
PORT = int(os.environ.get("PORT", 5000))

if __name__ == "__main__":
    logger.info(f"🚀 Starting Unified Server on http://localhost:{PORT}")
    
    uvicorn.run(
        root_app, 
        host="0.0.0.0", 
        port=PORT, 
        reload=False
    )
