# -*- coding: utf-8 -*-
"""
Deep Learning pipeline — Two-Tower model, AI embeddings, behavior-based.
"""
import logging
import random
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask import current_app

from ..models import Book, UserRatingCF, SearchHistory, UserBookView, BookEmbedding, BookStatus, UserGenre, Genre
from ..extensions import db, cache
from ..utils import fetch_google_books, get_text_embedding
from ai_book_recommender.user_intelligence.behavior_sequence import SessionEncoder
from .helpers import (
    _book_to_dict, _apply_mmr_diversity, run_in_context, get_dl_engine, logger as _hlogger
)
from .embedding_cache import _get_embeddings_matrix
from .collaborative import get_cf_similar
from .trending import get_trending

logger = logging.getLogger(__name__)


def _get_ai_embedding_recommendations(
    user_id,
    viewed_book_ids,
    search_queries=None,
    favorite_book_ids=None,
    high_rated_book_ids=None,
    explicit_genres=None,
    limit=12,
    offset=0,
    randomize=False
):
    """
    AI Embedding recommendations using user profile centroid.
    """
    try:
        # 1. Try AI Engine first
        try:
            from ..ai_client import ai_client
            if ai_client:
                # Provide minimal inputs mapping to the new get_recommendations signature
                result = ai_client.get_recommendations(
                    user_id=user_id,
                    k=limit
                )
                
                if result and isinstance(result, list) and len(result) > 0:
                    final_recs = []
                    for item in result:
                        b_id = item.get("book_id")
                        score = item.get("score", 0.5)
                        explanation = item.get("explanation", "AI Pick")
                        
                        b = Book.query.get(b_id) if isinstance(b_id, int) else Book.query.filter_by(google_id=b_id).first()
                        if b:
                            algo = "Smart Matching"
                            if "similar to" in explanation.lower():
                                algo = "Hybrid Ranking Engine"
                            elif "interest" in explanation.lower():
                                algo = "Behavioral Learning"

                            meta = {
                                "score": f"{score:.2f}",
                                "algorithm_used": algo,
                                "model_version": "v2.1 (Two-Tower)",
                                "reason_detail": explanation
                            }

                            d = _book_to_dict(b, source="AI Neural Brain", reason=explanation, extra_meta=meta)
                            if d:
                                final_recs.append(d)
                    
                    if offset < len(final_recs):
                        return final_recs[offset:offset+limit]
                    else:
                        return []
                        
        except Exception as e:
            logger.error(f"[AI-Bridge] Error contacting AI engine: {e}")

        # 2. FALLBACK: Local Logic
        search_queries = search_queries or []
        favorite_book_ids = favorite_book_ids or []
        high_rated_book_ids = high_rated_book_ids or []
        
        all_vectors = []
        
        if viewed_book_ids:
            view_embeds = BookEmbedding.query.filter(BookEmbedding.book_id.in_(viewed_book_ids)).all()
            for row in view_embeds:
                if row.vector is not None:
                    v = np.array(__import__("pickle").loads(row.vector) if isinstance(row.vector, bytes) else row.vector, dtype=np.float32)
                    if v.ndim == 1:
                        all_vectors.append(v)

        if favorite_book_ids:
            fav_embeds = BookEmbedding.query.filter(BookEmbedding.book_id.in_(favorite_book_ids)).all()
            for row in fav_embeds:
                if row.vector is not None:
                    v = np.array(__import__("pickle").loads(row.vector) if isinstance(row.vector, bytes) else row.vector, dtype=np.float32)
                    if v.ndim == 1:
                        all_vectors.append(v)
                        all_vectors.append(v)
                        all_vectors.append(v)

        if high_rated_book_ids:
            ids_only = list(high_rated_book_ids.keys()) if isinstance(high_rated_book_ids, dict) else high_rated_book_ids
            rated_embeds = BookEmbedding.query.filter(BookEmbedding.book_id.in_(ids_only)).all()
            for row in rated_embeds:
                if row.vector is not None:
                    v = np.array(__import__("pickle").loads(row.vector) if isinstance(row.vector, bytes) else row.vector, dtype=np.float32)
                    if v.ndim == 1:
                        weight = 2
                        if isinstance(high_rated_book_ids, dict):
                            stars = high_rated_book_ids.get(row.book_id, 4)
                            if stars >= 5: weight = 4
                            elif stars >= 4: weight = 2
                        for _ in range(weight):
                            all_vectors.append(v)

        processed_queries = 0
        for i, query in enumerate(search_queries):
            if not query: continue
            if processed_queries >= 5: break
            try:
                q_vec = get_text_embedding(query)
                if q_vec:
                    v = np.array(q_vec, dtype=np.float32)
                    if i == 0:
                        for _ in range(5): all_vectors.append(v)
                    elif i == 1:
                        for _ in range(3): all_vectors.append(v)
                    else:
                        all_vectors.append(v)
                    processed_queries += 1
            except Exception as e:
                logger.error(f"[AI-Embed] Search embed error: {e}")

        if explicit_genres:
            for genre in explicit_genres:
                try:
                    g_vec = get_text_embedding(f"Genre: {genre}")
                    if g_vec:
                        v = np.array(g_vec, dtype=np.float32)
                        for _ in range(4): all_vectors.append(v)
                except Exception as e:
                    logger.error(f"[AI-Embed] Genre embed error: {e}")

        if not all_vectors:
            logger.debug(f"[AI-Embed] No vectors found for user profile")
            return []
        
        target_dim = all_vectors[0].shape[0]
        consistent_vectors = [v for v in all_vectors if v.shape[0] == target_dim]
        
        if not consistent_vectors:
            logger.warning(f"[AI-Embed] No consistent vectors found for dimension {target_dim}")
            return []

        user_profile = np.mean(np.vstack(consistent_vectors), axis=0).reshape(1, -1)
        
        exclude_ids = set(viewed_book_ids) | set(favorite_book_ids or [])
        if isinstance(high_rated_book_ids, dict):
            exclude_ids |= set(high_rated_book_ids.keys())
        elif isinstance(high_rated_book_ids, list):
            exclude_ids |= set(high_rated_book_ids)

        matrix, matrix_ids = _get_embeddings_matrix()
        
        if matrix is None:
            logger.warning("[AI-Embed] Matrix is empty or not loaded.")
            return []

        if matrix.shape[1] != target_dim:
            logger.warning(f"[AI-Embed] Matrix dimension mismatch ({matrix.shape[1]}) vs Target ({target_dim})")
            return []

        candidate_ids = matrix_ids
        candidate_vectors = matrix
        
        exclude_indices = [i for i, bid in enumerate(candidate_ids) if bid in exclude_ids]
        
        mat = candidate_vectors
        sims = cosine_similarity(user_profile, mat)[0]
        
        if exclude_indices:
            sims[exclude_indices] = -1.0

        ranked_indices = np.argsort(sims)[::-1]
        
        recs = []
        seen_ids = set()
        start_idx = offset
        if start_idx >= len(ranked_indices):
            return []
             
        if randomize:
            pool_size = max(limit * 3, 30)
            candidate_pool = ranked_indices[start_idx : start_idx + pool_size]
            np.random.shuffle(candidate_pool)
            indices_to_iter = candidate_pool
        else:
            indices_to_iter = ranked_indices[start_idx:]
            
        # 🚀 [OPTIMIZATION] Bulk-fetch books for AI
        target_ids = [candidate_ids[idx] for idx in indices_to_iter if sims[idx] >= 0.25]
        if not target_ids:
            return []
            
        found_books = Book.query.filter(Book.id.in_(target_ids)).all()
        book_map = {b.id: b for b in found_books}
        
        # 🍏 Interests-Based Filtering Gatekeeper 🍏
        user_category_set = {g.lower() for g in (explicit_genres or [])}
        
        # ❄️ COLD START FALLBACK: If user has no interests, use high-quality defaults
        if not user_category_set:
            user_category_set = {"psychology", "philosophy", "science", "technology", "business", "self-help"}
        
        for idx in indices_to_iter:
            score = sims[idx]
            original_score = score
            
            book = book_map.get(candidate_ids[idx])
            if not book:
                continue

            # Category Match Logic
            book_cats = {c.strip().lower() for c in (book.categories or "").split(",")}
            has_match = any(cat in user_category_set for cat in book_cats)
            
            if user_category_set:
                if has_match:
                    # Boost for direct interest match
                    score = min(1.0, score * 1.5)
                else:
                    # Moderate Penalty instead of extreme (0.3x)
                    # Allows very high semantic matches to still surface
                    score = score * 0.3
            
            # Revised floor threshold to allow some variety while maintaining quality
            if score < 0.15:
                continue
                
            book_id_key = book.google_id or f"local_{book.id}"
            if book_id_key in seen_ids:
                continue
            seen_ids.add(book_id_key)
            
            meta = {
                "score": f"{score:.2f}",
                "algorithm_used": "Sematic Hybrid Embeddings",
                "model_version": "v1.5 (Local)",
                "reason_detail": f"Based on semantic similarity to your reading history ({int(score*100)}% match)."
            }

            book_dict = _book_to_dict(
                book,
                source="AI Smart Match",
                reason=f"🧠 تطابق ذكي: {int(score*100)}%",
                extra_meta=meta
            )
            if book_dict:
                book_dict["category"] = book.categories.split(",")[0].strip() if book.categories else "unknown"
                book_dict["rec_type"] = "ai_embedding"
                recs.append(book_dict)
            
            if len(recs) >= limit:
                break
        
        logger.info(f"[AI-Embed] Found {len(recs)} semantic recommendations from mixed signals")
        return recs
        
    except Exception as e:
        logger.error(f"[AI-Embed] Error: {e}", exc_info=True)
        return []


def apply_diversity(books, book_embeddings_map, lambda_=0.7):
    """
    تطبيق MMR (Maximal Marginal Relevance) والتنوع المطلوب في المتطلبات:
    1. لا تكرار لنفس المؤلف في أول 5 نتائج
    2. لا تكرار لنفس التصنيف (genre) أكثر من مرتين متتاليتين
    """
    if not books: return []
    
    # تحضير الخصائص لجميع الكتب
    for i, b in enumerate(books):
        if 'score' not in b:
            b['score'] = float(b.get('confidence', 1.0)) * (len(books) - i) / len(books)
            
    unselected = list(books)
    selected = []
    
    while unselected:
        if not selected:
            # نختار العنصر الأول (الأعلى تقييماً)
            selected.append(unselected.pop(0))
            continue
            
        best_mmr = -1000.0
        best_idx = -1
        
        for i, cand in enumerate(unselected):
            rel_score = float(cand.get('score', 0))
            
            author = cand.get('author', '')
            cat = cand.get('categories', '') or cand.get('category', 'unknown')
            
            # قيد: لا تكرار نفس المؤلف في أول 5 نتائج
            if len(selected) < 5 and author and author != "Unknown":
                if any(s.get('author') == author for s in selected):
                    continue
            
            # قيد: لا تكرار نفس التصنيف أكثر من مرتين متتاليتين
            if len(selected) >= 2 and cat and cat != "unknown":
                c1 = selected[-1].get('categories', '') or selected[-1].get('category', 'unknown')
                c2 = selected[-2].get('categories', '') or selected[-2].get('category', 'unknown')
                if cat == c1 == c2:
                    continue
            
            # حساب تشابه MMR إذا توفرت المتجهات
            sim_penalty = 0.0
            cid = cand.get('_internal_db_id')  # We attach this below
            if cid and book_embeddings_map and cid in book_embeddings_map and selected:
                cand_vec = book_embeddings_map[cid].reshape(1, -1)
                
                sel_vecs = []
                for s in selected:
                    sid = s.get('_internal_db_id')
                    if sid and sid in book_embeddings_map:
                        sel_vecs.append(book_embeddings_map[sid])
                
                if sel_vecs:
                    sel_mat = np.vstack(sel_vecs)
                    sims = cosine_similarity(cand_vec, sel_mat)[0]
                    sim_penalty = np.max(sims)
            
            mmr_score = lambda_ * rel_score - (1.0 - lambda_) * sim_penalty
            
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i
                
        if best_idx != -1:
            selected.append(unselected.pop(best_idx))
        else:
            # لا توجد عناصر تطابق القيود، نضيف المتبقي تدريجياً
            selected.append(unselected.pop(0))
    
    return selected

@cache.memoize(timeout=3600)
def get_deep_learning_recommendations(user_id, limit=100, randomize=False):
    """
    Get recommendations using the Two-Tower Deep Learning model 
    with FAISS candidate retrieval and BookStatus filtering.
    """
    import os
    import time
    import torch
    import numpy as np
    from flask import current_app
    from sqlalchemy import or_

    from ..models import Book, UserRatingCF, BookStatus, UserBookView, BookEmbedding
    from ..extensions import db
    from ..recommendation_logger import RecommendationPipelineLogger
    from ..advanced_recommender.neural_model import TwoTowerModel
    from .helpers import _book_to_dict
    from .trending import get_trending

    with RecommendationPipelineLogger(user_id or 0) as pipeline_log:
        try:
            if not user_id:
                pipeline_log.log_fallback("user_id is None/0 — user not logged in")
                pipeline_log.set_final_count(0)
                return [] # No trending fallback

            # ── Stage 1: TRANSFORMER (Embedding Retrieval) ──
            transformer_start = time.perf_counter()
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Lazy Load Model
            if not hasattr(current_app, 'dl_model'):
                model_path = "instance/models/two_tower_model.pt"
                model = TwoTowerModel()
                if os.path.exists(model_path):
                    try:
                        model.load_state_dict(torch.load(model_path, map_location=device))
                    except Exception as e:
                        print(f"Warning: Could not load TwoTowerModel weights properly: {e}")
                model.to(device)
                model.eval()
                current_app.dl_model = model
            
            # Lazy Load FAISS Index
            if not hasattr(current_app, 'faiss_index'):
                from ..models import BookEmbedding
                import faiss
                embeddings_data = BookEmbedding.query.all()
                if not embeddings_data:
                    pipeline_log.log_stage("transformer", time_ms=(time.perf_counter()-transformer_start)*1000, results=0)
                    pipeline_log.log_fallback("No book embeddings in database")
                    pipeline_log.set_final_count(0)
                    return [] # No trending fallback
                id_map = []
                vec_list = []
                for emb in embeddings_data:
                    if emb.vector is not None:
                        id_map.append(emb.book_id)
                        vec = __import__("pickle").loads(emb.vector) if isinstance(emb.vector, bytes) else emb.vector
                        vec_list.append(np.array(vec, dtype=np.float32))
                        
                dim = 384
                index = faiss.IndexFlatIP(dim)
                vectors = np.array(vec_list)
                faiss.normalize_L2(vectors)
                index.add(vectors)
                
                current_app.faiss_index = index
                current_app.faiss_id_map = id_map
                
            model = current_app.dl_model
            index = current_app.faiss_index
            id_map = current_app.faiss_id_map
            
            transformer_time = (time.perf_counter() - transformer_start) * 1000
            pipeline_log.log_stage("transformer", time_ms=transformer_time, results=len(id_map))
            
            # ── Stage 2: BEHAVIORAL (User Interaction Gathering) ──
            behavioral_start = time.perf_counter()
            
            # Gather view history
            views = UserBookView.query.filter_by(user_id=user_id).order_by(UserBookView.last_viewed_at.desc()).limit(15).all()
            view_bids = [v.book_id for v in views if v.book_id]
            
            # Gather ratings >= 3
            ratings = UserRatingCF.query.filter(UserRatingCF.user_id == user_id, UserRatingCF.rating >= 3.0).order_by(UserRatingCF.created_at.desc()).limit(15).all()
            google_ids = [r.google_id for r in ratings]
            rating_bids = [b.id for b in Book.query.filter(Book.google_id.in_(google_ids)).all()] if google_ids else []
            
            # Gather statuses
            statuses = BookStatus.query.filter(BookStatus.user_id == user_id, BookStatus.status.in_(['favorite', 'later'])).all()
            status_bids = [s.book_id for s in statuses]
            
            finished_statuses = BookStatus.query.filter_by(user_id=user_id, status='finished').all()
            finished_bids = {s.book_id for s in finished_statuses}
            
            # Combine all positive interaction book IDs
            interaction_ids = list(set(view_bids + rating_bids + status_bids))
            
            # Capture User Search History semantics
            from flask_book_recommendation.models import SearchHistory
            from flask_book_recommendation.extensions import db
            from flask_book_recommendation.utils import get_text_embedding, translate_to_english_with_gemini
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            searches = db.session.query(SearchHistory).filter_by(user_id=user_id).order_by(SearchHistory.created_at.desc()).limit(5).all()
            search_queries = [s.query for s in searches if s.query]
            
            # --- New: Interest-based signals (Genres) ---
            from ..models import UserGenre, Genre, UserPreference
            user_genres = db.session.query(Genre.name).join(UserGenre).filter(UserGenre.user_id == user_id).all()
            interest_genres = [g[0] for g in user_genres]
            
            # Also populate from UserPreference for robustness if UserGenre fails
            user_prefs = db.session.query(UserPreference.topic).filter(UserPreference.user_id == user_id).all()
            for p in user_prefs:
                if p[0] not in interest_genres:
                    interest_genres.append(p[0])
            if interest_genres:
                logger.info(f"[DL-Rec] User {user_id} has interest genres: {interest_genres}")
                # Treat genres as virtual high-intent search queries
                search_queries.extend(interest_genres[:5])

            search_hists = []
            
            if search_queries:
                def process_search(q):
                    try:
                        eng_query = translate_to_english_with_gemini(q) or q
                        emb = get_text_embedding(eng_query)
                        if emb and len(emb) == 384:
                            return np.array(emb, dtype=np.float32)
                    except Exception as e:
                        logger.error(f"Error processing search query '{q}': {e}")
                    return None

                from concurrent.futures import wait, FIRST_COMPLETED
                executor = ThreadPoolExecutor(max_workers=5)
                try:
                    future_to_query = {executor.submit(process_search, q): q for q in search_queries}
                    # Wait for up to 8s
                    done, not_done = wait(future_to_query.keys(), timeout=8)
                    
                    for future in done:
                        try:
                            res = future.result()
                            if res is not None:
                                search_hists.append(res)
                        except Exception as e:
                            logger.error(f"[DL-Rec] Search result error: {e}")
                    
                    if not_done:
                        logger.warning(f"[DL-Rec] {len(not_done)} search processes timed out for user {user_id}")
                finally:
                    # shutdown(wait=False) is key: it won't wait for the hanging Gemini threads
                    executor.shutdown(wait=False)
                        
            behavioral_time = (time.perf_counter() - behavioral_start) * 1000
            
            total_interactions = len(interaction_ids) + len(search_hists)
            pipeline_log.log_stage("behavioral", time_ms=behavioral_time, results=total_interactions)
            
            if not interaction_ids and not search_hists:
                pipeline_log.log_fallback(f"No interactions or interests found for user {user_id} (views={len(view_bids)}, ratings={len(rating_bids)}, statuses={len(status_bids)}, genres={len(interest_genres)})")
                pipeline_log.set_final_count(0)
                return [] # Still no fallback to random/trending
                
            embs = BookEmbedding.query.filter(BookEmbedding.book_id.in_(interaction_ids)).all()
            hists = [np.array(__import__("pickle").loads(e.vector) if isinstance(e.vector, bytes) else e.vector, dtype=np.float32) for e in embs if e.vector is not None]
            
            hists.extend(search_hists)
            
            if not hists:
                pipeline_log.log_fallback(f"User {user_id} has interactions but 0 matching embeddings")
                pipeline_log.set_final_count(0)
                return [] # No trending fallback
                
            # Pad or truncate history to 10
            hists = hists[-10:]
            while len(hists) < 10:
                hists.append(np.zeros(384, dtype=np.float32))
            hist_vec = np.array(hists)
            
            # Mean for interest vector
            int_vec = np.mean([h for h in hists if np.any(h)], axis=0)
            if not np.any(int_vec):
                int_vec = np.zeros(384, dtype=np.float32)
                
            # ── Stage 3: NEURAL (Two-Tower Model + FAISS Scoring) ──
            neural_start = time.perf_counter()
            
            # Compute User Tower Vector
            with torch.no_grad():
                u_id_t = torch.tensor([user_id], dtype=torch.long).clamp(0, 9999).to(device)
                hist_t = torch.tensor(np.array([hist_vec]), dtype=torch.float32).to(device)
                int_t = torch.tensor(np.array([int_vec]), dtype=torch.float32).to(device)
                
                user_emb_tensor = model.user_tower(u_id_t, hist_t, int_t) # (1, 128)
                
            # FAISS Search
            import faiss
            query_vec = int_vec.reshape(1, -1).copy()
            faiss.normalize_L2(query_vec)
            
            D, I = index.search(query_vec, k=600)
            candidate_ids = [id_map[i] for i in I[0] if i != -1]
            
            # Filter 'finished' books and interacted books
            interaction_set = set(interaction_ids)
            candidate_ids = [bid for bid in candidate_ids if bid not in finished_bids and bid not in interaction_set]
            
            # Prepare for TwoTower Scoring (Top 500)
            candidate_ids = candidate_ids[:500]
            if not candidate_ids:
                neural_time = (time.perf_counter() - neural_start) * 1000
                pipeline_log.log_stage("neural", time_ms=neural_time, results=0)
                pipeline_log.log_fallback("No candidates after filtering finished/interacted books")
                pipeline_log.set_final_count(0)
                return [] # No trending fallback
                
            cand_embs = BookEmbedding.query.filter(BookEmbedding.book_id.in_(candidate_ids)).all()
            cand_dict = {e.book_id: np.array(__import__("pickle").loads(e.vector) if isinstance(e.vector, bytes) else e.vector, dtype=np.float32) for e in cand_embs if e.vector is not None}
            
            act_ids = list(cand_dict.keys())
            act_vecs = np.array(list(cand_dict.values()))
            
            if len(act_ids) == 0:
                neural_time = (time.perf_counter() - neural_start) * 1000
                pipeline_log.log_stage("neural", time_ms=neural_time, results=0)
                pipeline_log.log_fallback("No candidate embeddings found")
                pipeline_log.set_final_count(0)
                return [] # No trending fallback
                
            with torch.no_grad():
                item_t = torch.tensor(act_vecs, dtype=torch.float32).to(device)
                item_embs = model.item_tower(item_t) # (N, 128)
                scores = (user_emb_tensor * item_embs).sum(dim=1).cpu().numpy()
            
            neural_time = (time.perf_counter() - neural_start) * 1000
            pipeline_log.log_stage("neural", time_ms=neural_time, results=len(act_ids))
                
            # 4. Sort
            scored_candidates = sorted(zip(act_ids, scores), key=lambda x: x[1], reverse=True)
            top_final = scored_candidates  # iterate over ALL candidates, break when we have enough
            
            # 🚀 [OPTIMIZATION] Bulk-fetch books for Behavioral
            found_books = Book.query.filter(Book.id.in_(act_ids)).all()
            book_map = {b.id: b for b in found_books}
            
            recs = []
            seen_ids = set()
            for rank, (bid, score) in enumerate(top_final):
                book = book_map.get(bid)
                if book:
                    book_id_key = book.google_id or f"local_{book.id}"
                    if book_id_key in seen_ids:
                        continue
                    seen_ids.add(book_id_key)
                    
                    confidence = int(max(0, min(1, (score + 1.0) / 2.0)) * 100)
                    meta = {
                        "algorithm_used": "TwoTower Neural + FAISS",
                        "model_version": "v3.0 (BPR Trained)",
                        "score": f"{score:.3f}",
                        "rank": rank + 1,
                        "reason_detail": f"Model Score: {score:.3f} | Retrieved via FAISS"
                    }
                    d = _book_to_dict(
                        book, 
                        source="الذكاء الاصطناعي", 
                        reason=f"🧠 دقة التوصية: {confidence}%", 
                        extra_meta=meta
                    )
                    if d:
                        recs.append(d)
                        
                    if len(recs) >= limit:
                        break
                    
            pipeline_log.set_final_count(len(recs))
            return recs
            
        except ImportError as e:
            pipeline_log.log_error(f"Missing dependency: {e}")
            return [] # No trending fallback
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"[DL-Rec] FAISS/DL Error: {e}", exc_info=True)
            pipeline_log.log_error(str(e))
            return [] # No trending fallback


def _fetch_behavior_hybrid_candidates(user_id, limit=12, offset=0, randomize=False, salt=0):
    """
    Heavy lifting: Fetches a large pool of hybrid candidates from various sources.
    """
    from datetime import datetime, timedelta
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        logger.info(f"[Behavior-Hybrid] Fetching pool for user {user_id}")
        
        recent_views = UserBookView.query.filter_by(user_id=user_id).order_by(UserBookView.last_viewed_at.desc()).limit(40).all()
        viewed_book_ids = [v.book_id for v in recent_views if v.book_id]
        viewed_google_ids = [v.google_id for v in recent_views if v.google_id]
        
        recent_searches = db.session.query(SearchHistory).filter_by(user_id=user_id).order_by(SearchHistory.created_at.desc()).limit(10).all()
        search_queries = [s.query for s in recent_searches if s.query]
        
        favorites = BookStatus.query.filter_by(user_id=user_id, status='favorite').all()
        favorite_book_ids = [f.book_id for f in favorites if f.book_id]

        user_ratings = UserRatingCF.query.filter(UserRatingCF.user_id==user_id, UserRatingCF.rating >= 4).all()
        high_rated_books = {r.id: r.rating for r in user_ratings}

        user_genres = db.session.query(Genre.name).join(UserGenre).filter(UserGenre.user_id == user_id).all()
        explicit_genres = [g[0] for g in user_genres]
        
        if not (viewed_book_ids or search_queries or favorite_book_ids or explicit_genres):
             return []

        all_recs = []
        
        app = current_app._get_current_object()
        
        ai_pool_limit = 150
        cf_pool_limit = 80
        explore_pool_limit = 80
        
        all_recs = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            futures["ai"] = executor.submit(
                run_in_context, app, _get_ai_embedding_recommendations,
                user_id, list(viewed_book_ids), search_queries[:5], list(favorite_book_ids),
                high_rated_books, explicit_genres, ai_pool_limit, 0, randomize
            )
            
            futures["cf"] = executor.submit(
                run_in_context, app, get_cf_similar, user_id, top_n=cf_pool_limit
            )
            
            def fetch_simple_explore():
                results = []
                seen_ex = set(viewed_google_ids)
                
                # Target should only be from real interests
                if not search_queries and not explicit_genres:
                    logger.info(f"[Hybrid-Explore] No user interests found, suppressing discovery")
                    return []
                
                # Default target from most recent search or genre
                target = search_queries[0] if search_queries else explicit_genres[0]
                
                if randomize:
                    target = random.choice(search_queries + explicit_genres)
                    logger.info(f"[Hybrid-Explore] Random target constrained to user interest: '{target}'")
                    
                try:
                    start_index = 0
                    if randomize:
                        start_index = (salt * 10) % 200
                    
                    books, _ = fetch_google_books(target, max_results=explore_pool_limit, start_index=start_index)
                    for b in books or []:
                        gid = b.get('id')
                        if gid and gid not in seen_ex:
                            vi = b.get('volumeInfo', {})
                            results.append({
                                "id": gid,
                                "title": vi.get("title"),
                                "author": ", ".join(vi.get("authors") or []),
                                "cover": (vi.get("imageLinks") or {}).get("thumbnail", "").replace("http://", "https://"),
                                "source": "استكشاف الذكاء الاصطناعي",
                                "reason": f"✨ مقترح بناءً على اهتمامك بـ {target}",
                                "score": 0.5,
                                "rec_type": "exploration"
                            })
                            seen_ex.add(gid)
                except: pass
                return results

            futures["explore"] = executor.submit(run_in_context, app, fetch_simple_explore)

            for key, future in futures.items():
                try:
                    res = future.result(timeout=12)
                    if res: all_recs.extend(res)
                except Exception as e:
                    logger.error(f"[Behavior-Hybrid] Future {key} failed: {e}")

        unique_final = []
        seen_ids = set()
        for r in all_recs:
            rid = r.get('id')
            if rid and rid not in seen_ids:
                seen_ids.add(rid)
                unique_final.append(r)
        
        logger.info(f"[Behavior-Hybrid] Pool fetching complete. Total candidates: {len(unique_final)}")
        return unique_final

    except Exception as e:
        logger.error(f"[Behavior-Hybrid] Pool fetch fatal error: {e}", exc_info=True)
        return []


def get_behavior_based_recommendations(user_id, limit=12, offset=0, randomize=False):
    """
    توصيات ذكية شاملة مدعمة بنموذج BehaviorSequenceModel والتنوع (MMR)
    """
    import torch
    from flask import current_app
    from ..models import UserEvent, Book, BookEmbedding

    if not user_id:
        return []

    try:
        salt = int(time.time() * 1000) % 100 if randomize else 0
        
        candidates = _fetch_behavior_hybrid_candidates(user_id, limit=limit*5, offset=offset, randomize=randomize, salt=salt)
        
        if not candidates:
            return [] # No trending fallback

        # ---------------------------------------------------------
        # Task 4 & 6: BehaviorSequenceModel + MMR Diversity
        # ---------------------------------------------------------
        
        # 1. Fetch last 10 session events
        last_event = UserEvent.query.filter_by(user_id=user_id).order_by(UserEvent.created_at.desc()).first()
        session_events = []
        if last_event and last_event.session_id:
            session_events = UserEvent.query.filter_by(user_id=user_id, session_id=last_event.session_id)\
                                 .order_by(UserEvent.created_at.desc()).limit(10).all()
        elif last_event:
            session_events = UserEvent.query.filter_by(user_id=user_id)\
                                 .order_by(UserEvent.created_at.desc()).limit(10).all()

        history_gids = [e.book_google_id for e in session_events if e.book_google_id]
        cand_ids = [c['id'] for c in candidates if c.get('id')]
        
        # 2. Get internal DB IDs to fetch embeddings
        all_query_ids = list(set(history_gids + [str(i) for i in cand_ids]))
        books_db = Book.query.filter(Book.google_id.in_(all_query_ids)).all()
        
        gid_to_dbid = {b.google_id: b.id for b in books_db if b.google_id}
        dbid_list = [b.id for b in books_db]
        
        # attach internal DB ID for MMR
        for c in candidates:
            c['_internal_db_id'] = gid_to_dbid.get(str(c.get('id')))
            
        vector_map = {}
        if dbid_list:
            embs = BookEmbedding.query.filter(BookEmbedding.book_id.in_(dbid_list)).all()
            for e in embs:
                if e.vector is not None:
                    vector_map[e.book_id] = np.array(__import__("pickle").loads(e.vector) if isinstance(e.vector, bytes) else e.vector, dtype=np.float32)

        # 3. Boost candidates via SessionEncoder
        history_vectors = []
        for gid in history_gids:
            dbid = gid_to_dbid.get(gid)
            if dbid and dbid in vector_map:
                history_vectors.append(vector_map[dbid])

        if history_vectors:
            # instantiate generic encoder (output 384 for direct cosine similarity)
            if not hasattr(current_app, 'session_encoder'):
                current_app.session_encoder = SessionEncoder(item_dim=384, hidden_dim=128, output_dim=384)
                current_app.session_encoder.eval()
            
            encoder = current_app.session_encoder
            
            # pad 10
            while len(history_vectors) < 10:
                history_vectors.append(np.zeros(384, dtype=np.float32))
            history_vectors = history_vectors[-10:]
            
            item_t = torch.tensor(np.array([history_vectors]), dtype=torch.float32)
            action_t = torch.zeros((1, 10), dtype=torch.long)
            time_t = torch.zeros((1, 10, 1), dtype=torch.float32)
            mask_t = torch.ones((1, 10), dtype=torch.bool)
            
            with torch.no_grad():
                session_emb = encoder(item_t, action_t, time_t, mask_t).numpy()[0]
            
            # apply boost
            for c in candidates:
                dbid = c.get('_internal_db_id')
                if dbid and dbid in vector_map:
                    c_vec = vector_map[dbid].reshape(1, -1)
                    sim = cosine_similarity(session_emb.reshape(1, -1), c_vec)[0][0]
                    # boost score
                    old_score = float(c.get('score', 0.5))
                    c['score'] = old_score + float(sim)*0.3  # blending

        # 4. Sort and apply MMR
        candidates.sort(key=lambda x: float(x.get('score', 0.0)), reverse=True)
        
        diverse_candidates = apply_diversity(candidates, vector_map, 0.7)
        
        sampled = diverse_candidates[:limit]
            
        logger.info(f"[Behavior-Hybrid] Returning {len(sampled)} items after SequenceBoost + MMR")
        return sampled

    except Exception as e:
        logger.error(f"[Behavior-Hybrid] Wrapper Error: {e}", exc_info=True)
        return get_trending(limit=limit)


@cache.memoize(timeout=300)
def _get_behavior_based_recommendations_legacy(user_id, limit=12):
    """
    [DEPRECATED] Legacy function replaced by get_behavior_based_recommendations
    """
    from datetime import datetime, timedelta
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        recent_views = (
            UserBookView.query
            .filter_by(user_id=user_id)
            .order_by(UserBookView.last_viewed_at.desc())
            .limit(50)
            .all()
        )
        
        if not recent_views:
            logger.debug(f"[Behavior] No views found for user {user_id}")
            return []
        
        category_weights = defaultdict(float)
        author_weights = defaultdict(float)
        now = datetime.utcnow()
        week_ago = now - timedelta(days=7)
        
        viewed_google_ids = set()
        
        for view in recent_views:
            recency_factor = 1.5 if view.last_viewed_at and view.last_viewed_at > week_ago else 1.0
            view_weight = (view.view_count or 1) * recency_factor
            
            if view.google_id:
                viewed_google_ids.add(view.google_id)
            
            book = None
            if view.book_id:
                book = Book.query.get(view.book_id)
            elif view.google_id:
                book = Book.query.filter_by(google_id=view.google_id).first()
            
            if not book:
                continue
            
            categories = book.categories or ""
            if categories:
                try:
                    import json
                    cats = json.loads(categories) if categories.startswith('[') else categories.split(',')
                except:
                    cats = categories.split(',')
                
                for cat in cats:
                    cat = cat.strip()
                    if cat and len(cat) > 2:
                        category_weights[cat] += view_weight
            
            if book.author:
                first_author = book.author.split(',')[0].strip()
                if first_author and first_author not in ['Unknown', 'مؤلف غير معروف']:
                    author_weights[first_author] += view_weight * 1.5
        
        top_categories = sorted(category_weights.items(), key=lambda x: x[1], reverse=True)[:5]
        top_authors = sorted(author_weights.items(), key=lambda x: x[1], reverse=True)[:3]
        
        logger.info(f"[Behavior] User {user_id} - Top categories: {top_categories[:3]}, Top authors: {top_authors[:2]}")
        
        if not top_categories and not top_authors:
            logger.debug(f"[Behavior] No behavior patterns found for user {user_id}")
            return []
        
        all_recs = []
        seen_ids = set(viewed_google_ids)
        
        def search_by_category(category, weight):
            try:
                items, _ = fetch_google_books(f"subject:{category}", max_results=8)
                results = []
                for it in items or []:
                    gid = it.get("id")
                    if not gid: continue
                    vi = it.get("volumeInfo", {})
                    title = vi.get("title")
                    if not title: continue
                    imgs = vi.get("imageLinks", {}) or {}
                    cover = imgs.get("thumbnail") or ""
                    if cover.startswith("http://"):
                        cover = "https://" + cover[7:]
                    
                    results.append({
                        "id": gid, "title": title,
                        "author": ", ".join(vi.get("authors", [])),
                        "cover": cover, "source": "سلوكك",
                        "reason": f"📚 من تصنيف: {category}",
                        "rating": vi.get("averageRating"),
                        "weight": weight, "type": "category"
                    })
                return results
            except Exception as e:
                logger.error(f"[Behavior] Category search error for {category}: {e}")
                return []
        
        def search_by_author(author, weight):
            try:
                items, _ = fetch_google_books(f"inauthor:{author}", max_results=6)
                results = []
                for it in items or []:
                    gid = it.get("id")
                    if not gid: continue
                    vi = it.get("volumeInfo", {})
                    title = vi.get("title")
                    if not title: continue
                    imgs = vi.get("imageLinks", {}) or {}
                    cover = imgs.get("thumbnail") or ""
                    if cover.startswith("http://"):
                        cover = "https://" + cover[7:]
                    
                    results.append({
                        "id": gid, "title": title,
                        "author": ", ".join(vi.get("authors", [])),
                        "cover": cover, "source": "سلوكك",
                        "reason": f"✍️ أعمال: {author}",
                        "rating": vi.get("averageRating"),
                        "weight": weight * 1.2, "type": "author"
                    })
                return results
            except Exception as e:
                logger.error(f"[Behavior] Author search error for {author}: {e}")
                return []
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for cat, weight in top_categories:
                futures.append(executor.submit(search_by_category, cat, weight))
            for author, weight in top_authors:
                futures.append(executor.submit(search_by_author, author, weight))
            
            from concurrent.futures import as_completed
            for future in as_completed(futures, timeout=10):
                try:
                    results = future.result(timeout=8)
                    for book in results:
                        if book["id"] not in seen_ids:
                            seen_ids.add(book["id"])
                            all_recs.append(book)
                except Exception as e:
                    logger.error(f"[Behavior] Future error: {e}")
        
        all_recs.sort(key=lambda x: x.get("weight", 0), reverse=True)
        
        category_recs = [r for r in all_recs if r.get("type") == "category"]
        author_recs = [r for r in all_recs if r.get("type") == "author"]
        
        final_recs = []
        cat_count = int(limit * 0.6)
        auth_count = limit - cat_count
        
        final_recs.extend(category_recs[:cat_count])
        final_recs.extend(author_recs[:auth_count])
        
        if len(final_recs) < limit:
            remaining = [r for r in all_recs if r not in final_recs]
            final_recs.extend(remaining[:limit - len(final_recs)])
        
        random.shuffle(final_recs)
        
        for rec in final_recs:
            rec.pop("weight", None)
            rec.pop("type", None)
        
        logger.info(f"[Behavior] Generated {len(final_recs)} recommendations for user {user_id}")
        return final_recs[:limit]
        
    except Exception as e:
        logger.error(f"[Behavior] Error: {e}", exc_info=True)
        return []
