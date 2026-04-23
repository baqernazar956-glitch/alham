# routes/api/ai.py
"""
API AI endpoints - الدردشة والملخصات والمسابقات
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
import requests

# استيراد دوال AI الموجودة
from flask_book_recommendation.utils import (
    chat_with_ai,
    generate_ai_description,
    generate_book_summary as utils_generate_summary
)

api_ai_bp = Blueprint('api_ai', __name__, url_prefix='/ai')


@api_ai_bp.route('/health', methods=['GET'])
def ai_health():
    """
    حالة نظام الذكاء الاصطناعي
    GET /api/ai/health
    """
    result = {
        "status": "online",
        "local_scorer": {"status": "not_loaded"}
    }
    
    try:
        from ...ai_client import ai_client
        health = ai_client.get_health()
        result.update(health)
    except Exception as e:
        result["error"] = str(e)
    
    return jsonify(result)


def get_book_info(gid: str) -> dict:
    """جلب معلومات الكتاب من Google Books"""
    try:
        resp = requests.get(f"https://www.googleapis.com/books/v1/volumes/{gid}", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            info = data.get('volumeInfo', {})
            return {
                'title': info.get('title', ''),
                'author': ', '.join(info.get('authors', [])),
                'description': info.get('description', ''),
                'categories': info.get('categories', [])
            }
    except:
        pass
    return {}


@api_ai_bp.route('/chat', methods=['POST'])
@jwt_required(optional=True)
def general_chat():
    """
    محادثة AI عامة
    POST /api/ai/chat
    Body: {"message": "اقترح لي كتب عن البرمجة"}
    """
    data = request.get_json() or {}
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({
            'success': False,
            'error': 'يرجى إدخال رسالة'
        }), 400
    
    try:
        if chat_with_ai:
            response = chat_with_ai(message)
            return jsonify({
                'success': True,
                'response': response
            })
        else:
            return jsonify({
                'success': False,
                'error': 'خدمة AI غير متاحة'
            }), 503
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_ai_bp.route('/book/<gid>/chat', methods=['POST'])
@jwt_required()
def book_chat(gid: str):
    """
    محادثة AI عن كتاب معين
    POST /api/ai/book/<gid>/chat
    Body: {"message": "ما هي أهم أفكار هذا الكتاب؟"}
    """
    data = request.get_json() or {}
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({
            'success': False,
            'error': 'يرجى إدخال رسالة'
        }), 400
    
    # جلب معلومات الكتاب
    book_info = get_book_info(gid)
    if not book_info.get('title'):
        return jsonify({
            'success': False,
            'error': 'الكتاب غير موجود'
        }), 404
    
    # بناء السياق
    context_message = f"""
    أنت مساعد خبير بالكتب. المستخدم يسأل عن كتاب:
    العنوان: {book_info.get('title')}
    المؤلف: {book_info.get('author')}
    الوصف: {book_info.get('description', '')[:500]}
    
    سؤال المستخدم: {message}
    
    أجب بشكل مفيد ومختصر بنفس لغة المستخدم (إذا كان السؤال بالإنجليزية أجب بالإنجليزية، وإذا كان بالعربية أجب بالعربية).
    """
    
    try:
        if chat_with_ai:
            response = chat_with_ai(context_message)
            return jsonify({
                'success': True,
                'response': response,
                'book': {
                    'title': book_info.get('title'),
                    'author': book_info.get('author')
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'خدمة AI غير متاحة'
            }), 503
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_ai_bp.route('/book/<gid>/summary', methods=['GET'])
def book_summary(gid: str):
    """
    ملخص AI للكتاب
    GET /api/ai/book/<gid>/summary
    """
    book_info = get_book_info(gid)
    if not book_info.get('title'):
        return jsonify({
            'success': False,
            'error': 'الكتاب غير موجود'
        }), 404
    
    try:
        if utils_generate_summary:
            summary_res = utils_generate_summary(book_info)
            summary = summary_res.get('summary') if isinstance(summary_res, dict) else summary_res
            return jsonify({
                'success': True,
                'summary': summary,
                'book': {
                    'title': book_info.get('title'),
                    'author': book_info.get('author')
                }
            })
        elif chat_with_ai:
            prompt = f"""
            اكتب ملخصاً موجزاً (3-4 فقرات) لكتاب:
            العنوان: {book_info.get('title')}
            المؤلف: {book_info.get('author')}
            الوصف: {book_info.get('description', '')}
            
            اجعل الملخص مفيداً وجذاباً بالعربية.
            """
            summary = chat_with_ai(prompt)
            return jsonify({
                'success': True,
                'summary': summary,
                'book': {
                    'title': book_info.get('title'),
                    'author': book_info.get('author')
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'خدمة AI غير متاحة'
            }), 503
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_ai_bp.route('/book/<gid>/quotes', methods=['GET'])
def book_quotes(gid: str):
    """
    اقتباسات من الكتاب
    GET /api/ai/book/<gid>/quotes
    """
    book_info = get_book_info(gid)
    if not book_info.get('title'):
        return jsonify({
            'success': False,
            'error': 'الكتاب غير موجود'
        }), 404
    
    try:
        if chat_with_ai:
            prompt = f"""
            اكتب 5 اقتباسات ملهمة من كتاب "{book_info.get('title')}" للمؤلف {book_info.get('author')}.
            إذا لم تكن تعرف اقتباسات حقيقية، اكتب اقتباسات تعكس روح الكتاب وموضوعه.
            
            اكتب كل اقتباس في سطر منفصل مع علامات تنصيص.
            بالعربية فقط.
            """
            response = chat_with_ai(prompt)
            # تحويل النص إلى قائمة
            quotes = [q.strip().strip('"').strip('"').strip('"') 
                     for q in response.split('\n') if q.strip()]
            
            return jsonify({
                'success': True,
                'quotes': quotes[:5],
                'book': {
                    'title': book_info.get('title'),
                    'author': book_info.get('author')
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'خدمة AI غير متاحة'
            }), 503
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_ai_bp.route('/book/<gid>/quiz', methods=['GET'])
def book_quiz(gid: str):
    """
    مسابقة عن الكتاب
    GET /api/ai/book/<gid>/quiz
    """
    book_info = get_book_info(gid)
    if not book_info.get('title'):
        return jsonify({
            'success': False,
            'error': 'الكتاب غير موجود'
        }), 404
    
    try:
        if chat_with_ai:
            prompt = f"""
            أنشئ 3 أسئلة اختيار من متعدد عن كتاب "{book_info.get('title')}" للمؤلف {book_info.get('author')}.
            
            لكل سؤال:
            - السؤال
            - 4 خيارات (أ، ب، ج، د)
            - الإجابة الصحيحة
            
            اكتب بالعربية بتنسيق JSON:
            [
                {{
                    "question": "...",
                    "options": ["أ) ...", "ب) ...", "ج) ...", "د) ..."],
                    "correct": 0
                }}
            ]
            """
            response = chat_with_ai(prompt)
            
            # محاولة تحليل JSON
            import json
            try:
                # البحث عن JSON في النص
                start = response.find('[')
                end = response.rfind(']') + 1
                if start >= 0 and end > start:
                    quiz = json.loads(response[start:end])
                else:
                    quiz = []
            except:
                quiz = []
            
            return jsonify({
                'success': True,
                'quiz': quiz,
                'book': {
                    'title': book_info.get('title'),
                    'author': book_info.get('author')
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'خدمة AI غير متاحة'
            }), 503
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_ai_bp.route('/book/<gid>/why-like', methods=['GET'])
@jwt_required()
def why_like_book(gid: str):
    """
    لماذا قد يعجبك هذا الكتاب (مخصص للمستخدم)
    GET /api/ai/book/<gid>/why-like
    """
    from ...models import UserPreference
    
    user_id = int(get_jwt_identity())
    book_info = get_book_info(gid)
    
    if not book_info.get('title'):
        return jsonify({
            'success': False,
            'error': 'الكتاب غير موجود'
        }), 404
    
    # جلب اهتمامات المستخدم
    prefs = UserPreference.query.filter_by(user_id=user_id).all()
    interests = [p.topic for p in prefs]
    
    try:
        if chat_with_ai:
            prompt = f"""
            بناءً على اهتمامات المستخدم: {', '.join(interests) if interests else 'غير محددة'}
            
            اشرح في 3-4 نقاط لماذا قد يستمتع بقراءة كتاب:
            العنوان: {book_info.get('title')}
            المؤلف: {book_info.get('author')}
            التصنيفات: {', '.join(book_info.get('categories', []))}
            
            اكتب بأسلوب تشجيعي ومقنع بالعربية.
            """
            response = chat_with_ai(prompt)
            
            return jsonify({
                'success': True,
                'reasons': response,
                'book': {
                    'title': book_info.get('title'),
                    'author': book_info.get('author')
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'خدمة AI غير متاحة'
            }), 503
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
