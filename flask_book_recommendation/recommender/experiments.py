# -*- coding: utf-8 -*-
"""
A/B Testing Framework Functions
===============================
"""
import hashlib
import scipy.stats
import logging
from flask import current_app
from sqlalchemy import func
from ..models import Experiment, ExperimentAssignment, ExperimentMetric, db

logger = logging.getLogger(__name__)

# أول تجربة جاهزة حسب المتطلبات
EXPERIMENT_1 = {
    'name': 'twotower_vs_cf_v1',
    'description': 'هل TwoTower يتفوق على CF في CTR؟',
    'traffic_split': 0.2,  # 20% فقط في البداية للأمان
    'metric': 'click_through_rate',
    'guardrail': 'completion_rate > 0.25'
}

def _get_or_create_experiment(exp_config: dict):
    """
    يضمن وجود التجربة في قاعدة البيانات.
    """
    exp = Experiment.query.filter_by(name=exp_config['name']).first()
    if not exp:
        exp = Experiment(
            name=exp_config['name'],
            description=exp_config['description'],
            traffic_split=exp_config.get('traffic_split', 0.5),
            status='active'
        )
        db.session.add(exp)
        try:
            db.session.commit()
        except BaseException as e:
            db.session.rollback()
            logger.error(f"Error creating experiment {exp_config['name']}: {e}")
            exp = Experiment.query.filter_by(name=exp_config['name']).first()
    return exp

def assign(user_id: int, exp_name: str) -> str:
    """
    يعين المستخدم لمتغير (control أو treatment) بطريقة حتمية (Deterministic).
    التحقق IDempotent يحفظ فقط في المرة الأولى.
    """
    if not user_id:
        return 'control'
        
    exp = Experiment.query.filter_by(name=exp_name).first()
    if not exp:
        if exp_name == EXPERIMENT_1['name']:
            exp = _get_or_create_experiment(EXPERIMENT_1)
        else:
            logger.warning(f"Experiment {exp_name} not found.")
            return 'control'
            
    if exp.status != 'active':
        return getattr(exp, 'winning_variant', 'control') or 'control'
        
    # التحقق مما إذا كان مخصصاً من قبل
    assignment = ExperimentAssignment.query.filter_by(user_id=user_id, experiment_id=exp.id).first()
    if assignment:
        return assignment.variant
        
    # الهاش الحتمي:
    # deterministic: hash(str(user_id) + exp_name) % 100 < split -> treatment
    hash_input = f"{user_id}{exp_name}".encode('utf-8')
    hash_val = int(hashlib.md5(hash_input).hexdigest()[:8], 16)
    
    # تحويل نسبة المرور (traffic_split) إلى نسبة مئوية (مثلاً 0.2 -> 20)
    split_threshold = int(exp.traffic_split * 100)
    
    mod_val = hash_val % 100
    if mod_val < split_threshold:
        variant = 'treatment'
    else:
        variant = 'control'
        
    # الحفظ في قاعدة البيانات
    new_assignment = ExperimentAssignment(user_id=user_id, experiment_id=exp.id, variant=variant)
    db.session.add(new_assignment)
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error assigning user {user_id} to experiment {exp_name}: {e}")
        
    return variant

def get_results(exp_name: str) -> dict:
    """
    يحسب نتائج تجربة A/B والأهمية الإحصائية (Statistical Significance).
    """
    exp = Experiment.query.filter_by(name=exp_name).first()
    if not exp:
        return {"error": "Experiment not found"}
        
    results = {
        "mean_control": 0.0,
        "mean_treatment": 0.0,
        "lift": 0.0,
        "p_value": 1.0,
        "is_significant": False,
        "n_control": 0,
        "n_treatment": 0
    }
    
    # تجميع المقاييس من قاعدة البيانات
    control_metrics = db.session.query(ExperimentMetric.metric_value)\
        .filter_by(experiment_id=exp.id, variant='control').all()
    treatment_metrics = db.session.query(ExperimentMetric.metric_value)\
        .filter_by(experiment_id=exp.id, variant='treatment').all()
        
    control_values = [m[0] for m in control_metrics]
    treatment_values = [m[0] for m in treatment_metrics]
    
    results["n_control"] = len(control_values)
    results["n_treatment"] = len(treatment_values)
    
    if results["n_control"] > 0:
        results["mean_control"] = sum(control_values) / results["n_control"]
        
    if results["n_treatment"] > 0:
        results["mean_treatment"] = sum(treatment_values) / results["n_treatment"]
        
    if results["mean_control"] > 0:
        results["lift"] = ((results["mean_treatment"] - results["mean_control"]) / results["mean_control"]) * 100
        
    # Statistical test:
    if results["n_control"] > 1 and results["n_treatment"] > 1:
        try:
            stat, p_val = scipy.stats.ttest_ind(control_values, treatment_values, equal_var=False)
            results["p_value"] = p_val
            
            # is_significant = p < 0.05 and n_control > 50 and n_treatment > 50
            results["is_significant"] = bool(
                p_val < 0.05 and 
                results["n_control"] > 50 and 
                results["n_treatment"] > 50
            )
        except Exception as e:
            logger.error(f"Error calculating t-test for {exp_name}: {e}")
            
    return results

def log_metric(user_id: int, exp_name: str, metric_name: str, metric_value: float):
    """
    تسجيل مقياس لمستخدم ضمن تجربة.
    """
    exp = Experiment.query.filter_by(name=exp_name).first()
    if not exp:
        return
        
    assignment = ExperimentAssignment.query.filter_by(user_id=user_id, experiment_id=exp.id).first()
    if not assignment:
        # لم يتم تعيين المستخدم بعد للتجربة
        return
        
    metric = ExperimentMetric(
        experiment_id=exp.id,
        variant=assignment.variant,
        metric_name=metric_name,
        metric_value=metric_value
    )
    db.session.add(metric)
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error logging metric for {exp_name}: {e}")
