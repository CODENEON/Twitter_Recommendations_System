from flask import Blueprint, jsonify, current_app
from flask_login import login_required, current_user

api_bp = Blueprint('api', __name__)

@api_bp.route('/recommendations')
@login_required
def get_recommendations():
    """Get user and hashtag recommendations"""
    user_recommendations = current_app.neo4j.get_user_recommendations(current_user.id)
    hashtag_recommendations = current_app.neo4j.get_hashtag_recommendations(current_user.id)
    
    return jsonify({
        'user_recommendations': user_recommendations,
        'hashtag_recommendations': hashtag_recommendations
    })

@api_bp.route('/similar-users')
@login_required
def get_similar_users():
    """Get users with similar interaction patterns"""
    similar_users = current_app.neo4j.find_similar_users(current_user.id)
    return jsonify({'similar_users': similar_users})

@api_bp.route('/communities')
@login_required
def get_communities():
    """Get detected communities"""
    communities = current_app.neo4j.detect_communities()
    return jsonify({'communities': communities})

@api_bp.route('/hashtag-relationships')
@login_required
def analyze_hashtags():
    """Analyze hashtag relationships"""
    current_app.neo4j.analyze_hashtag_relationships()
    return jsonify({'status': 'success', 'message': 'Hashtag relationships analyzed'}) 