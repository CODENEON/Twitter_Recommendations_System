from flask import Blueprint, render_template
from flask_login import login_required
import os

network_view = Blueprint('network_view', __name__)

@network_view.route('/network-clusters')
@login_required
def view_network():
    network_path = 'app/static/img/simple_clusters/network.png'
    cluster_path = 'app/static/img/simple_clusters/clusters.png'
    
    if not os.path.exists(network_path) or not os.path.exists(cluster_path):
        return render_template('network_view/error.html',
                             message="Network visualization not found. Please run generate_clusters.py first.")
    
    return render_template('network_view/network.html',
                          network_path='img/simple_clusters/network.png',
                          cluster_path='img/simple_clusters/clusters.png') 