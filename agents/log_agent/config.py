import os
import json
 
def get_log_agent_config():
    config_path = os.path.join(os.path.dirname(__file__), '../../config/log_agent_settings.json')
    with open(config_path, 'r') as f:
        return json.load(f) 