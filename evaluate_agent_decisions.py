from agents.log_agent.log_handler import LogHandler
from agents.log_agent.config import get_log_agent_config

import json

def main():
    import sys
    session_id = sys.argv[1] if len(sys.argv) > 1 else input("Enter session_id: ")
    config = get_log_agent_config()
    log_handler = LogHandler(
        mongo_uri=config.get('mongo_uri', ''),
        mongo_db=config.get('mongo_db', ''),
        mongo_collection=config.get('mongo_collection', ''),
        qdrant_url=config['qdrant_url'],
        openai_api_key=config['openai_api_key'],
        qdrant_collection=config.get('qdrant_collection', 'agent_logs')
    )
    tree_result = log_handler.get_tree_structure(session_id)
    tree = tree_result.get('tree', {})
    import re
    # Find the "finalize" decision node (robust search)
    finalize_node_id = None
    for nid, node in tree.items():
        data = node['data']
        agent = data.get('agent_name', '')
        msg = data.get('message', '')
        if agent and agent != "N/A" and ("Agent" in agent or "TimeSeries" in agent):
            if "decision: finalize" in msg.lower():
                finalize_node_id = nid
                break
    if not finalize_node_id:
        print("No finalize decision node found for this session. Printing all node_ids and messages for debugging:")
        for nid, node in tree.items():
            data = node['data']
            agent = data.get('agent_name', '')
            msg = data.get('message', '')
            print(f"- [{nid}] {agent} | message: {msg[:200]}")
        print("\nIf you see a finalize decision message above, please copy and paste it here so I can adjust the matching logic.")
        return
    # Walk up the tree from finalize_node_id to root
    path = []
    current_id = finalize_node_id
    while current_id:
        node = tree.get(current_id)
        if not node:
            break
        data = node['data']
        agent = data.get('agent_name', '')
        op = data.get('operation', node.get('operation', ''))
        msg = data.get('message', '')
        path.append({
            'node_id': current_id,
            'agent_name': agent,
            'operation': op,
            'message': msg
        })
        parent_id = node.get('parent_id')
        if not parent_id or parent_id == current_id:
            break
        current_id = parent_id
    print("Path taken by OrchestratorAgent to solution (from root to finalize):")
    for step in reversed(path):
        print(f"- [{step['node_id']}] {step['agent_name']} | {step['operation']}")
        print(f"  Message: {step['message']}")
        print("")

if __name__ == "__main__":
    main()
