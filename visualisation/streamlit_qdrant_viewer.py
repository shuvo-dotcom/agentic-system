import streamlit as st
import streamlit_mermaid as st_mermaid
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.log_agent.log_handler import LogHandler
import json

st.title('Agentic System Call Tree Visualizer')
st.markdown('---')

session_id = st.text_input('Enter Session ID for Call Tree', '')

# Load Qdrant config and environment variables
with open('config/log_agent_settings.json', 'r') as f:
    log_config = json.load(f)
qdrant_url = log_config['qdrant_url']
# Prioritize environment variable for API key
openai_api_key = os.getenv("OPENAI_API_KEY", log_config.get('openai_api_key', 'YOUR_API_KEY'))
qdrant_collection = log_config.get('qdrant_collection', 'agent_logs')

log_handler = LogHandler(
    mongo_uri=log_config.get('mongo_uri', ''),
    mongo_db=log_config.get('mongo_db', ''),
    mongo_collection=log_config.get('mongo_collection', ''),
    qdrant_url=qdrant_url,
    openai_api_key=openai_api_key,
    qdrant_collection=qdrant_collection
)

def is_agent_node(node):
    agent = node['data'].get('agent_name', '')
    # Only show nodes with a real agent
    return agent and agent != "N/A" and ("Agent" in agent or "TimeSeries" in agent)

if session_id:
    tree_result = log_handler.get_tree_structure(session_id)
    tree = tree_result.get('tree', {})
    # Print all agent_name and operation values for debugging
    st.markdown("#### Debug: All node metadata for this session")
    for nid, n in tree.items():
        st.markdown(f"- [{nid}] metadata: `{json.dumps(n['data'], default=str)}`")
    # Filter to only agent nodes
    agent_tree = {nid: n for nid, n in tree.items() if is_agent_node(n)}
    # Recompute root nodes for agent tree
    root_nodes = [nid for nid, n in agent_tree.items() if not n['parent_id'] or n['parent_id'] not in agent_tree]
    if not agent_tree or not root_nodes:
        st.warning('No agent call tree found for this session.')
    else:
        st.markdown(f"### 🌳 Agent Call Tree for Session `{session_id}`")
        # Indented tree view
        def print_tree(tree, node_id, indent=0):
            node = tree.get(node_id)
            if not node:
                return
            agent = node['data'].get('agent_name', 'Unknown')
            op = node['data'].get('operation', node.get('operation', ''))
            params = node['data'].get('parameters', node.get('parameters', ''))
            # Color for agent type
            if 'Orchestrator' in agent:
                agent_md = f'<span style="color:#1f77b4;font-weight:bold">{agent}</span>'
            elif 'TimeSeries' in agent:
                agent_md = f'<span style="color:#2ca02c;font-weight:bold">{agent}</span>'
            else:
                agent_md = f'<span style="color:#d62728;font-weight:bold">{agent}</span>'
            # Operation highlight
            op_md = f'<span style="background-color:#f9e79f;padding:2px 6px;border-radius:4px">{op}</span>'
            # Parameters pretty
            param_md = ''
            if params:
                param_md = f'<span style="color:#888;font-size:90%">{params}</span>'
            st.markdown(
                f"{'&nbsp;'*4*indent}- [<b>{node_id}</b>] {agent_md}.{op_md} {param_md}",
                unsafe_allow_html=True
            )
            for child_id in node.get('children', []):
                if child_id in tree:
                    print_tree(tree, child_id, indent + 1)
        for root_id in root_nodes:
            print_tree(agent_tree, root_id)
        st.markdown('---')
        # Mermaid graph view
        def build_mermaid(tree):
            lines = ["graph TD"]
            for node_id, node in tree.items():
                agent = node['data'].get('agent_name', 'Unknown')
                op = node['data'].get('operation', node.get('operation', ''))
                lines.append(f'    {node_id}["{agent}\\n{op}"]')
            for node_id, node in tree.items():
                for child_id in node.get('children', []):
                    if child_id in tree:
                        lines.append(f"    {node_id} --> {child_id}")
            return '\n'.join(lines)
        mermaid_code = build_mermaid(agent_tree)
        # Prepend Mermaid init directive for larger font and node size
        mermaid_code = "%%{init: { 'themeVariables': { 'fontSize': '48px' }}}%%\n" + mermaid_code
        st.markdown('#### :art: Graph View (Mermaid)')
        st.code(mermaid_code, language='mermaid')
        st_mermaid.st_mermaid(mermaid_code, height="3600", width="1800")
        if st.button('Export Mermaid as HTML'):
            html_code = f"""
            <html>
            <head>
            <script type='module'>
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{startOnLoad:true}});
            </script>
            </head>
            <body>
            <div class='mermaid'>
            {mermaid_code}
            </div>
            </body>
            </html>
            """
            with open('call_tree_graph.html', 'w') as f:
                f.write(html_code)
            st.success('Exported Mermaid graph to call_tree_graph.html in your project directory.')
else:
    st.info('Enter a session ID to view the call tree.')
