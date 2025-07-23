import streamlit as st
import json
from agents.log_agent.vector_store import QdrantVectorStore
import uuid
import os
from agents.log_agent.log_handler import LogHandler
import streamlit_mermaid as st_mermaid

# Load config
with open(os.path.join('config', 'log_agent_settings.json'), 'r') as f:
    config = json.load(f)
qdrant_url = config['qdrant_url']
openai_api_key = config['openai_api_key']
collection_name = config.get('qdrant_collection', 'agent_logs')

vector_store = QdrantVectorStore(qdrant_url, openai_api_key, collection_name=collection_name)

st.title('Qdrant Log CRUD App')

# Sidebar: Search/Add
st.sidebar.header('Search/Add Log')
search_mode = st.sidebar.radio('Search mode', ['Keyword', 'Semantic'])
search_query = st.sidebar.text_input('Search logs')
search_btn = st.sidebar.button('Search')

st.sidebar.markdown('---')
st.sidebar.header('Add New Log')
new_message = st.sidebar.text_area('Log message')
new_level = st.sidebar.selectbox('Level', ['INFO', 'WARNING', 'ERROR', 'DEBUG'])
add_btn = st.sidebar.button('Add Log')

# Add new log
if add_btn and new_message:
    log_id = str(uuid.uuid4())
    embedding = vector_store.embed_text(new_message)
    payload = {
        'level': new_level,
        'message': new_message,
    }
    vector_store.upsert_log(log_id, new_message, embedding, metadata=payload)
    st.sidebar.success('Log added!')
    st.rerun()

# Search logs
logs = []
if search_btn and search_query:
    if search_mode == 'Keyword':
        logs = vector_store.semantic_search(search_query, top_k=10)
    else:
        logs = vector_store.semantic_search(search_query, top_k=10)
else:
    logs = vector_store.semantic_search('log', top_k=10)

# Main: List logs
st.header('Logs')
if st.button('Delete All Logs', type='primary'):
    vector_store.delete_all_logs()
    st.success('All logs deleted!')
    st.rerun()
selected_ids = set(st.session_state.get('selected_ids', []))
if not logs:
    st.info('No logs found.')
else:
    with st.form('bulk_delete_form'):
        new_selected_ids = set()
        for i, log in enumerate(logs):
            log_id = log.get('_qdrant_id')
            with st.expander(f"{log.get('level', 'INFO')}: {log.get('message', '')[:60]}"):
                checked = st.checkbox('Select', value=(log_id in selected_ids), key=f'select_{i}')
                if checked and log_id:
                    new_selected_ids.add(log_id)
                st.write('**Message:**', log.get('message', ''))
                st.write('**Level:**', log.get('level', ''))
                st.write('**Name:**', log.get('name', ''))
                st.write('**Asctime:**', log.get('asctime', ''))
                st.write('**Pathname:**', log.get('pathname', ''))
                st.write('**Lineno:**', log.get('lineno', ''))
                st.write('**FuncName:**', log.get('funcName', ''))
                st.write('**Qdrant ID:**', log_id or 'N/A')
        st.session_state['selected_ids'] = list(new_selected_ids)
        if st.form_submit_button('Delete Selected'):
            if new_selected_ids:
                for log_id in new_selected_ids:
                    vector_store.delete_log(log_id)
                st.success(f'Deleted {len(new_selected_ids)} logs!')
                st.session_state['selected_ids'] = []
                st.rerun()
            else:
                st.warning('No logs selected for deletion.')
    for i, log in enumerate(logs):
        log_id = log.get('_qdrant_id')
        with st.expander(f"{log.get('level', 'INFO')}: {log.get('message', '')[:60]}"):
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f'Edit_{i}'):
                    st.session_state['edit_log'] = log
                    st.session_state['edit_index'] = i
            with col2:
                if st.button(f'Delete_{i}'):
                    if log_id:
                        vector_store.delete_log(log_id)
                        st.success('Log deleted!')
                        st.rerun()
                    else:
                        st.warning('Log ID not found, cannot delete.')

if 'edit_log' in st.session_state:
    edit_log = st.session_state['edit_log']
    st.header('Edit Log')
    edit_message = st.text_area('Edit message', edit_log.get('message', ''))
    edit_level = st.selectbox('Edit level', ['INFO', 'WARNING', 'ERROR', 'DEBUG'], index=['INFO', 'WARNING', 'ERROR', 'DEBUG'].index(edit_log.get('level', 'INFO')))
    if st.button('Save Changes'):
        log_id = str(uuid.uuid4())
        embedding = vector_store.embed_text(edit_message)
        payload = edit_log.copy()
        payload['message'] = edit_message
        payload['level'] = edit_level
        vector_store.upsert_log(log_id, edit_message, embedding, metadata=payload)
        st.success('Log updated!')
        del st.session_state['edit_log']
        st.rerun()
    if st.button('Cancel Edit'):
        del st.session_state['edit_log']
        st.rerun()

# Only Call Tree Viewer UI remains
st.sidebar.markdown('---')
st.sidebar.header('Call Tree Viewer')
call_tree_session_id = st.sidebar.text_input('Session ID for Call Tree')

if st.sidebar.button('Show ALL Call Trees in Qdrant'):
    st.sidebar.write('---')
    st.sidebar.write('**[DEBUG] All call_tree logs in Qdrant:**')
    all_logs = vector_store.semantic_search('call_tree', top_k=500)
    count = 0
    for log in all_logs:
        if log.get('stage') == 'call_tree' and 'call_tree' in log:
            count += 1
            st.sidebar.write(f"Session: {log.get('session_id')} | Qdrant ID: {log.get('_qdrant_id')}")
            st.sidebar.write(f"call_tree keys: {list(log['call_tree'].keys())}")
    if count == 0:
        st.sidebar.write('No call_tree logs found in Qdrant.')
    else:
        st.sidebar.write(f'Found {count} call_tree logs.')

if st.sidebar.button('Show Call Tree'):
    if call_tree_session_id:
        try:
            # Fetch all call_tree logs and filter by session_id
            all_call_tree_logs = vector_store.semantic_search('call_tree', top_k=500)
            call_tree_log = next(
                (log for log in all_call_tree_logs if log.get('stage') == 'call_tree' and str(log.get('session_id')) == str(call_tree_session_id)),
                None
            )
            st.sidebar.write('---')
            st.sidebar.write('**[DEBUG] All call_tree logs for this session:**')
            for i, log in enumerate(all_call_tree_logs):
                if str(log.get('session_id')) == str(call_tree_session_id):
                    st.sidebar.write(f'Log {i}: {log}')
            st.sidebar.write('---')
            st.sidebar.write(f'**[DEBUG] call_tree_log:** {call_tree_log}')
            if not call_tree_log or 'call_tree' not in call_tree_log:
                st.sidebar.warning('No call tree found for this session.')
            else:
                call_tree = call_tree_log['call_tree']
                st.sidebar.markdown('---')
                st.sidebar.markdown(f"### :deciduous_tree: Call Tree for Session `{call_tree_session_id}`")
                # Graph view with Mermaid
                if st.sidebar.button('Show Graph View'):
                    def build_mermaid(tree):
                        lines = ["graph TD"]
                        for node_id, node in tree.items():
                            label = f"{node_id}({node['agent_name']}\\n{node['operation']})"
                            lines.append(f"    {node_id}[\"{node['agent_name']}\\n{node['operation']}\"]")
                        for node_id, node in tree.items():
                            parent = node.get('parent_id')
                            if parent and parent != 'root' and parent in tree:
                                lines.append(f"    {parent} --> {node_id}")
                        return '\n'.join(lines)
                    mermaid_code = build_mermaid(call_tree)
                    st.sidebar.markdown('#### :art: Graph View (Mermaid)')
                    st.sidebar.markdown('##### [DEBUG] Mermaid code:')
                    st.sidebar.code(mermaid_code, language='mermaid')
                    st_mermaid.mermaid(mermaid_code, height=900)
                    # Export button
                    if st.sidebar.button('Export Mermaid as HTML'):
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
                        st.sidebar.success('Exported Mermaid graph to call_tree_graph.html in your project directory.')
                # Indented tree view as before
                def print_tree(tree, node_id, indent=0):
                    node = tree.get(node_id)
                    if not node:
                        return
                    agent = node['agent_name']
                    op = node['operation']
                    params = node.get('parameters')
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
                    st.sidebar.markdown(
                        f"{'&nbsp;'*4*indent}- [<b>{node_id}</b>] {agent_md}.{op_md} {param_md}",
                        unsafe_allow_html=True
                    )
                    children = [nid for nid, n in tree.items() if n.get('parent_id') == node_id]
                    for child_id in children:
                        print_tree(tree, child_id, indent + 1)
                root_nodes = [nid for nid, n in call_tree.items() if n.get('parent_id') in (None, 'root')]
                for root_id in root_nodes:
                    print_tree(call_tree, root_id)
        except Exception as e:
            st.sidebar.error(f'Error displaying call tree: {e}')
    else:
        st.sidebar.warning('Please enter a session ID.') 