#!/usr/bin/env python3
"""
Enhanced Agentic System Server

A Flask-based server that accepts user prompts and automatically processes them
using the Enhanced Orchestrator Agent with CSV integration capabilities.
"""
import os
import sys
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import threading
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.enhanced_orchestrator_agent import EnhancedOrchestratorAgent

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/server.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def format_orchestrator_response(result: Dict[str, Any], original_query: str) -> str:
    """
    Format the orchestrator response into a beautiful, user-friendly format.
    """
    try:
        # Header
        formatted_response = f"🔍 **Analysis Results for:** {original_query}\n"
        formatted_response += "=" * 60 + "\n\n"
        
        # Check if CSV data was used
        if 'csv_data_used' in result:
            csv_info = result['csv_data_used']
            formatted_response += "📊 **Data Source Integration**\n"
            formatted_response += f"   • File: {csv_info.get('file_name', 'N/A')}\n"
            formatted_response += f"   • Records analyzed: {csv_info.get('data_summary', {}).get('total_records', 'N/A'):,}\n"
            formatted_response += f"   • Selection confidence: {csv_info.get('selection_confidence', 'N/A'):.1%}\n"
            
            # Check for parameter fallback information
            if 'parameter_fallback' in csv_info:
                fallback_info = csv_info['parameter_fallback']
                formatted_response += f"\n⚠️  **Parameter Resolution Fallback**\n"
                formatted_response += f"   • Reason: {fallback_info.get('fallback_reason', 'Unknown')}\n"
                
                if fallback_info.get('unresolved_parameters'):
                    unresolved = fallback_info['unresolved_parameters']
                    formatted_response += f"   • Unresolved from CSV: {', '.join(unresolved)}\n"
                
                if fallback_info.get('will_use_defaults'):
                    formatted_response += f"   • Using: Default/estimated values instead of CSV data\n"
                    formatted_response += f"   • Note: Results may be less accurate due to parameter estimation\n"
            
            # Show key statistics if available
            if 'key_statistics' in csv_info.get('data_summary', {}):
                stats = csv_info['data_summary']['key_statistics']
                if 'value' in stats:
                    value_stats = stats['value']
                    formatted_response += f"   • Data range: {value_stats.get('min', 0):.2f} - {value_stats.get('max', 0):.2f}\n"
                    formatted_response += f"   • Average value: {value_stats.get('average', 0):.2f}\n"
            
            formatted_response += "\n"
        
        # Main results
        results = result.get('results', [])
        if results:
            formatted_response += "⚡ **Calculation Results**\n"
            
            for i, res in enumerate(results, 1):
                res_data = res.get('result', {})
                if isinstance(res_data, dict):
                    # Extract key information
                    formula = res_data.get('formula', 'N/A')
                    parameters = res_data.get('parameters', {})
                    final_result = res_data.get('result', 'N/A')
                    
                    formatted_response += f"\n{i}. **Calculation #{i}**\n"
                    formatted_response += f"   • Formula: {formula}\n"
                    
                    # Show parameters used
                    if parameters:
                        formatted_response += "   • Parameters used:\n"
                        for param, value in parameters.items():
                            if isinstance(value, (int, float)):
                                formatted_response += f"     - {param}: {value:,.6f}\n"
                            else:
                                formatted_response += f"     - {param}: {value}\n"
                    
                    # Show final result
                    if isinstance(final_result, (int, float)):
                        formatted_response += f"   • **Result: {final_result:,.6f}**\n"
                    else:
                        formatted_response += f"   • **Result: {final_result}**\n"
        
        # Sub-prompts analysis
        sub_prompts = result.get('sub_prompts', [])
        if len(sub_prompts) > 1:
            formatted_response += f"\n🔄 **Query Breakdown** ({len(sub_prompts)} sub-queries)\n"
            for i, sub_prompt in enumerate(sub_prompts, 1):
                formatted_response += f"   {i}. {sub_prompt}\n"
        
        # Summary
        summary = result.get('summary', '')
        if summary and summary != 'No results to summarize.':
            formatted_response += f"\n📋 **Summary**\n{summary}\n"
        
        # Answer interpretation for Belgium nuclear generation
        if 'belgium' in original_query.lower() and 'nuclear' in original_query.lower():
            if results and len(results) > 0:
                res_data = results[0].get('result', {})
                final_result = res_data.get('result')
                if isinstance(final_result, (int, float)):
                    # Check if there's temporal validation info
                    csv_info = result.get('csv_data_used', {})
                    temporal_validation = csv_info.get('temporal_validation', {})
                    
                    formatted_response += f"\n💡 **Interpretation**\n"
                    
                    if temporal_validation.get('temporal_mismatch'):
                        formatted_response += f"⚠️  **IMPORTANT TEMPORAL WARNING**: {temporal_validation.get('warning_message', '')}\n"
                        formatted_response += f"🔍 The value {final_result:.2f} TWh is from available data, not the requested time period.\n"
                        formatted_response += f"📅 Suggested action: {temporal_validation.get('suggested_action', '')}\n"
                    else:
                        formatted_response += f"The nuclear generation in Belgium is **{final_result:.2f} TWh** based on the analyzed data.\n"
                        formatted_response += f"This value represents the nuclear power output extracted from real generation data.\n"
        
        # Data source acknowledgment with enhanced transparency and usability validation
        csv_info = result.get('csv_data_used')
        csv_rejected = result.get('csv_data_rejected')
        
        # Check if temporal mismatch or data usability failure was detected
        temporal_mismatch_detected = False
        data_usability_failure = False
        
        if csv_info and 'temporal_validation' in csv_info:
            temporal_validation = csv_info.get('temporal_validation', {})
            temporal_mismatch_detected = temporal_validation.get('temporal_mismatch', False)
            
            # Check for data usability issues 
            data_usability = temporal_validation.get('data_usability', {})
            if data_usability and not data_usability.get('is_usable', True):
                data_usability_failure = True
        
        overall_data_rejection = temporal_mismatch_detected or data_usability_failure

        # Unified Data Quality and Source reporting
        effective_source = result.get('data_source')
        import math
        unusable_result = False
        results_list = result.get('results', [])
        if results_list:
            for r in results_list:
                rd = r.get('result', {})
                fr = rd.get('result', None)
                if fr is None or (isinstance(fr, float) and (math.isnan(fr) or math.isinf(fr))):
                    unusable_result = True
                    break

        if csv_info and not overall_data_rejection:
            formatted_response += f"\n📈 **Data Quality**\n"
            if effective_source == 'LLM' or unusable_result or csv_info.get('used_default_values'):
                formatted_response += f"� **CSV data was not usable - Using LLM estimates**\n"
                formatted_response += f"🤖 LLM defaults used for parameters\n"
                formatted_response += f"⚠️  Results are estimates, not real data\n"
                fallback_info = csv_info.get('parameter_fallback', {})
                if fallback_info:
                    formatted_response += f"\n🔍 **Fallback Details:**\n"
                    formatted_response += f"   • Reason: {fallback_info.get('fallback_reason', 'Unknown')}\n"
                    if fallback_info.get('error'):
                        formatted_response += f"   • Error: {fallback_info['error']}\n"
            else:
                formatted_response += f"✅ **Real data from CSV dataset**\n"
                formatted_response += f"✅ Verified calculation parameters\n"
                formatted_response += f"✅ No assumptions or estimated values\n"
                formatted_response += f"📄 Data source: {csv_info.get('file_name', 'CSV dataset')}\n"

        elif csv_rejected or overall_data_rejection:
            # CSV was available but rejected for temporal mismatch or data usability issues
            formatted_response += f"\n📈 **Data Quality**\n"
            formatted_response += f"🚫 **CSV data rejected - Using LLM estimates**\n"
            
            if temporal_mismatch_detected:
                info_safe = csv_info or {}
                formatted_response += f"📊 CSV file available: {info_safe.get('file_name', 'unknown')}\n"
                formatted_response += f"🕒 **Temporal mismatch detected**\n"
                temporal_validation = info_safe.get('temporal_validation', {})
                if temporal_validation:
                    formatted_response += f"   • Issue: {temporal_validation.get('warning_message', 'Data period mismatch')}\n"
                    formatted_response += f"   • Suggested action: {temporal_validation.get('suggested_action', 'Use data from matching time periods')}\n"
                else:
                    formatted_response += f"   • CSV data period does not match query requirements\n"
                formatted_response += f"   • Action: Rejected CSV, used LLM estimates instead\n"
            elif data_usability_failure:
                info_safe = csv_info or {}
                temp_val = info_safe.get('temporal_validation', {})
                data_usability = temp_val.get('data_usability', {})
                formatted_response += f"📊 CSV file available: {info_safe.get('file_name', 'unknown')}\n"
                formatted_response += f"📉 **Data unusable after filtering**\n"
                formatted_response += f"   • Issue: {data_usability.get('rejection_reason', 'Filtered data is not usable')}\n"
                formatted_response += f"   • Filtered records: {data_usability.get('filtered_rows', 0)}\n"
                formatted_response += f"   • Data quality score: {data_usability.get('data_quality_score', 0):.1%}\n"
                formatted_response += f"   • Action: Rejected CSV, used LLM estimates instead\n"
                # Extract parameter sources from the results to show they are LLM-generated
                results = result.get('results', [])
                if results:
                    formatted_response += f"\n🤖 **Parameter Sources (LLM-Generated Defaults):**\n"
                    for res in results:
                        res_data = res.get('result', {})
                        if isinstance(res_data, dict) and 'parameters' in res_data:
                            params = res_data['parameters']
                            for param_name, param_value in params.items():
                                formatted_response += f"   • {param_name}: {param_value} (LLM estimate)\n"
                formatted_response += f"\n⚠️  **Results are estimates based on typical values, not real data**\n"
                formatted_response += f"⚠️  **For accurate results, use data from matching time periods**\n"
            else:
                # Regular CSV rejection case
                rejected_safe = csv_rejected or {}
                formatted_response += f"📊 CSV file available: {rejected_safe.get('file_name', 'unknown')}\n"
                formatted_response += f"⚠️ Rejection: {rejected_safe.get('rejection_reason', 'Unknown reason')}\n"
                if rejected_safe.get('temporal_mismatch'):
                    formatted_response += f"🕒 **Temporal mismatch detected**\n"
                    temporal_details = rejected_safe.get('temporal_details', {})
                    if temporal_details:
                        if 'query_years' in temporal_details:
                            formatted_response += f"   • Query asks for: {temporal_details['query_years']}\n"
                        if 'available_years' in temporal_details:
                            available = temporal_details['available_years']
                            formatted_response += f"   • Data available for: {available}\n"
                formatted_response += f"\n🤖 **Using LLM-generated estimates instead**\n"
                formatted_response += f"⚠️ **Results are estimates, not real data**\n"
            
        else:
            formatted_response += f"\n🤖 **Processing Method & Sources**\n"
            formatted_response += f"ℹ️  Standard AI processing (no CSV data integration)\n"
            formatted_response += f"🔧 Using LLM-generated default values\n"
            
            # Show parameter sources for non-CSV processing
            results = result.get('results', [])
            if results:
                formatted_response += f"\n🤖 **Parameter Sources (LLM-Generated):**\n"
                for res in results:
                    res_data = res.get('result', {})
                    if isinstance(res_data, dict) and 'parameters' in res_data:
                        params = res_data['parameters']
                        for param_name, param_value in params.items():
                            formatted_response += f"   • {param_name}: {param_value} (LLM default)\n"
        
        return formatted_response
        
    except Exception as e:
        logger.error(f"Error formatting response: {e}")
        # Fallback to a simpler format
        return f"Query: {original_query}\n\nResults:\n{json.dumps(result, indent=2)}"

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global orchestrator instance
orchestrator = None

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Agentic System</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
        }
        .input-section {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            color: #34495e;
        }
        textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #bdc3c7;
            border-radius: 5px;
            font-size: 16px;
            resize: vertical;
            min-height: 100px;
        }
        textarea:focus {
            border-color: #3498db;
            outline: none;
        }
        .button-section {
            text-align: center;
            margin: 20px 0;
        }
        button {
            background-color: #3498db;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #2980b9;
        }
        button:disabled {
            background-color: #95a5a6;
            cursor: not-allowed;
        }
        .loading {
            display: none;
            text-align: center;
            color: #3498db;
            margin: 20px 0;
        }
        .response-section {
            margin-top: 30px;
            padding: 20px;
            background-color: #ecf0f1;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }
        .response-content {
            white-space: pre-wrap;
            font-family: 'SF Pro Display', 'Segoe UI', system-ui, sans-serif;
            line-height: 1.8;
            font-size: 14px;
            color: #2c3e50;
        }
        .response-content strong {
            font-weight: 600;
            color: #1a1a1a;
        }
        .response-content h1, .response-content h2, .response-content h3 {
            color: #2980b9;
            margin: 16px 0 8px 0;
        }
        .error {
            border-left-color: #e74c3c;
            background-color: #fdf2f2;
        }
        .examples {
            margin-top: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .example-item {
            margin: 10px 0;
            padding: 8px;
            background: white;
            border-radius: 3px;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        .example-item:hover {
            background-color: #e3f2fd;
        }
        .status-indicator {
            display: inline-block;
            margin-left: 10px;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }
        .status-csv {
            background-color: #4caf50;
            color: white;
        }
        .status-llm {
            background-color: #ff9800;
            color: white;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Enhanced Agentic System</h1>
        <p class="subtitle">Intelligent Energy Analysis with CSV Integration</p>
        
        <div class="input-section">
            <label for="prompt">Enter your query:</label>
            <textarea id="prompt" placeholder="Ask about energy data, calculations, or general questions...
Examples:
• What is the nuclear generation in Belgium for 2023?
• Calculate the load factor for Spanish energy plants
• Show me energy production data for France
• Explain renewable energy concepts"></textarea>
        </div>
        
        <div class="button-section">
            <button onclick="processQuery()" id="submitBtn">Process Query</button>
        </div>
        
        <div class="loading" id="loading">
            <p>🔄 Processing your query... This may take a moment.</p>
        </div>
        
        <div class="examples">
            <h3>💡 Example Queries:</h3>
            <div class="example-item" onclick="setExample('What is the nuclear generation in Belgium for 2023?')">
                📊 What is the nuclear generation in Belgium for 2023? <span class="status-indicator status-csv">CSV DATA</span>
            </div>
            <div class="example-item" onclick="setExample('Calculate the load factor for nuclear plants in Spain')">
                ⚡ Calculate the load factor for nuclear plants in Spain <span class="status-indicator status-csv">CSV DATA</span>
            </div>
            <div class="example-item" onclick="setExample('Show me energy production trends for France')">
                📈 Show me energy production trends for France <span class="status-indicator status-csv">CSV DATA</span>
            </div>
            <div class="example-item" onclick="setExample('Explain renewable energy concepts and technologies')">
                🌱 Explain renewable energy concepts and technologies <span class="status-indicator status-llm">LLM ONLY</span>
            </div>
        </div>
        
        <div class="response-section" id="response" style="display: none;">
            <h3>Response:</h3>
            <div class="response-content" id="responseContent"></div>
        </div>
    </div>

    <script>
        function setExample(text) {
            document.getElementById('prompt').value = text;
        }
        
        async function processQuery() {
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) {
                alert('Please enter a query');
                return;
            }
            
            const submitBtn = document.getElementById('submitBtn');
            const loading = document.getElementById('loading');
            const responseDiv = document.getElementById('response');
            const responseContent = document.getElementById('responseContent');
            
            // Show loading state
            submitBtn.disabled = true;
            loading.style.display = 'block';
            responseDiv.style.display = 'none';
            
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ prompt: prompt })
                });
                
                const data = await response.json();
                
                // Show response
                responseDiv.style.display = 'block';
                responseDiv.className = 'response-section' + (data.status === 'error' ? ' error' : '');
                
                // Process markdown-style formatting
                let formattedContent = data.status === 'error' ? 
                    `Error: ${data.message}` : 
                    data.response;
                
                // Simple markdown processing for better display
                formattedContent = formattedContent
                    .replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')  // Bold
                    .replace(/\\*(.*?)\\*/g, '<em>$1</em>')              // Italic
                    .replace(/^### (.*$)/gm, '<h3>$1</h3>')           // H3
                    .replace(/^## (.*$)/gm, '<h2>$1</h2>')            // H2
                    .replace(/^# (.*$)/gm, '<h1>$1</h1>')             // H1
                    .replace(/^• (.*$)/gm, '<span style="color: #3498db;">•</span> $1')  // Bullet points
                    .replace(/^✅ (.*$)/gm, '<span style="color: #27ae60;">✅</span> $1')  // Checkmarks
                    .replace(/^📊 (.*$)/gm, '<span style="color: #8e44ad;">📊</span> $1')  // Charts
                    .replace(/^⚡ (.*$)/gm, '<span style="color: #f39c12;">⚡</span> $1')  // Lightning
                    .replace(/^💡 (.*$)/gm, '<span style="color: #e67e22;">💡</span> $1')  // Bulb
                    .replace(/^🔍 (.*$)/gm, '<span style="color: #2980b9;">🔍</span> $1')  // Search
                    .replace(/^📈 (.*$)/gm, '<span style="color: #16a085;">📈</span> $1')  // Chart up
                    .replace(/^🔄 (.*$)/gm, '<span style="color: #9b59b6;">🔄</span> $1')  // Refresh
                    .replace(/^📋 (.*$)/gm, '<span style="color: #34495e;">📋</span> $1'); // Clipboard
                
                responseContent.innerHTML = formattedContent;
                
            } catch (error) {
                responseDiv.style.display = 'block';
                responseDiv.className = 'response-section error';
                responseContent.textContent = `Error: ${error.message}`;
            } finally {
                // Hide loading state
                submitBtn.disabled = false;
                loading.style.display = 'none';
            }
        }
        
        // Allow Enter key to submit (Ctrl+Enter for new line)
        document.getElementById('prompt').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.ctrlKey) {
                e.preventDefault();
                processQuery();
            }
        });
    </script>
</body>
</html>
"""


def initialize_orchestrator():
    """Initialize the Enhanced Orchestrator Agent."""
    global orchestrator
    try:
        logger.info("Initializing Enhanced Orchestrator Agent...")
        orchestrator = EnhancedOrchestratorAgent()
        logger.info("✅ Enhanced Orchestrator Agent initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Enhanced Orchestrator Agent: {e}")
        logger.error(traceback.format_exc())
        return False


@app.route('/')
def index():
    """Serve the web interface."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'orchestrator_ready': orchestrator is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/process', methods=['POST'])
def process_query():
    """Process a user query through the Enhanced Orchestrator Agent."""
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No prompt provided'
            }), 400
        
        user_prompt = data['prompt'].strip()
        if not user_prompt:
            return jsonify({
                'status': 'error',
                'message': 'Empty prompt provided'
            }), 400
        
        logger.info(f"Processing query: {user_prompt[:100]}...")
        
        if orchestrator is None:
            logger.error("Orchestrator not initialized")
            return jsonify({
                'status': 'error',
                'message': 'System not properly initialized'
            }), 500
        
        # Process the query
        start_time = datetime.now()
        
        # The analyze method is async, so we need to run it in an async context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(orchestrator.analyze(user_prompt))
            # Format the response beautifully
            response = format_orchestrator_response(result, user_prompt)
        finally:
            loop.close()
        
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        logger.info(f"Query processed successfully in {processing_time:.2f} seconds")
        
        return jsonify({
            'status': 'success',
            'response': response,
            'processing_time': processing_time,
            'timestamp': end_time.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/process', methods=['POST'])
def api_process_query():
    """API endpoint for programmatic access."""
    return process_query()


@app.route('/status')
def status():
    """Get system status and configuration."""
    try:
        return jsonify({
            'system_status': 'operational',
            'orchestrator_initialized': orchestrator is not None,
            'available_endpoints': [
                '/ - Web interface',
                '/health - Health check',
                '/process - Process query (POST)',
                '/api/process - API endpoint (POST)',
                '/status - System status'
            ],
            'example_queries': [
                'What is the nuclear generation in Belgium for 2023?',
                'Calculate the load factor for nuclear plants in Spain',
                'Show me energy production data for France',
                'Explain renewable energy concepts'
            ],
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


def run_server(host='127.0.0.1', port=5000, debug=False):
    """Run the Flask server."""
    logger.info(f"🚀 Starting Enhanced Agentic System Server...")
    logger.info(f"📍 Server will be available at: http://{host}:{port}")
    logger.info(f"🌐 Web interface: http://{host}:{port}")
    logger.info(f"🔗 API endpoint: http://{host}:{port}/api/process")
    
    # Initialize the orchestrator
    if not initialize_orchestrator():
        logger.error("❌ Failed to initialize orchestrator. Exiting.")
        sys.exit(1)
    
    logger.info("✅ Server initialization complete!")
    logger.info("=" * 60)
    
    # Run the Flask app
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Agentic System Server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("🚀 Enhanced Agentic System Server")
    print("=" * 40)
    print(f"🌐 Starting server on http://{args.host}:{args.port}")
    print("📊 Features:")
    print("  • Smart CSV integration for energy queries")
    print("  • Automatic data extraction and analysis")
    print("  • Web interface and REST API")
    print("  • Real-time query processing")
    print("=" * 40)
    
    run_server(host=args.host, port=args.port, debug=args.debug)
