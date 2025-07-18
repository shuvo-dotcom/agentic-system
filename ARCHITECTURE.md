# System Architecture Documentation

## Overview

The Agentic System is designed as a modular, multi-agent architecture that processes complex energy analysis queries through specialized agents. Each agent has a specific responsibility and can be called independently or as part of a coordinated workflow.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                     │
├─────────────────────────────────────────────────────────────┤
│                    Task Manager                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Query Router & Orchestrator                ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                    Agent Layer                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ Plexos   │ │   Data   │ │   RAG    │ │ Formula  │      │
│  │ Loader   │ │Harvester │ │ Indexer  │ │Resolver  │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                   │
│  │   Calc   │ │    QC    │ │ Exporter │                   │
│  │Executor  │ │ Auditor  │ │          │                   │
│  └──────────┘ └──────────┘ └──────────┘                   │
├─────────────────────────────────────────────────────────────┤
│                    Data Layer                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ Plexos   │ │ Vector   │ │   SQL    │ │ External │      │
│  │   DB     │ │   DB     │ │   DB     │ │   APIs   │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
├─────────────────────────────────────────────────────────────┤
│                 External Services                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                   │
│  │ OpenAI   │ │ Plexos   │ │ Energy   │                   │
│  │   API    │ │Software  │ │Data APIs │                   │
│  └──────────┘ └──────────┘ └──────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Task Manager (`SimpleTaskManager`)

**Responsibility**: Central orchestrator that routes queries to appropriate agents and manages the overall workflow.

**Key Functions**:
- Query analysis and routing
- Agent coordination and sequencing
- Result aggregation and formatting
- Error handling and recovery
- Task history management

**Workflow**:
1. Receive user query and context
2. Analyze query to determine required agents
3. Execute agents in appropriate sequence
4. Aggregate results and generate final response
5. Log activity and store in history

### 2. Base Agent (`SimpleBaseAgent`)

**Responsibility**: Abstract base class providing common functionality for all specialized agents.

**Key Features**:
- Standardized input/output handling
- OpenAI API integration
- Activity logging and monitoring
- Error handling and response formatting
- Configuration management

**Interface**:
```python
async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Main processing method that all agents must implement"""
    pass
```

## Specialized Agents

### 1. PlexosCSVLoader

**Purpose**: Load and process Plexos energy modeling CSV files

**Capabilities**:
- CSV file parsing and validation
- Data type detection and conversion
- Time series data handling
- Metadata extraction
- Data quality checks

**Input**: File paths, data type specifications
**Output**: Structured data ready for analysis

### 2. DataHarvester

**Purpose**: Retrieve data from external sources and APIs

**Capabilities**:
- Web scraping with rate limiting
- API integration and authentication
- Data caching and storage
- Multi-source data aggregation
- Real-time data updates

**Data Sources**:
- Energy market APIs
- Government databases
- Research publications
- Industry reports

### 3. RAGIndexer

**Purpose**: Build searchable vector and SQL indices for knowledge retrieval

**Capabilities**:
- Document embedding generation
- Vector database management (ChromaDB)
- SQL index creation and optimization
- Hybrid search (vector + SQL)
- Knowledge base maintenance

**Components**:
- Vector embeddings using OpenAI
- Semantic search capabilities
- Structured query processing
- Result ranking and filtering

### 4. FormulaResolver

**Purpose**: Identify required metrics and retrieve canonical formulas

**Capabilities**:
- Metric identification from natural language
- Formula database management
- Parameter extraction and validation
- Formula dependency analysis
- Custom formula registration

**Formula Database**:
- LCOE calculations
- Capacity factor formulas
- Financial metrics (NPV, IRR)
- Environmental calculations
- System performance metrics

### 5. CalcExecutor

**Purpose**: Execute numeric computations using Python sandbox

**Capabilities**:
- Safe code execution environment
- Mathematical formula evaluation
- Time series calculations
- Financial modeling
- Statistical analysis

**Execution Types**:
- Standard formulas
- Time series operations
- Iterative calculations (IRR)
- Custom Python code

### 6. QCAuditor

**Purpose**: Validate results for accuracy, unit consistency, and citations

**Capabilities**:
- Numerical validation and range checking
- Unit consistency verification
- Citation link validation
- Calculation audit trails
- Quality assurance reporting

**Validation Checks**:
- Value range validation
- Unit compatibility
- Precision analysis
- Source verification
- Dimensional consistency

### 7. Exporter

**Purpose**: Format and export final answers in various formats

**Capabilities**:
- Multi-format export (JSON, Markdown, CSV, XML, HTML, PDF)
- Template-based formatting
- Custom styling and layouts
- Metadata inclusion
- File generation and management

**Export Formats**:
- JSON: Structured data with formatting options
- Markdown: Human-readable documentation
- CSV: Spreadsheet-compatible data
- XML: Structured markup
- HTML: Web-ready with styling
- PDF: Professional reports

## Data Flow Architecture

### Typical Query Processing Flow

```
User Query
    ↓
Task Manager (Query Analysis)
    ↓
FormulaResolver (Identify Metric)
    ↓
DataHarvester (Collect Required Data)
    ↓
RAGIndexer (Knowledge Retrieval)
    ↓
CalcExecutor (Perform Calculations)
    ↓
QCAuditor (Quality Control)
    ↓
Exporter (Format Results)
    ↓
Final Response
```

### Data Storage Architecture

#### Vector Database (ChromaDB)
- **Purpose**: Semantic search and document retrieval
- **Content**: Research papers, technical documentation, historical data
- **Access Pattern**: Similarity search, contextual retrieval

#### SQL Database (SQLite/PostgreSQL)
- **Purpose**: Structured data storage and querying
- **Content**: Time series data, calculation results, metadata
- **Access Pattern**: Structured queries, aggregations, joins

#### File System
- **Purpose**: Raw data files, exports, logs
- **Content**: CSV files, generated reports, system logs
- **Access Pattern**: Direct file I/O, batch processing

## Integration Patterns

### Agent Communication

Agents communicate through standardized message formats:

```python
{
    "success": bool,
    "data": Dict[str, Any],
    "metadata": {
        "agent": str,
        "timestamp": str,
        "processing_time": float
    },
    "error": {
        "message": str,
        "type": str,
        "details": Dict[str, Any]
    }
}
```

### Error Handling Strategy

1. **Agent Level**: Individual agents handle their specific errors
2. **Task Manager Level**: Coordinates error recovery and fallback strategies
3. **System Level**: Global error handling and logging

### Scalability Considerations

#### Horizontal Scaling
- Agents can be deployed as separate services
- Message queues for asynchronous processing
- Load balancing for high-throughput scenarios

#### Vertical Scaling
- Optimized database queries and indices
- Caching strategies for frequently accessed data
- Parallel processing within agents

## Security Architecture

### API Security
- OpenAI API key management
- Rate limiting and quota management
- Request/response validation

### Data Security
- Input sanitization and validation
- Safe code execution environments
- Secure file handling

### Access Control
- Agent-level permissions
- Data access restrictions
- Audit logging

## Configuration Management

### Environment Variables
- API keys and credentials
- Database connection strings
- System parameters and thresholds

### Agent Configuration
- Individual agent settings
- Processing parameters
- Quality control thresholds

### Runtime Configuration
- Dynamic parameter adjustment
- Feature flags
- Performance tuning

## Monitoring and Observability

### Logging Strategy
- Structured logging with JSON format
- Agent-specific log namespaces
- Centralized log aggregation

### Metrics Collection
- Processing times and throughput
- Error rates and types
- Resource utilization

### Health Monitoring
- Agent health checks
- Database connectivity
- External service availability

## Extension Points

### Adding New Agents
1. Inherit from `SimpleBaseAgent`
2. Implement `process` method
3. Register with task manager
4. Add routing logic

### Custom Formulas
1. Add to formula database
2. Implement calculation logic
3. Add validation rules
4. Update documentation

### New Data Sources
1. Extend DataHarvester
2. Add authentication logic
3. Implement data parsing
4. Add quality checks

## Performance Optimization

### Database Optimization
- Proper indexing strategies
- Query optimization
- Connection pooling

### Caching Strategy
- Result caching for expensive calculations
- Data caching for frequently accessed information
- Cache invalidation policies

### Parallel Processing
- Concurrent agent execution
- Batch processing for large datasets
- Asynchronous I/O operations

## Deployment Architecture

### Development Environment
- Local SQLite database
- File-based vector storage
- Direct API calls

### Production Environment
- PostgreSQL database
- Distributed vector storage
- Load balancers and API gateways
- Container orchestration

### Cloud Deployment
- Microservices architecture
- Managed databases
- Auto-scaling capabilities
- Monitoring and alerting

This architecture provides a solid foundation for energy analysis workflows while maintaining flexibility for future enhancements and scaling requirements.

