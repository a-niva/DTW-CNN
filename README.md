# DTW-CNN
CNN-accelerated DTW patterns spotting, using vector embedding and FAISS index

## Overview

A comprehensive algotrading framework system that leverages **deep learning embeddings** and **vector similarity search** to identify market patterns and predict asset returns. The framework combines neural time series encoding, FAISS-based similarity matching, **Dynamic Time Warping (DTW)**, and advanced clustering techniques to provide quantitative insights for algorithmic trading and portfolio management.

## Key Features

### 🧠 Neural Time Series Encoding
- **Multi-scale CNN layers** with adaptive kernel sizes (3, 7, 15) for temporal feature extraction
- **Self-attention mechanism** for capturing long-range dependencies in financial time series
- **Bidirectional GRU** networks for asymmetric temporal modeling
- **Mixed-precision training** with GPU optimization for large-scale financial datasets

### 🔍 Vector Similarity Search
- **FAISS indexing** with IVF clustering for sub-linear similarity search
- **128-dimensional embeddings** optimized for financial pattern representation
- **Adaptive batch processing** with GPU memory management
- **Incremental index updates** for real-time market data integration

### 📈 Dynamic Time Warping (DTW)
- **FastDTW implementation** with adaptive radius constraints
- **Financial-specific distance metrics** with volatility adjustments
- **Pruning algorithms** for computational efficiency
- **Multi-scale pattern matching** across different time horizons

### ⚙️ Bayesian Weight Optimization
- **Gaussian Process optimization** using scikit-optimize for hyperparameter tuning
- **Multi-objective evaluation** combining MAPE, directional accuracy, calibration, and pattern variance
- **Adaptive sampling strategies** with convergence detection
- **Cross-validation** on historical market regimes

### 📊 Advanced Pattern Analysis
- **FLASC clustering** (Flare-based Spatial Clustering) for trajectory grouping
- **Market regime classification** (uptrend, downtrend, sideways, volatility expansion/contraction)
- **Outlier detection** using IQR-based filtering with adaptive bounds
- **Weighted return aggregation** with similarity-based confidence scoring

## Architecture

### System Overview
The framework follows a three-layer architecture optimized for financial time series analysis:

#### **Data Layer**
- **Parquet Data Storage**: High-performance columnar format for OHLCV data
- **Technical Indicators**: RSI, MACD, SMA/EMA computed on-demand
- **Preprocessing Pipeline**: Robust normalization with NaN handling and outlier detection

#### **Encoding Layer** 
- **CNN Feature Extraction**: Multi-scale convolutions (3, 7, 15 kernels) for temporal patterns
- **Self-Attention Mechanism**: Captures long-range dependencies in price movements
- **GRU Sequence Modeling**: Bidirectional processing for asymmetric market behavior
- **Embedding Generation**: 128-dimensional vectors optimized for financial similarity

#### **Analysis Layer**
- **FAISS Vector Search**: Sub-linear similarity search across millions of patterns
- **DTW Pattern Matching**: Dynamic time warping for robust temporal alignment
- **FLASC Clustering**: Advanced trajectory grouping with flare detection
- **Return Prediction**: Weighted aggregation with confidence estimation

### Component Integration

| Component | Input | Output | Technology |
|-----------|-------|--------|------------|
| **Data Manager** | Raw OHLCV | Normalized features | Pandas + PyArrow |
| **Neural Encoder** | Feature windows | 128D embeddings | PyTorch + CUDA |
| **Vector Index** | Embeddings | Similarity rankings | FAISS-GPU |
| **DTW Analyzer** | Price sequences | Alignment scores | FastDTW |
| **FLASC Clusterer** | Similar patterns | Trajectory groups | NetworkX |
| **Return Predictor** | Pattern clusters | Expected returns | Weighted aggregation |


## Financial Metrics Processing

The system processes **10 core financial metrics** with specialized normalization:

| Metric | Processing Method | Weight Range |
|--------|------------------|--------------|
| **OHLC Prices** | Log-return normalization with ATR scaling | 0.1 - 2.0 |
| **Volume** | Log-transformation with robust z-score | 0.1 - 2.0 |
| **RSI14** | Non-linear transformation emphasizing extremes | 0.1 - 2.0 |
| **MACD** | ATR-normalized momentum signals | 0.1 - 2.0 |
| **SMA50/EMA20** | Price-relative normalization | 0.1 - 2.0 |

## Algorithms & Techniques

### Time Series Encoding
- **Multi-head attention** with 2 heads for computational efficiency
- **Temporal decay weighting** (linear, exponential, quadratic)
- **Gradient clipping** and **mixed-precision** training for stability

### Similarity Computation
- **Cosine similarity** on normalized embeddings with NaN handling
- **DTW distance** with adaptive radius and volatility penalties
- **Coverage penalty** for incomplete pattern matches
- **Time-weighted similarity** emphasizing recent market behavior

### Clustering & Forecasting
- **Community detection** using Louvain algorithm on similarity graphs
- **Adaptive radius calculation** based on local density estimation
- **Weighted return aggregation** with outlier filtering

## Output & Visualizations

### Quantitative Outputs
- **Expected return** with confidence intervals
- **Directional probability** (bullish/bearish)
- **Pattern similarity scores** (0-1 range)
- **Cluster-based forecasts** with uncertainty quantification

### Visual Analytics
- **FLASC trajectory plots** showing historical and predicted price paths
- **Outcome distribution histograms** with KDE smoothing
- **Optimization convergence plots** tracking hyperparameter evolution
- **t-SNE embeddings visualization** for pattern space exploration

## Configuration & Performance

### Scalability
- **Streaming data processing** with configurable chunk sizes (1K-16K samples)
- **LRU caching** with memory-aware eviction (configurable capacity)
- **Parallel processing** using ThreadPoolExecutor for I/O operations
- **GPU memory optimization** with automatic fallback to CPU

### Key Parameters
```python
EMBEDDING_DIM = 128          # Vector dimension
FORECAST_HORIZON = 20        # Prediction timeframe  
N_SIMILAR = 40              # Similar patterns to find
BATCH_SIZE = 1024           # GPU batch size
CACHE_CAPACITY_MB = 2000    # Memory cache limit
DTW_WINDOW = 0.1            # DTW radius constraint
DTW_USE_FAST = True         # FastDTW acceleration

Requirements

Python 3.8+
PyTorch 2.0+ with CUDA support
FAISS-GPU for similarity search
scikit-optimize for Bayesian optimization
HDF5/PyTables for embedding storage
fastdtw for Dynamic Time Warping

Performance Benchmarks
OperationDataset SizeExecution TimeMemory UsageModel Training50K windows~45 minutes8GB GPUEmbedding Generation100K patterns~15 minutes4GB GPUSimilarity Search1M embeddings<1 second2GB RAMDTW Pattern Matching1K comparisons~10 seconds1GB RAMPattern Analysis1K similar patterns~30 seconds1GB RAM

This framework is designed for quantitative researchers and algorithmic traders seeking robust, scalable solutions for financial pattern recognition and return prediction.
Basic data requirements are needed, do not hesitate to contact me for more information.
