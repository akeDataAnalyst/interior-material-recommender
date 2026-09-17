# AI-Powered Interior Material Recommender & Matching Engine
[![Live Demo](https://img.shields.io/badge/Streamlit-Live%20Demo-brightgreen)](https://ai-interior-material-recommender-zgpgdb84fz5znjushytlts.streamlit.app/)


An intelligent recommendation system that helps users find matching interior finishing materials (tiles, paints, glass, sanitary ware, etc.) based on natural language queries (style, color, budget, room type, material preferences).

## Project Highlights

- Web scraping & data collection from Ethiopian source - https://con.2merkato.com/prices (Construction Market Watch)  
- Data cleaning & feature engineering  
- Text embeddings & similarity search (Hugging Face sentence-transformers)  
- Interactive recommendation engine  
- Modern Streamlit dashboard with real-time matching

## Features

- Natural language input: "modern blue tiles under 2000 ETB", "white marble for kitchen counter", "frosted glass partition", etc.  
- Semantic similarity search using sentence embeddings (all-MiniLM-L6-v2)  
- Ranked recommendations with similarity scores (0–1)  
- Price formatting (ETB), unit normalization, category filtering  
- Adjustable parameters: number of results, minimum similarity threshold  
- Data from real Ethiopian market (2merkato.com)

## Tech Stack

| Layer                 | Tools / Libraries                              |
|-----------------------|------------------------------------------------|
| Language              | Python 3.10+                                   |
| Data Processing       | pandas, numpy                                  |
| Embeddings & Similarity | sentence-transformers (all-MiniLM-L6-v2), scikit-learn |
| Dashboard             | Streamlit                                      |
| Data Format           | Parquet (embeddings), CSV (backup)             |


## Results & Performance

- Dataset: 299 real Ethiopian interior/finishing materials (Finishing, Sanitary, Electrical, Tiles & Ceramics, Roofing & Ceiling)  
- Price range: 15 – 146,500 ETB (mean 3,559 ETB)  
- Embedding model: all-MiniLM-L6-v2 (384 dimensions)  
- Similarity performance:  Exact/similar items (e.g. Clear Glass variants) score 0.98–0.99  
- Related items (frosted glass, galvanized sheets) score 0.55–0.70  
- Threshold 0.4–0.45 gives 4–8 relevant matches per query

