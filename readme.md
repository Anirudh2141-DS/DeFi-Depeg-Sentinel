# DeFi Depeg Sentinel

A real-time watchdog that monitors stablecoin pools and flags depeg risks using anomaly detection, volatility signatures, and liquidity shifts. Designed for LPs, risk teams, and DeFi protocols that need early warnings before the market blows up.

---

## 🎯 What it does
- Tracks pool prices, peg stability, liquidity depth, and volume  
- Identifies abnormal deviations using statistical + ML anomaly models  
- Generates "depeg alerts" with severity scores  
- Visualizes risk timelines & pool health  
- (Optional) Tokenized reward flow for those who catch depegs early  

---

## ⚙️ Tech Stack
**Python** · pandas · scikit-learn · NumPy  
**ML Models:** Isolation Forest · Z-Score Volatility · Rolling Residual Deviation  
**Visualization:** Streamlit / Plotly  
**DeFi Data:** APIs like CoinGecko, DexScreener, Uniswap Subgraph  

---

## 📊 Architecture
1. **Ingestion Layer**  
   Pulls live stablecoin and pool metrics.

2. **Feature Engine**  
   Builds volatility windows, depth deltas, price deviation curves.

3. **Anomaly Detector Suite**  
   - Stat-model layer (Z-score)  
   - ML layer (Isolation Forest)  
   - Blended score aggregator

4. **Risk Scoring System**  
   Outputs: green, yellow, orange, red (severe depeg).

5. **Visualization UI**  
   Streamlit dashboard with live charts and alerts.

---

## 🚀 Quickstart
```bash
git clone https://github.com/Anirudh2141-DS/DeFi-Depeg-Sentinel
cd DeFi-Depeg-Sentinel
pip install -r requirements.txt
streamlit run app.py
