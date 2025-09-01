Sentinel Depeg Forecasting System
Overview
The Sentinel Depeg Forecasting System is a real-time monitoring and forecasting pipeline for stablecoin liquidity pools. It continuously tracks on-chain data and off-chain signals to detect anomalies (potential depeg events) and produce calibrated risk forecasts. The system integrates multiple anomaly detectors and machine learning models to warn of impending stablecoin “depeg” events (when a stablecoin strays from its peg). It provides a Streamlit dashboard for visualization and an optional on-chain reporting mechanism (via a smart contract token) to reward timely anomaly detection. In summary, Sentinel is an end-to-end solution that detects anomalies, predicts risk, calibrates the confidence of those predictions, and presents the results in both a web dashboard and on the blockchain for transparent, incentivized reporting.
System Architecture
The Sentinel system is composed of several components working in concert, as shown below:
Anomaly Detectors (Parallel “Zoo” of Detectors)
Sentinel runs a suite of anomaly detection algorithms in parallel on live pool data. Each detector produces a score (0.0 to 1.0) indicating how unusual the latest pool state is:
z_if (Isolation Forest): Learns a baseline of normal behavior and assigns a higher anomaly score to outlier observations.
z_lof (Local Outlier Factor): Detects anomalies by comparing the local density of each data point to that of its neighbors.
z_ocsvm (One-Class SVM): A one-class support vector machine that distinguishes the normal data cluster from outliers.
z_cusum (Cumulative Sum): A statistical detector that flags a shift in the mean of a sequence (useful for detecting abrupt depeg movements in price deviation).
z_ae (Autoencoder): A neural network that attempts to reconstruct input features; large reconstruction error indicates an anomaly in pool metrics.
z_ae_seq (Sequential Autoencoder): A sequence model (e.g. LSTM autoencoder) that captures temporal patterns; it flags anomalies in the time-series sequence of pool states.
These detectors run in parallel on incoming data (thus the term “score zoo”). Each yields an independent anomaly score for the current state of each pool.
Anomaly Score Fusion
To combine the detectors’ outputs into a single robust anomaly indicator, Sentinel employs a fusion logic. The simplest fusion rule (used by default) is a maximization: the fused anomaly score anom_fused is the maximum of all individual detector scores for a given pool at a given time. This ensures that if any one detector is highly confident of an anomaly, the fused score reflects it (i.e. anom_fused = max(z_if, z_lof, z_ocsvm, z_cusum, z_ae, z_ae_seq)). The fused score lies in [0,1] and acts as an aggregate anomaly metric:
Low values (near 0): The pool appears normal across all detectors.
High values (near 1): At least one detector strongly believes the pool state is abnormal (potential depeg developing).
Note: Other fusion strategies (e.g. averaging or weighted voting) can be configured if needed. The max fusion is chosen to maximize recall (catch any potential issue), with the trade-off of possibly lower precision (more false positives). The system’s nightly evaluation reports help tune this fusion logic.
Risk Forecasters (10m & 30m Ahead)
Sentinel includes machine learning forecasters that predict the probability of a depeg event in the near future. Specifically:
10-minute Forecaster: An XGBoost classification model that outputs the probability of a significant depeg event occurring in the next 10 minutes.
30-minute Forecaster: A similar XGBoost model for a 30-minute horizon.
These models take a vector of engineered features for each pool and time window. Key features include:
Deviation metrics: e.g. dev (current price deviation of the pool from the peg or expected price), dev_roll_std (recent volatility of that deviation).
Liquidity flows: e.g. tvl_outflow_rate (rate of liquidity leaving the pool).
Price differences: e.g. spot_twap_gap_bps (difference between the pool’s spot price and a time-weighted average price, in basis points), oracle_ratio (ratio of pool price to an external oracle price).
Anomaly indicators: the fused anomaly score anom_fused and possibly individual detector outputs.
Cross-pool signals: e.g. features from other related pools (neighbor pool max anomaly, average anomaly, correlation and lead/lag with peers) to capture systemic moves.
Each forecaster is trained on historical data labeled with whether a depeg “incident” occurred in that horizon. For example, the 10m model’s label y_10m is 1 if within 10 minutes a notable depeg happened, otherwise 0. (See Calibration Notes below for how incidents are defined by thresholds like dev > 0.005.) The models use gradient boosted trees (XGBoost) to capture non-linear interactions between features. They output a probability (0 to 1) that the pool will experience a depeg event in the given timeframe.
Probability Calibrators (Isotonic Regression)
Raw probabilities from the forecasters may be uncalibrated – e.g. a model might consistently output 0.2 when the true frequency of events is 0.1. To address this, Sentinel applies isotonic regression calibration on the model outputs:
After training each XGBoost forecaster, an isotonic regression model is fitted on the validation set, mapping the model’s raw prediction to a calibrated probability. This ensures that if the model says “0.8” risk, approximately 80% of those cases truly resulted in an event historically.
The calibrated models (for 10m and 30m) are saved and used at runtime to adjust forecast outputs. The result is more reliable risk probabilities that reflect actual odds.
In addition, the system may calibrate severity scores using isotonic regression or similar techniques. For example, if severity of an event (how far the peg breaks) is treated as a predicted quantity or category, an isotonic calibrator can map a raw severity predictor to a probability of a “critical” versus “mild” outcome. This calibration of severity ensures the system’s internal confidence in how severe an anomaly is (magnitude of deviation) is well-aligned with reality.
Meta-Controller & Policy Logic
A high-level Meta-Controller oversees the detectors and forecasters to make policy decisions and manage the model lifecycle:
Alert Level Decision: The meta-controller ingests the latest fused anomaly scores, forecasted risk probabilities, and other contextual flags (like data feed status) to determine an overall alert level for each pool or the system as a whole. It implements policy rules to classify the situation as, for example:
Green (Normal): No action needed (risks below thresholds).
Yellow (Warning): Elevated risk or anomaly (e.g. forecast > 0.5 or moderate anomaly) – monitor closely.
Red (Critical): High risk of depeg or an ongoing depeg (e.g. anomaly score ~1 and forecast > 0.9) – trigger mitigation or alerts.
Confidence and Multi-signal Analysis: The meta logic also evaluates the consistency of signals. For instance, if anom_fused is very high and forecast probability is high, the system might label “Confidence High” in its alert. If signals conflict (e.g. anomaly high but model predicts low risk), it may label it as uncertain or “Confidence Medium/Low”, affecting how operators respond.
Consecutive Event Tracking: Policy rules can incorporate memory of recent alerts (e.g. “3 consecutive red alerts”) to decide on escalations. This prevents flickering signals from causing constant on-chain writes or mitigations – instead, sustained issues trigger stronger actions.
Automated Mitigation Actions: While Sentinel itself doesn’t alter pool parameters, it can suggest actions in its Analyst Note (see Dashboard) or through its API. For example, it might recommend “if risk stays high for 5 minutes or if 3 red alerts occur in a row, consider pausing the pool or adjusting swap parameters.” These suggestions are encoded as simple rules in the meta-controller and can be customized per deployment.
Retrain and Drift Monitoring: The meta-controller continuously monitors data drift and model performance to decide if retraining is necessary. A /policy/retrain_check endpoint is provided (returns e.g. {"should_retrain": true/false, ...}) based on:
Feature Distribution Drift: Uses statistical tests (Population Stability Index, Kolmogorov–Smirnov) on recent data vs. training data (see Retrain Strategy below).
Scheduled Intervals: e.g. force retraining nightly or weekly regardless, to incorporate new data.
Model performance degradation: e.g. if the detectors’ precision-recall metrics drop or if the forecasters consistently mispredict recent events (this can be tracked via rolling evaluation of predictions).
Nightly Reports and Snapshots: A background job (the “nightly” meta-controller routine) generates a daily report. This includes:
The day’s best performing detector (based on precision-recall or AP scores on that day’s data) – useful for understanding which anomaly method was most effective.
Calibration plots showing how well the forecast probabilities aligned with actual outcomes (to ensure calibration remains valid).
Incident summary – list of any notable depeg events detected in the last 24 hours.
This report is saved (as a Markdown and PDF file) for analysts. Additionally, the meta-controller produces an Analyst Note snapshot in real-time when an alert triggers, summarizing the situation in plain language (used in the dashboard).
In essence, the meta-controller is the “brain” that interprets the raw analytics and ensures the system remains up-to-date and actionable.
Streamlit Dashboard Interface
For an immediate visual insight into the system’s output, Sentinel provides an integrated Streamlit dashboard. This web application displays real-time analytics in a user-friendly way:
Time-Series Plots: Key metrics like dev (depeg deviation) and anom_fused are plotted over time for each monitored pool. This lets users see when anomalies spike and how the pool’s price or liquidity is moving.
Risk Gauges or Indicators: The current 10-minute and 30-minute risk probabilities are shown (e.g. as dials or colored indicators) to quickly convey the short-term vs. medium-term depeg risk.
Top Features (Feature Attribution): The dashboard highlights the top contributing factors to the current risk level. For example, it might list “Top drivers: pool price deviation, oracle price gap, volatility” with scores indicating their importance. This is derived from the forecaster explanation (permutation importance or SHAP values).
Alerts and Events Feed: A panel shows recent events (such as on-chain governance announcements or status updates fetched from external sources) and any alerts generated by the meta-controller. For instance, if a pool entered a “red” critical state, the dashboard might flash an alert with the time and recommended action.
Analyst Note: In critical situations, the dashboard can display the auto-generated analyst note – a sentence or two summarizing the anomaly and risk (e.g. “Fused anomaly now = 1.00 (max), 30-min risk = 86%. Confidence High. DAI/USDC pool is leading this move (corr=0.96). Suggested action: monitor closely; if risk > 0.6 for 5m, consider mitigation.”).
All these components update live as the pipeline runs, providing both developers and DeFi operators a clear window into the system’s decision-making. The dashboard can run in live mode (connected to the running pipeline) or demo mode (with sample data), which we describe in Dashboard section below.
On-Chain Integration (SentinelDataToken)
Sentinel can optionally interface with an Ethereum smart contract called SentinelDataToken. This contract is designed to record anomaly events on-chain and reward addresses (e.g. liquidity providers or reporters) with tokens for contributing to the monitoring effort:
The pipeline, upon detecting a high-severity event (e.g. a red alert), can call the contract’s submit_data_block function to log the event’s details on-chain (see Smart Contract Interface below for specifics). This call includes encoded data on the anomaly and risk (such as anomaly score in basis points, novelty, severity level, etc.).
The contract mints or distributes reward tokens (SentinelDataTokens) to a specified recipient address as an incentive. This provides a decentralized acknowledgment of the anomaly detection – useful for creating a trustless alerts network or rewarding community members who run the Sentinel system.
On-chain logs (events) are emitted so anyone can track when a depeg warning was posted, its severity, and the reward given.
This blockchain integration is optional: if enabled, it adds transparency and incentive alignment (the idea of a “watchtower” node that earns tokens for catching depeg events early). If not needed, the pipeline can be run purely off-chain with no web3 component.
To summarize, the architecture flows as: Data Collection -> Parallel Anomaly Detection -> Score Fusion -> Risk Forecasting -> Calibration -> Policy Decisions -> Outputs (Dashboard visuals, API responses, optional blockchain calls).
Installation
Setting up the Sentinel Depeg Forecasting System is straightforward. Ensure you have Python 3.10+ (the system is tested on Python 3.12) and pip available. Then follow these steps:
Clone the Repository (or download the source code):
git clone https://github.com/your-org/sentinel-depeg-forecast.git
cd sentinel-depeg-forecast
Create a Virtual Environment (optional but recommended):
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
Install Required Packages:
The repository provides a requirements.txt with all needed Python libraries. Install them with pip:
pip install -r requirements.txt
This will install core libraries like pandas, numpy, scikit-learn, xgboost, pytorch (torch), fastapi, uvicorn, streamlit, web3.py, requests, faiss-cpu, shap, weasyprint, seaborn, etc. These cover data handling, machine learning, web app, and blockchain interactions. (If you encounter issues installing some packages, ensure your pip is up-to-date. On Linux, you may need system packages for WeasyPrint’s HTML rendering or for faiss; refer to their docs if needed.)
Recommended Directory Layout:
After installation, your project structure should look roughly like:
sentinel-depeg-forecasting/
├── sentinel_runtime/        # Core pipeline code (Python modules)
├── dashboard/               # Streamlit dashboard code (e.g. app.py or similar)
├── models/                  # (Optional) Directory to store trained model files (will be created on first run)
├── outputs/                 # Directory for runtime output artifacts (can be configured, default is `outputs/`)
├── data/                    # (Optional) Static sample data or config files (if provided for demo)
├── requirements.txt
├── README.md
└── ... (other scripts and files)
By default, the system will use an output folder (e.g. outputs/) to save logs, the live dataset, model artifacts, etc. You can override this location by setting an environment variable (explained below). Ensure the process has write permissions to whatever directory is used for outputs.
Environment Configuration:
Before running the system, you may want to set certain environment variables:
ETH_RPC: (Required for live on-chain data) The Ethereum RPC endpoint URL to use for fetching on-chain data. For example, you can use a public node or Infura URL. If not set and MOCK_MODE is not enabled, the system will not be able to retrieve live pool data.
OUT: Path to the output directory. If not set, a default like ./outputs or the current directory will be used. It’s recommended to set OUT="./outputs" (the repository may already default to this).
LIVE_CSV: Path to the live dataset CSV file. By default this will be <OUT>/live_dataset.csv. In most cases you don’t need to set this manually; it will be created automatically. But if you want to preload it with historical data or use a custom path, you can set this.
HMAC_SECRET and API_KEY: Used for securing the API endpoints (FastAPI). In a dev environment, these default to "secret" and a dummy key; you can leave them or set your own for production. The HMAC secret is used to sign requests (for authentication on the REST API).
MOCK_MODE: Set MOCK_MODE=1 to enable simulation mode. In mock mode, the pipeline will generate synthetic data instead of calling real on-chain data. This is useful for testing or demo if you don’t have an RPC or if you want deterministic fake anomalies to see how the system responds.
You can put these in a .env file or export them in your shell before running. For example:
export ETH_RPC="https://mainnet.infura.io/v3/<your_project_id>"
export OUT="$(pwd)/outputs"
export HMAC_SECRET="mysecret"
export API_KEY="myapikey"
At this point, the software is installed and configured. Next, we will run the pipeline and confirm everything is working.
Running the Pipeline
Once installed, you can run the Sentinel pipeline to start monitoring data and generating predictions. There are a few modes of operation depending on your use case. Below we describe how to run inference and get results:
1. Data Ingestion and Input Structure
You do not need to manually provide data for live operation – the pipeline will fetch required data from on-chain and off-chain sources automatically:
On launch, the system will connect to each configured pool (addresses for pools are typically defined in the code or a config – e.g. USDC/USDT Uniswap v3 pool, DAI/USDC Curve pool, etc. By default, a few common pools are included, and you can modify the list in the configuration).
For each pool, the pipeline periodically samples metrics such as reserves, prices, and calculates derived features (dev, tvl_outflow_rate, etc.). It aims to append a new data row for each pool every few seconds (configurable, e.g. every 15 seconds).
Live Data Format: Internally, the data for all pools is stored in a single CSV (live_dataset.csv in the output directory). Each row contains at least:
ts: timestamp of the observation (UTC time).
pool: identifier of the pool (e.g. "USDC/USDT_univ3").
Basic pool metrics: e.g. price or reserves (depending on pool type), which are used to compute dev (deviation from peg).
Derived features: dev, dev_roll_std, tvl_outflow_rate, spot_twap_gap_bps, oracle_ratio, etc.
Detector scores: as the pipeline runs, columns for z_if, z_lof, ... anom_fused will be added for each timestamp.
Forecast outputs: once the forecaster runs, it may append risk_forecast_10m and risk_forecast_30m columns (and corresponding label columns y_10m, y_30m if computed for training).
If you want to run the pipeline on historical or custom data (for research/backtesting), you can populate live_dataset.csv with your dataset beforehand. Just ensure it has the required columns (ts, pool, and any feature columns expected by the models). The anomaly detectors will still run on it, and you can call forecast functions on that data.
In mock/simulation mode, the pipeline will synthesize a live_dataset.csv with dummy data (e.g. gradually increasing dev values to simulate a depeg scenario) so that detectors and forecasters have something to chew on. This is useful for demonstration or test runs.
2. Starting the Sentinel Service
The pipeline can be executed as a background service. There are two primary ways to run it:
Direct Python Execution (with internal loop): Simply run the main script which starts data collection and processing:
python -m sentinel_runtime.main
(Replace with the actual path to the main module if different; e.g. it might be python sentinel_runtime.py if the entry point is a file.) This will launch an infinite loop that:
Bootstraps initial data (fetches or generates enough data for detectors to have context).
Enters a cycle, every N seconds (e.g. 60s by default):
Samples new data for each pool (appends to live CSV).
Updates anomaly scores for the latest data points (each detector yields a z_* value, and anom_fused is updated).
Computes or updates risk forecasts for the latest window using the pre-trained models (or trains them on the fly if not already trained).
Periodically, performs maintenance: update network features, fetch off-chain events (like checking a DeFi protocol status page or governance feed for relevant news), and evaluate if retraining is needed.
Logs outputs and saves artifacts (more on artifacts below).
You will see console logs indicating progress (e.g. new data appended, anomaly scores, sleep intervals, etc.). On first run, if no models are present, it may train models – this is automatic if sufficient labeled data is available. Otherwise, it will operate with anomaly detectors alone until enough data accumulates to train the forecasters.
As a FastAPI Web Service: The pipeline is integrated with a FastAPI app that provides REST endpoints for inference results. You can launch it via Uvicorn:
uvicorn sentinel_runtime.app:app --host 0.0.0.0 --port 8000
(The actual import path for app may vary; check the repository. Often it might be something like sentinel_runtime:app if the FastAPI instance is created at the module level.) This will start the API server. Note: In this mode, you should ensure the data collection loop is also running. In some configurations, launching the FastAPI app will automatically start background threads for data collection (for example, via an @app.on_event("startup") handler in the code). If not, you may need to start the pipeline loop manually or run the direct execution in parallel. Check the documentation/comments in the code – typically, the system is designed such that running via FastAPI still triggers the internal loop. Once running, the service exposes several endpoints under /ml and /policy (see API Inference Endpoints below). These endpoints require HMAC header authentication by default (using the HMAC_SECRET and API_KEY you configured). For local testing, you can disable auth or use the provided example keys.
Verify Running: After starting, you should see the live_dataset.csv growing in the outputs directory, and files like forecast_10m.parquet being written after some time. If using FastAPI, you can open a browser at http://localhost:8000/docs to see the interactive API docs (OpenAPI schema), confirming the server is up.
3. Inference APIs and Usage
Once the pipeline is running, you can retrieve live results via the provided APIs or by reading output files. The primary inference endpoints include:
GET /ml/score_zoo – Parallel Anomaly Scores: Returns the latest anomaly scores from all detectors for the monitored pools. The response includes each pool and its current z_if, z_lof, z_ocsvm, z_cusum, z_ae, z_ae_seq, as well as the anom_fused. This gives a quick snapshot of which detector (if any) is signaling an anomaly. For example, a JSON response might look like:
{
  "scores": {
     "USDC/USDT_univ3": { "z_if": 0.2, "z_lof": 0.05, "z_ocsvm": 0.1, "z_cusum": 0.0, "z_ae": 0.1, "z_ae_seq": 0.15, "anom_fused": 0.2 },
     "DAI/USDC_univ3": { ... },
     ...
  },
  "ts": "2025-09-01T21:35:00Z"
}
(The exact format may differ slightly; consult the docs. You can filter by pool via a query parameter if desired.)
GET /ml/forecast – Risk Forecasts: Returns the latest risk probability predictions for each pool, for both 10-minute and 30-minute horizons. The output is typically an array of the most recent data points (e.g. last few minutes) with fields:
ts: timestamp
pool: pool name
anom_fused: fused anomaly score at that time
risk_forecast_10m: calibrated probability of depeg in next 10 min
risk_forecast_30m: calibrated probability of depeg in next 30 min
For example, a portion of the response might be:
{
  "items": [
    {
      "ts": "2025-09-01T21:35:00Z",
      "pool": "USDC/USDT_univ3",
      "anom_fused": 0.87,
      "risk_forecast_10m": 0.62,
      "risk_forecast_30m": 0.86
    },
    {
      "ts": "2025-09-01T21:35:00Z",
      "pool": "DAI/USDC_univ3",
      "anom_fused": 0.45,
      "risk_forecast_10m": 0.10,
      "risk_forecast_30m": 0.15
    }
  ]
}
This shows, for instance, that the USDC/USDT pool had a fused anomaly of 0.87 and the model believes a ~86% chance of a depeg event in the next 30 minutes (a very high risk). Meanwhile, the DAI/USDC pool looks normal with low risk. These probabilities are already calibrated via isotonic regression, making them interpretable as actual likelihoods.
GET /ml/explain – Forecast Explainability: Provides the top contributing features for the current risk forecast of each pool. The response might list features (by name) that had the largest influence on pushing the risk prediction up or down. For example:
{
  "top_contributors": ["dev", "oracle_ratio", "dev_roll_std"]
}
indicating that the deviation (dev), the oracle price ratio, and recent volatility were the main drivers of the model’s prediction at the moment. Under the hood, this is generated by permutation importance on the latest data (or SHAP values, if enabled). These help users trust and interpret why the model is predicting a certain risk level.
GET /ml/top_events – Recent High-Severity Events: Returns a list of recent events recorded by the system that had high severity (e.g. incidents where an anomaly triggered a red alert). This is essentially reading from an events.json log that the pipeline maintains. Each event entry might include:
ts – timestamp of the event
type – type of event (e.g. "anomaly_alert", "governance_vote", "status_update", etc.)
severity – severity level (e.g. 2 for warning, 3 for critical)
summary – a short description (for external events this might be a text like “Oracle price delayed”; for Sentinel-detected events it could be “Pool X deviated 1% from peg”)
source – source of the event info (could be a URL or identifier, e.g. a link to a governance forum post or an on-chain transaction ID).
This endpoint is useful for quickly querying “what notable things happened recently?” from an external system.
GET /ml/network – Cross-Pool Network Signals: Returns the latest computed network features that show inter-pool relationships. Each item might include:
pool – the reference pool
neighbor_max_dev – the maximum deviation observed in any correlated neighbor pool at the same time
neighbor_avg_anom – average anomaly score among neighbor pools
lead_lag_best – indicates if the pool is leading or lagging others in movements (positive means this pool leads, negative means lags, zero means synchronous or no clear lead/lag)
corr_best – correlation coefficient of this pool with its most correlated peer
With these, you can see if, say, one Curve pool is moving in tandem with another or if one is an early indicator of trouble in another pool. For example, it might show “DAI/USDC_univ3 leads peers (corr=0.96)” as we saw in an analyst note, meaning that pool’s movements are highly correlated and slightly ahead in time compared to others (which could suggest a systemic issue).
POST /policy/decide – Meta-Controller Decision: This endpoint runs the meta-controller logic to produce the current system-level decision and recommended action. Typically, you pass in the recent forecast probabilities or state, and it responds with:
An alert_level or status (e.g. "green", "yellow", "red").
A confidence rating ("High", "Medium", "Low").
Optionally, an action recommendation (text or code for what to do next).
It may also echo back the inputs or include which pool is causing the alert.
In practice, you might call this like:
curl -X POST /policy/decide \
     -H "X-API-Key: ... (and other auth)..." \
     -d '{"feeds_fresh": true, "recent_forecasts": {"USDC/USDT_univ3": 0.62, "DAI/USDC_univ3": 0.10}}'
and get a response:
{
  "level": "red",
  "confidence": "High",
  "action": "engage_mitigation"
}
(Example only – actual fields could differ, but that’s the idea.) This indicates the system thinks we’re in a red alert scenario with high confidence and suggests taking mitigation actions (which your external controller could implement, such as pausing a contract or notifying admins). The meta-controller uses internal rules combining anomaly and forecast info, as described earlier, to come up with this.
GET /policy/retrain_check – Retraining Signal: Returns a small JSON stating whether models should be retrained. For example:
{ 
  "should_retrain": true,
  "reason": "feature drift",
  "drift": { ... detailed drift metrics ... }
}
If should_retrain is true, the reason will be either "feature drift" (data distribution has changed beyond allowed threshold) or "scheduled" (i.e. it’s the scheduled retrain time). The drift object includes which features drifted and the metrics (see Retrain Strategy). This endpoint allows a scheduler or devOps script to poll and decide to initiate a retraining process (if not automatically done by the pipeline).
GET /policy/snapshot – Analyst Snapshot: Returns the latest Analyst Note and pointers to reports. The response might be something like:
{
  "note": {
    "ts": "2025-09-01T21:35:00Z",
    "text": "Fused anomaly now=1.00; 10-min risk=0.62; 30-min risk=0.86. Confidence High. Top drivers: dev_roll_std, oracle_ratio. Propagation: DAI/USDC_univ3 leads peers (corr=0.96). Action: monitor; if risk > 0.60 or 3 reds, engage mitigation."
  },
  "report": {
    "date": "2025-09-01",
    "pdf_path": "/reports/2025-09-01_report.pdf"
  }
}
The note.text is the human-readable summary of the most urgent current situation (it’s the same text shown in the dashboard alert panel). The report link (if applicable) points to the full nightly report for that date (you might need to host or open the PDF manually; the system saves it to the outputs folder).
Accessing these results:
Programmatically, you can call these endpoints with any HTTP client. Remember to include authentication headers (X-API-Key, X-Timestamp, X-Signature for HMAC) if you left security on. For initial testing, you could disable auth or use the provided default secret ("secret" as key with HMAC signature as shown in contract.md example).
Alternatively, if you’re not running the API, you can directly inspect the output files (described below). For example, open outputs/forecast_10m.parquet in Python to see recent forecasts, or read outputs/explain.json for top features.
4. Calibration & Fused Output Generation
The anomaly fusion and probability calibration generally happen automatically during pipeline operation:
Anomaly Fusion: As mentioned, every cycle the code updates anom_fused for new data (by combining detectors). You don’t need to run anything manually; any time you query the anomaly scores (via API or by reading the live CSV), the fused score is already there. If you were analyzing offline, you could recompute anom_fused as max(z_if, z_lof, ...) easily.
Forecast Calibration: Each forecaster model’s output is passed through its isotonic calibrator before being saved or returned. If you retrain a model, you should also retrain the calibrator:
The repository includes utilities to generate calibration curves. For example, after training, it will output calibration_10m.json and calibration_30m.json that contain the mapping of raw model probability to calibrated probability (in a series of bins).
If you ever retrain or adjust the model, run the calibration step on a validation set. In code, this is done by IsotonicRegression from scikit-learn. The README of the repo or training script likely shows how to produce these JSON files. Essentially, the calibrator takes the model’s predictions on historical data and fits a non-decreasing function to match predicted vs actual event frequency.
The pipeline will automatically load forecast_10m_calib.joblib (and 30m) if present and apply it. If they don’t exist or calibration hasn’t been done, you might get uncalibrated results or a warning.
Generating Fused Outputs / Alerts: The combination of anomaly detection + forecast forms the basis of system alerts (e.g. raising an event when both anom_fused and risk are high). To actually emit an alert (on-chain or via API), the meta-controller uses thresholds (see Calibration Notes on how these thresholds are chosen). The fused outputs are written to events.json for any alerts triggered:
For example, if a critical anomaly is detected in pool A, an entry might be added to events.json: {"type": "anomaly_alert", "severity": 3, "ts": "...", "summary": "Pool A critical anomaly detected (fused=0.95)", "source": "sentinel"}.
This fused output is also used for on-chain submission if enabled (the pipeline would call the smart contract with anomaly details).
Soft Alerts: The system can also produce lower-severity “warnings” (severity 2) if only the forecast or anomaly is moderately elevated. These might not appear on-chain but could be shown on the dashboard or in logs for operators to review.
In summary, as an end-user you typically don’t need to manually invoke calibration or fusion logic – it’s built-in. Just be aware that if you adjust thresholds or model outputs, recalibration is needed to maintain output quality.
5. Runtime Artifacts and Logs
During operation, Sentinel will create and update various artifact files in the OUT directory (default outputs/). These include:
live_dataset.csv: The main rolling dataset of all observations. It appends new rows as time progresses. You can open this in Pandas or a CSV viewer to see all raw features and appended results. (Be cautious if it grows very large over long periods; you may want to implement log rotation or pruning for extremely long runs.)
forecast_10m.parquet & forecast_30m.parquet: These Parquet files store the most recent forecast results (for quick loading by the dashboard or analysis). Typically, after each cycle, the latest N rows (e.g. 60 rows) of live data with their risk predictions are saved to these files. This allows the dashboard to load a small window of recent forecasts efficiently without parsing the entire CSV.
explain.json (and possibly explain_30m.json): A JSON file containing the latest feature importance explanations for the forecasters. The format might include a list of top features and their importance values or scores. The 10m and 30m explanations might be separate files. This is used by the dashboard to display “Top drivers”.
events.json: A log of events and alerts. Every time the system identifies an off-chain event (like a relevant news or governance update) or triggers an anomaly alert, it appends an entry here. This serves as an audit trail of what happened and when. It’s also used by the /ml/top_events API.
detector_pr_auc.json: A JSON report of the precision-recall performance of each detector (z_if, z_lof, etc.) computed during the last nightly evaluation. It contains metrics like the average precision (AP) or area under PR curve for each detector’s ability to predict the labeled incidents. The system uses this to decide a “winner detector” each day (the one with highest AP) for informational purposes. Developers can examine this file to understand which anomaly method is most effective on recent data.
feature_drift.json: This file contains the output of the latest drift computation. It lists each monitored feature with its KS statistic and PSI value comparing current distribution vs training baseline. It also has the boolean drift flag and the thresholds used. By checking this file, you can see which feature triggered drift (e.g. maybe oracle_ratio distribution changed significantly). The /policy/retrain_check essentially surfaces this information.
calibration_10m.json & calibration_30m.json: These store calibration curves data (bin counts, predicted vs observed probabilities) for the forecasters. They are produced during training and used mostly for analysis/documentation. The actual calibrator model objects are saved as forecast_10m_calib.joblib (binary) in the models/ directory.
Model files: Trained model artifacts like forecast_10m_xgb.joblib and forecast_30m_xgb.joblib will be saved (typically in a models/ or within outputs/). These are loaded at runtime for making predictions. If you retrain, these get overwritten with new versions.
RUN_META.json: A metadata file containing info about the current run/environment. It includes timestamps, version numbers of the software and libraries, configuration parameters (like which pools are being monitored, what RPC endpoint is used, etc.), and paths of key artifacts. It’s useful for debugging and record-keeping – essentially a snapshot of the system state when it started the current session.
Reports and Notes: The nightly report is saved as a Markdown and PDF in the outputs (e.g. report.pdf or dated like 2025-09-01_report.pdf). The analyst note for the latest alert is saved as analyst_note.pdf as well (and possibly analyst_note.md). These can be opened to see how the system is summarizing information for human analysts.
Logs: If you run directly via Python, logs are printed to stdout (you can redirect to a file). If using Uvicorn/FastAPI, you’ll have server logs as well. There isn’t a separate log file by default aside from the JSON artifacts listed above, but you can configure Python logging if desired.
All these artifacts are meant to help you interact with or audit the system’s operations. The Streamlit dashboard, described next, reads many of these files to present an interactive view.
Dashboard
The Sentinel Streamlit dashboard is a web application that allows users to visualize and interact with the forecasting system’s output in real time. It’s a crucial tool for both developers (to debug and improve models) and stakeholders (to monitor stablecoin pool health).
Launching the Dashboard
After the pipeline has been started (or even if it’s not running and you want to view historical data or demo), you can launch the Streamlit app. If the repository has a file like dashboard/app.py or app.py in the root, use:
streamlit run dashboard/app.py
(If the app file is named differently, adjust the path accordingly. Some repos might put the dashboard code in streamlit_app.py or similar.) This will start a local web server for the dashboard. By default, Streamlit will open a browser window (usually at http://localhost:8501). You should see the Sentinel dashboard UI load. Placing Artifacts: The dashboard needs access to the output artifacts generated by the pipeline:
Ensure that the OUT environment variable is the same for the dashboard process as it is for the pipeline. The dashboard will look for files like live_dataset.csv, forecast_10m.parquet, etc., in that OUT directory. If you set OUT="./outputs" and ran the pipeline, then launch Streamlit in the same environment (or you can set export OUT="./outputs" before streamlit run).
Alternatively, the dashboard might have a configuration section at the top of the script where you can specify paths. Check the code — often it will do something like data_path = os.getenv("OUT", "./outputs") to find where to read from.
If running the dashboard on a different machine or after the fact, copy the content of the outputs/ directory from the pipeline run onto the machine where the dashboard is running. For example, if you have an outputs folder with artifact files, you can place that in the same directory structure that the dashboard expects.
Once the dashboard is up, it will continuously refresh the data (Streamlit allows auto-refresh or can be set up with a refresh interval). You will see various sections as described earlier:
Charts for each pool’s metrics and anomaly scores,
Current risk levels,
Tables or lists of events,
etc.
Dashboard Features
The layout may vary, but typical interactions include:
Selecting Pools: A sidebar might list the monitored pools with checkboxes or a dropdown, allowing you to focus on one or multiple pools’ graphs at a time.
Time Window Controls: You might adjust how much history to display (e.g. last 1 hour, last 24 hours). The dashboard can simply read more rows from live_dataset.csv or similar to show longer history.
Manual Refresh or Pause: Options to pause auto-refresh if you want to inspect a certain time, or a refresh button to fetch the latest immediately.
Demo Mode Toggle: If available, a switch that says “Demo Mode” or “Use Example Data”. This would load a static dataset instead of expecting live updates. For example, on first launch, if no live_dataset.csv is found or MOCK_MODE is true, the app might load a bundled sample dataset and some saved predictions to illustrate how it looks. This is useful for trying out the interface without needing to actually connect to Ethereum or run the whole pipeline.
Demo Mode
In demo mode, the dashboard operates on example data:
The repository may include a file like demo/live_demo.csv and corresponding forecast_demo.parquet, events_demo.json, etc. These would be snapshots from a past run or synthetically generated scenarios.
When demo mode is active, the app loads these files instead of waiting for new data. It might simulate time progression by iterating through the data or just display it all.
You’ll see the same graphs and metrics, but they won’t update in real-time (unless the app is coded to simulate a clock).
This is great for onboarding new users – they can see what a depeg incident would look like (if the demo data includes one) out of the box.
To use demo mode, either set the MOCK_MODE=1 env before launching the app or use any UI control provided (some dashboards include a checkbox “Use demo data”). Consult the README or comments in the app script – often it will mention if you need to press a certain key or change a variable. By default, if the pipeline isn’t running and thus no fresh data is coming in, the dashboard may automatically fall back to a demo dataset.
Using the Dashboard
On the running dashboard, you will:
Observe real-time updates: as soon as an anomaly score spikes or risk goes above a threshold, you might see a color change or an alert box. For example, if a pool enters a warning state, its chart might highlight that region in yellow; if critical, in red.
Click on features or points: Some dashboards allow hovering over a data point to see the exact values of detector scores or feature values. If implemented, you might click a specific timestamp on the chart to reveal what each detector was outputting at that moment.
Read the Analyst Note: During a critical event, the note might be prominently displayed (or accessible via a button, e.g. “View Analysis”). This note gives context and recommended next steps.
Filter events: The events panel might let you filter to only show anomaly alerts vs governance updates, etc., or sort them by time or severity.
The dashboard is meant to be fairly self-explanatory and requires no coding to use, once it’s connected to the data.
Shutting Down
To close the dashboard, just stop the Streamlit process (Ctrl+C in the terminal where it’s running). This doesn’t affect the pipeline, which runs independently. You can keep the pipeline running continuously (24/7 monitoring) and only launch the dashboard when you want to visualize. In summary, the Streamlit dashboard is your front-end for Sentinel, providing intuitive insight into both model internals and system outputs. It’s highly recommended to run it, especially when testing new model tweaks or during periods of market volatility, as it can quickly highlight any early signs of a stablecoin depeg.
Smart Contract Interface (On-Chain Integration)
One of the unique features of Sentinel is the ability to report anomalies on-chain and reward those who detect them, via the SentinelDataToken smart contract. This section describes the smart contract’s interface and how to interact with it from the Sentinel system or any web3 client.
SentinelDataToken Overview
SentinelDataToken is an Ethereum token contract (ERC-20 compatible) with added functionality to accept data submissions from the Sentinel system. It acts as:
A token: It implements standard ERC-20 functions (transfer, balanceOf, etc.), so it can represent value or reputation for anomaly detection.
A data registry: It has a custom function submit_data_block that anyone (or specifically authorized addresses, depending on contract setup) can call to record an anomaly detection event. Calling this function mints or distributes a certain number of tokens as a reward to the caller (or a specified recipient).
An event emitter: Each submission triggers a DataBlockSubmitted event on-chain, which can be monitored by anyone to get real-time depeg warnings from Sentinel in a decentralized way.
The ABI (Application Binary Interface) of the contract defines the inputs and outputs of submit_data_block and other relevant details.
submit_data_block Method Details
The primary integration point is:
function submit_data_block(bytes32 block_hash, uint256 anomaly_bps, uint256 novelty_bps, uint256 severity, address recipient) public returns (uint256 reward);
Parameters:
block_hash (bytes32): A unique identifier or hash for the data block being submitted. In practice, the Sentinel system can generate a hash of the relevant data (such as a hash of the current live data window or a hash of a message describing the event). This ensures a unique reference on-chain and could be used to verify the data off-chain if needed. It might be, for example, keccak256(pool_id, timestamp, anomaly_bps, novelty_bps, severity). (The exact contents are up to the implementation; uniqueness is key to avoid duplicate submissions.)
anomaly_bps (uint256): The anomaly score expressed in basis points (bps). This is an integer typically from 0 to 10,000 representing the percentage of anomaly. Sentinel can take the anom_fused score (0.0 to 1.0) and multiply by 10,000 to get bps. For example, anom_fused = 0.87 becomes 8700 bps. Alternatively, one might use the actual deviation percentage of the pool if that’s more meaningful (e.g. if the pool is off by 0.5% from peg, that’s 50 bps anomaly).
novelty_bps (uint256): The novelty of the anomaly, also in basis points. Novelty here refers to how unusual or new this pattern is compared to historical data. If the system has a measure of novelty (for instance, derived from the Local Outlier Factor or autoencoder error relative to past anomalies), it can be scaled to bps. A value of 0 would mean the event is completely expected or common (not novel), and higher values (up to 10000) mean very novel (the system hasn’t seen something like this often). If the concept of novelty isn’t explicitly calculated by the pipeline, this could be set equal to the anomaly_bps or another proxy. Ideally, it differentiates between, say, a recurring known issue vs a brand new type of anomaly.
severity (uint256): A categorical severity indicator for the event. This is typically a small integer where a higher number means more severe:
We might use 1 = low severity (informational), 2 = warning, 3 = critical. In the Sentinel pipeline, if an event is just a caution (yellow alert), you might submit severity 2, whereas a full-blown depeg (red alert) is severity 3.
This field allows the contract or any readers of the event to quickly gauge the seriousness without parsing the other values.
The Sentinel system decides what severity to tag an event (based on thresholds on anomaly and risk as discussed).
recipient (address): The Ethereum address which should receive the reward tokens for this submission. This could be the address of the bot/operator running Sentinel, a governance or insurance fund, or any stakeholder you want to reward. By separating it from the caller (msg.sender), the system could authorize certain trusted bots to call the function but credit rewards to a community multisig or distribute to multiple addresses off-chain (though currently it’s one address in the function).
Return Value:
reward (uint256): The amount of SentinelDataTokens awarded for this submission (in the smallest unit, typically wei if the token has 18 decimals like ETH). The contract will calculate this based on its reward logic (see below) and mint or transfer that many tokens to the recipient. The function returns the same value for convenience (so the caller knows how much was given, without having to wait for event or recompute).
Event Emitted:
The contract likely emits an event when data is submitted. For example:
event DataBlockSubmitted(bytes32 block_hash, uint256 anomaly_bps, uint256 novelty_bps, uint256 severity, uint256 reward, address to);
When submit_data_block is called, an event with these fields will be logged:
The block_hash, anomaly_bps, novelty_bps, severity (echoing inputs for transparency),
reward (the number of tokens disbursed),
to (the recipient address).
This event can be tracked by front-ends or other contracts. For instance, a Dune Analytics or The Graph integration could listen for DataBlockSubmitted events to populate a public dashboard of stablecoin anomalies reported by Sentinel.
Reward Logic Suggestion
The smart contract’s code (not included here) defines how the reward is computed from the inputs. While you can implement any logic, a sensible approach is:
Base the reward primarily on the severity and anomaly magnitude. A critical, high anomaly event should yield more tokens than a minor anomaly.
Incorporate novelty to reward catching new or unexpected issues. If the system reports something that hasn’t been seen before (high novelty), perhaps bonus tokens are given.
Ensure the reward is within reasonable bounds to avoid draining the token supply on one event (unless the token is meant to be abundantly inflationary).
Example formula:
reward = (anomaly_bps * severity) + (novelty_bps * bonus_factor)
This could then be scaled or capped. For instance:
If anomaly_bps = 8700 and severity = 3 (a serious event), anomaly contribution = 26100.
novelty_bps maybe = 9000 (very novel), and bonus_factor could be 0.5, giving 4500.
Total = 30600 (in whatever base units; if token has 18 decimals, this might actually be 30600 * 10^18 units).
The contract might mint 30,600 tokens (if 1 token = 1 unit for simplicity in this example).
Alternatively, the logic might be multiplicative: e.g.
reward = anomaly_bps * severity * novelty_multiplier
where novelty_multiplier is >1 if novelty is high. Or a piecewise scheme:
If severity < 3 (not critical), reward a nominal amount (or even 0 if you only want to incentivize critical reporting).
If severity = 3, then reward = anomaly_bps (so up to 10000 tokens for a full 100% anomaly).
Additionally, if novelty_bps > 8000, add a +20% bonus, etc.
The specific scheme will depend on tokenomics decisions. The SentinelDataToken could have a fixed supply and distribute from a pool, or be inflationary minting new tokens per event. Important: The Sentinel pipeline does not enforce the reward logic; it only supplies the numbers. The contract itself computes and returns the reward. So any “suggested logic” here would actually be coded in the contract’s Solidity. Make sure to update both the contract and off-chain expectations in tandem if you tweak the formula.
Sample Web3 Write Call
To illustrate how the pipeline or a user would call submit_data_block, here’s an example using web3.py (Python):
from web3 import Web3
import json

# Connect to Ethereum (use your RPC, ensure it's the same network the contract is on)
w3 = Web3(Web3.HTTPProvider("https://ethereum-sepolia.rpc.example"))  # e.g. using Sepolia testnet

# Load the contract ABI (assuming SentinelDataToken.abi.json is provided in the repo)
with open("SentinelDataToken.abi.json") as f:
    abi = json.load(f)

contract_address = Web3.to_checksum_address("0xYourSentinelTokenAddress")  # replace with actual deployed address
token_contract = w3.eth.contract(address=contract_address, abi=abi)

# Prepare data for submission (this would come from the Sentinel system logic when an event triggers)
block_hash = Web3.keccak(text="USDC/USDT_univ3-2025-09-01T21:35:00Z")  # example: hash of pool id + timestamp or some unique string
anomaly_bps = 8700    # e.g. 0.87 anomaly
novelty_bps = 9000    # e.g. high novelty
severity = 3          # critical
recipient_addr = Web3.to_checksum_address("0xRecipientAddress")

# Build transaction
account = "0xYourSenderAddress"
private_key = "0x...YourPrivateKey..."
txn = token_contract.functions.submit_data_block(block_hash, anomaly_bps, novelty_bps, severity, recipient_addr).build_transaction({
    'from': account,
    'nonce': w3.eth.get_transaction_count(account),
    'gas': 200_000,           # estimate or adjust as needed
    'gasPrice': w3.to_wei('5', 'gwei')
})
# Sign and send
signed_txn = w3.eth.account.sign_transaction(txn, private_key=private_key)
tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
print(f"Submitted data block, tx hash = {tx_hash.hex()}")
A few notes on this snippet:
Replace the RPC URL and addresses with real ones. In a real deployment, you’d have the contract address (for testnet or mainnet) from the SentinelDataToken deployment.
We create a block_hash by hashing some data. Here we used Web3.keccak(text="...") just as an example. In practice, the pipeline code will likely generate this for you (it could hash a JSON blob of the event or just use a random UUID).
We then call submit_data_block via the contract object. We need to send a transaction because it’s a state-changing function (it will mint tokens).
We specify a gas limit (gas) high enough for the transaction (200k is arbitrary here; actual usage might be lower, but allow some buffer).
After sending, we get a transaction hash. Once mined, the event will appear on-chain and the recipient will have new tokens.
If you prefer a JavaScript example using ethers.js:
// Assuming ethers.js is set up and you have a signer (with provider and private key)
const contract = new ethers.Contract(tokenAddress, tokenAbi, signer);

const tx = await contract.submit_data_block(blockHash, anomalyBps, noveltyBps, severity, recipientAddr);
console.log("Tx sent, hash:", tx.hash);
const receipt = await tx.wait();
console.log("Tx mined, gas used:", receipt.gasUsed.toString());
This accomplishes the same thing using ethers in Node.js. Security & Usage Considerations:
The contract may restrict who can call submit_data_block (e.g. only authorized or only EOA accounts, etc.) to prevent spam or malicious usage. If so, ensure your sending address has permission.
Each data submission costs gas. When running Sentinel in production with on-chain reporting enabled, make sure the account used has ETH balance for gas fees. You could configure the pipeline to only post critical events to minimize cost.
Decide how to manage the private key. You might store it in an environment variable or use a wallet with RPC privileges. Always secure the key – if compromised, an attacker could spam the contract or steal rewards.
Integrating On-Chain Calls into Sentinel
If you enable on-chain integration, the typical flow is:
Sentinel detects a critical anomaly (red alert).
In the meta-controller logic, you add a hook: when level == "red" with high confidence, call the submit_data_block function via web3 (as illustrated). This could be done asynchronously so as not to block the main loop.
Include in the call data the appropriate values. Severity would correspond to the alert level. Anomaly and novelty bps can be derived from the current state (e.g., take the last anomaly fused score *10000, and perhaps compute novelty as difference from previous anomalies).
Use the pool ID and timestamp to create a unique block_hash.
Provide the recipient address for rewards (for instance, the address that operates the Sentinel node).
After submission, you might log the event (the contract will emit it anyway).
The reward returned (and event data) can be logged or even fed back into Sentinel (e.g., to tally how many tokens earned, or to reduce frequency of submissions if needed).
By having this on-chain record, later analysis can confirm that an alert was raised at a specific time (even if off-chain systems fail or are accused of missing something, the on-chain log is an immutable proof of reporting). In summary, SentinelDataToken adds a decentralized dimension to the monitoring system: anomalies are not just detected, but also immutably recorded and incentivized. This mechanism is particularly relevant in community-driven or trustless environments where you want many independent “sentinels” watching for depeg events and being rewarded without a central coordinator.
Calibration Notes
Designing and maintaining the labels and calibration for the forecasters is critical for accurate predictions. Here we outline how labeling was done and tips for dual-mode (warning/critical) forecasting and calibration.
Labeling Thresholds for Events: In order to train the 10m and 30m risk models, we need to label past data points as “event” or “no event”. Sentinel uses threshold-based labeling on two key signals:
dev > 0.005 (0.5%): If the pool’s price deviation exceeds 0.5% from the peg at any point in the future window, it’s a sign of a depeg event.
anom_fused > 0.90: If the fused anomaly score goes above 0.90 (very high confidence of anomaly) in the window.
Specifically, a data point at time t is labeled as y_10m = 1 (positive event) if in the next 10 minutes there exists any moment where both conditions hold: dev > 0.005 and anom_fused > 0.9. Similarly for y_30m over 30 minutes ahead. The rationale is to label only genuine, strong depeg incidents: the price moved significantly off peg and our detectors flagged it as anomalous. By requiring both, we avoid labeling cases where price moved a bit but it wasn’t actually a concerning anomaly (or was just noise), and avoid cases where detectors spiked but the price didn’t actually budge much. These threshold values (0.5% deviation, 0.90 fused score) were chosen based on domain knowledge and empirical analysis. You may adjust them for different stablecoins or sensitivity. For instance, if monitoring a very tightly pegged asset, you might lower the dev threshold to 0.2%. Or if you want more training examples (at risk of including some false positives), you could lower the fused score threshold so that smaller anomalies count as events.
Dual-Mode Forecasts (Warning vs Critical): The system inherently can handle two “modes” of alerts – e.g. warning (yellow) and critical (red). If you want the forecasters to predict both levels, you have a couple of options:
Train Separate Models – e.g. one model predicts the probability of a warning-level event (a milder criterion, like dev > 0.002 maybe) and another for critical events (dev > 0.01). This way you’d get two probabilities and could surface both. However, this doubles the model maintenance.
Train a Single Model with Soft Labels – incorporate severity into a single probabilistic model. For example, rather than a binary 0/1 label, assign:
0 for no event,
0.5 for a warning event,
1.0 for a critical event.
Then train a regressor or classifier on these values. The model will output a number between 0 and 1 which you can interpret as a severity-weighted risk. You might then decide thresholds on this output: e.g. >0.7 corresponds to likely critical, >0.3 corresponds to likely at least a warning. In practice, XGBoost can handle regression or you can transform this into a classification by rounding 0.5 to 1 for training (but better to keep as regression for soft outcome). Isotonic regression can still calibrate this output to a probability distribution of severity.
Multi-class Classification – Alternatively, treat it as 3 classes (no event, warning event, critical event) and train a multi-class model. This can directly predict, say, 70% no event, 20% warning, 10% critical. However, calibrating multi-class outputs is more complex (you’d calibrate each class probability separately with something like temperature scaling or isotonic per class).
In our implementation, we stuck with binary classification for simplicity (targeting critical-level events primarily). We then derive warnings as a lower threshold on the same probability output. For instance, if 30m risk > 0.3, we might call it a warning, and if > 0.6, a critical alert. This effectively bins the probability into modes. If you wish to explicitly label warnings, you could use dev > 0.003 & fused > 0.7 (just an example) to label some points as “warnings” in addition to “critical” labels for higher thresholds. These can augment your training set with intermediate examples (the soft label approach).
Generating Soft Labels: If you do incorporate dual criteria, generating “soft” labels means assigning fractional values to represent uncertainty or severity. For example:
If an event barely crossed the threshold (say dev = 0.0051 and fused = 0.91), you might label it 0.8 instead of 1, to indicate it’s a borderline event.
If an event blew way past the threshold (dev = 0.02, fused = 1.0 for an extended time), label could be 1.2 (clamped to 1 for actual training, but you get the idea of weighting it higher).
Practically, one approach is to use the actual dev value as part of the label. E.g., label = min(1, (max_dev_in_horizon / 0.005)). This would give 1.0 for anything >=0.5% dev, but for a milder bump of 0.003, it would yield 0.6. Combined with anomaly requirement, you could multiply by an anomaly factor similarly. However, be cautious: too much complexity in labeling can confuse the model. Most users find binary labels easiest to maintain. Soft labeling is an advanced technique to squeeze more information out of limited events, at the cost of interpretability.
Class Imbalance & Bin Balancing: Depeg events are (fortunately) rare. This means our training data is highly imbalanced (many more negatives than positives). Strategies to handle this:
Oversample positives or undersample negatives during model training. We often oversample the anomaly events (or weight them higher) so that the model pays attention to them. XGBoost allows setting a scale_pos_weight or you can just duplicate positive samples.
Define a reasonable horizon/window to avoid diluting labels. Using a 30-minute horizon means any event within that window makes the label 1, but if you choose too large a window (say 24h), almost every day might have some anomaly and you’d label too many positives, or you’d miss pinpointing when exactly the risk arises. Our choice of 10m and 30m balances immediate reaction vs short-term planning.
Calibration Bin Balancing: When calibrating via isotonic regression, ensure you have a good spread of predicted probabilities. If the model only ever outputs 0.0 or 0.99 (which can happen if it’s very confident), isotonic might overfit those few values. You can enforce a cap on model output probabilities (e.g. in training, use sigmoid with limit) or ensure you have enough validation points in each probability bin. In practice, we saw calibration curves with maybe 2-3 bins given the scarcity of events (see the nightly report example: a bin around 0.01 predicted vs 0.17 observed, etc.). It’s not a lot of data points, but it’s better than nothing.
If you find calibration is unstable (one bin says 1% vs 17% as in our example, which is a big difference), it means the model is underestimating risk in that range. You might want to adjust the model or threshold for labeling. Sometimes adding a bit more positive examples (like including near-misses as positives) can help the model not be overly confident that nothing will happen. Also consider updating calibration frequently. As more events happen (especially if market conditions change), yesterday’s calibration might not hold tomorrow. Isotonic regression is non-parametric and will adjust to new data if retrained.
Threshold Tuning: After calibration, you still might set specific thresholds for actions (like what probability triggers an alert). These thresholds can be tuned to your risk tolerance:
For example, we set in meta-controller: 10-min risk > 0.6 or anomaly now > 0.9 triggers a high alert. These numbers came from analysis and can be tuned. If you want fewer false alarms, raise them (e.g. require >0.8 probability). If you prefer caution and don’t mind occasional false alerts, lower them.
Use the Precision-Recall metrics to choose. If at 0.5 probability threshold you get 90% precision and 70% recall, that might be a good trade-off. We use Average Precision (AP) as a summary metric to evaluate detectors and presumably could for forecasters as well, focusing on capturing those rare events.
In short, calibrating a system like this is as much an art as a science. The provided default thresholds and methods come from experience with stablecoin pools; however, you should feel free to experiment:
If the system is too quiet (misses an event), consider lowering event thresholds or increasing model sensitivity.
If it’s too noisy (many false alarms), consider raising thresholds or adding a requirement like “anomaly must persist for X minutes to count as event” to filter out blips.
The Calibration and Labeling process is iterative. After deploying, regularly check how predictions vs reality align (the nightly report’s calibration section helps here). Adjust thresholds as needed and retrain/calibrate models to continuously improve accuracy.
Retrain Strategy
Model maintenance is critical in a non-stationary environment (markets evolve, new stablecoins behave differently, etc.). Sentinel implements both automatic drift detection and an easy process for retraining models. Here’s the strategy:
Feature Drift Detection (PSI & KS): The pipeline periodically (by default, once every data collection cycle or every few cycles) computes statistical drift metrics:
Kolmogorov–Smirnov (KS) test: For each numeric feature, it splits the data into two sets – the original training set distribution vs the recent live distribution – and computes the KS statistic (maximum difference in CDFs). A high KS value (close to 1) means the distributions differ significantly.
Population Stability Index (PSI): It also buckets the feature values and compares proportions between the training baseline and recent data to calculate PSI. Higher PSI indicates drift.
We use thresholds KS >= 0.20 or PSI >= 0.25 on any feature to signal drift. These are moderate thresholds; in practice:
KS 0.2 is a noticeable shift (not extreme, but enough to likely affect model predictions).
PSI 0.25 is on the higher end of what’s typically acceptable (PSI < 0.1 often considered no drift, 0.1-0.2 slight, >0.2 significant).
If either condition is met for any key feature (like dev, anom_fused, etc.), the system sets drift = true and notes which feature triggered it in feature_drift.json. This prompts a retraining recommendation.
Scheduled Retraining: Even if no obvious drift is detected, the system is configured to retrain models on a regular schedule as a precaution and to incorporate the latest data. By default, we schedule a retrain nightly (once every 24 hours). The /policy/retrain_check will return should_retrain: true with reason “scheduled” at the designated time (for instance, every midnight UTC or after N hours of runtime). This ensures the models don’t grow stale. Frequent retraining (daily) is feasible here because the models are lightweight (XGBoost on a few hundred data points) and new data arrives continuously. If your use case has expensive training or not much new data, you could do weekly retrains or retrain when drift is flagged only.
Performance-based Retraining: Another trigger mentioned is if model performance degrades, such as precision-recall AUC dropping below a threshold. We log daily AP for detectors; similarly, you could track forecast model Brier score or AUC on recent data (if you accumulate labels on the fly). If those metrics worsen significantly, that’s a sign the model might be losing predictive power, so retraining (or even re-engineering features) is warranted. In our current setup, performance-based triggers are not explicitly coded (beyond drift, which indirectly catches when features behave differently). But you can add conditions like: “if today’s AP < 70% of yesterday’s AP, set should_retrain.”
Retraining Process: When a retrain is needed, how to do it:
The pipeline can retrain automatically if configured. In our loop, every few hours it calls train_forecaster_10m() and train_forecaster_30m() on the accumulated data. These functions:
Prepare training matrices from the live dataset (splitting by time or using all with labels).
Fit a new XGBoost model for classification.
Evaluate metrics (AP, Brier) on a validation split and print them.
Fit an isotonic regression calibrator on the validation set outcomes (if enough positives exist; otherwise it might skip or reuse old calibrator).
Save the model and calibrator to disk (forecast_10m_xgb.joblib, forecast_10m_calib.joblib, etc.).
Optionally, update any meta info (like feature list, or model version).
Also, output updated calibration JSON and detector PR AUC metrics as part of the nightly report.
Manual Retrain: If you prefer manual control (for example, you want to retrain in a Jupyter notebook to inspect results):
Stop or pause the pipeline (to avoid new data coming in during retrain, or run retrain in a separate environment on a copy of the data).
Load live_dataset.csv (or a curated subset if it’s huge).
Ensure the labels y_10m and y_30m are present for the data. (You can generate them by running the label function in sentinel_runtime: there may be a helper like ensure_labels_fixed_on_live() that fills in y_10m and y_30m using the current threshold logic.)
Use the provided training functions or write your own: e.g. train an XGBoost with sklearn.XGBClassifier or xgboost.train. The repository likely includes a training.py or you can call sentinel_runtime.train_forecaster_10m(feature_cols=..., label_col="y_10m") directly.
Once the model is trained on historical data, evaluate it. Check confusion matrix, PR curve, etc., to ensure it’s learning effectively.
Train the calibrator: scikit-learn’s IsotonicRegression can be fit on the model’s validation predictions vs actual labels. Save the calibrator (which could simply be pickled or joblib).
Replace the old model files with the new ones in models/ directory.
Resume the pipeline. It will now use the updated model for future predictions.
Always retrain both 10m and 30m models if using both. They can be trained on the same dataset just with different label horizons. If the drift was in input features, you might also consider updating the feature set (maybe add a new feature that accounts for a new behavior, etc.) – this is more involved and may require code changes in feature engineering parts.
Drift Reset: After retraining, the baseline for drift detection should ideally be updated. For instance, you might reset the “training set” distribution to the more recent data used to train, so that drift is measured relative to that. In practice, one can simply update the base dataset slice used in compute_feature_drift() to the last X days instead of always the original training set. Our implementation splits the live data 70/30 for drift calc; when you retrain, that live data inherently includes the new distribution, so drift should subside until something changes again.
Versioning: The RUN_META.json or logs will note model version changes. It’s good to version your models (even if just by date or a run counter) to keep track of when a new model went live. If something goes wrong after a retrain, you can roll back to a previous model (keep backups of the last known-good model files).
Continuous Learning vs Static Model: Our approach is a form of continuous learning (updating daily). This keeps the models adaptive but beware of overfitting to recent noise or forgetting older patterns. Always maintain some hold-out evaluation: e.g. evaluate the new model on data from a week ago (if still relevant) to ensure it hasn’t skewed too far to just the last day’s peculiarities. If you see oscillation (one day it overfits, next day corrects, etc.), you might want to retrain less often or use a rolling window of data for training rather than all data.
In conclusion, the retrain strategy is:
Monitor -> Trigger -> Retrain -> Deploy -> Repeat.
The system provides the monitoring and trigger, and basic retrain functions, but you have full control to intervene, confirm, and adjust models during this process. With this strategy, Sentinel can continue to provide accurate forecasts even as market conditions evolve or new types of anomalies emerge.
Contributing
Contributions to the Sentinel Depeg Forecasting System are welcome! Whether you are a developer wanting to improve the code or a researcher with ideas for better detection, we appreciate community involvement. Here are some guidelines and areas where you can contribute:
Bug Reports & Issues: If you encounter any bugs, inconsistencies, or have trouble running the system, please open an issue on the GitHub repository. Include details like error messages, conditions that triggered the bug, and steps to reproduce. We aim to make the system robust, and your reports help us get there.
Feature Improvements: Have an idea to make Sentinel better? Some ideas include:
New Detectors: Integrate additional anomaly detection algorithms (e.g., Prophet for seasonality, GAN-based detectors, etc.). If you add one, ensure it outputs a z_<name> score and integrate it into the fusion logic and evaluation pipeline.
Alternative Models: Improve the forecasters by trying other algorithms (we listed LightGBM in requirements, you could experiment with that or even neural networks if appropriate). Just make sure any new dependencies are noted and the model can still run in real-time.
Feature Engineering: Perhaps you have domain knowledge that suggests a new feature (e.g., incorporating exchange order book data or Twitter sentiment for the stablecoin). You can add features to the data ingestion and use them in the models. Just ensure you update the training and drift logic to handle the new feature.
Dashboard Enhancements: The Streamlit dashboard can always be improved – whether it’s UI/UX (making it prettier, adding tooltips, etc.) or new visualizations (maybe a map of correlated pools, or a gauge for overall system health).
On-chain Integration: If you want to extend how Sentinel interacts with smart contracts (like maybe also pulling on-chain oracle prices directly, or integrating with a DAO for automated actions), feel free to contribute in that area. Just maintain security best practices if dealing with keys or contracts.
Contributing Code:
Fork the repository, create a new branch for your feature/bugfix, and commit your changes with clear messages.
Ensure that existing tests pass (run pytest if tests are included, or any provided test script).
Add new tests for any new feature you introduce or bug you fix. For example, if you add a detector, include a test case simulating data where that detector should flag an anomaly.
Follow the code style conventions of the project (PEP8 for Python, etc.). It’s often useful to run a linter or code formatter.
Once your changes are ready, open a Pull Request with a description of what it does. We’ll review it and discuss any feedback or required changes.
Retrain Configs & Model Updates: The community can contribute improved model parameters or configurations as well. If you found that adjusting the XGBoost hyperparameters yields better performance (maybe using a different max_depth or adding regularization), you can share that:
For model configs, either create a config file (if the project uses one) or mention in the PR so we can incorporate it.
If you want to contribute a completely new model (say a pre-trained neural network), discuss first in an issue – large models might be hard to include directly, but perhaps we can include an option to download them or an implementation stub that users can fill.
You can also contribute calibration improvements – e.g. code for Platt scaling, or a reliability diagram plot in the report to visualize calibration.
Governance Rules & Policy: The meta-controller’s rules (when to alert, how to escalate, etc.) can be adjusted based on community consensus. If you have suggestions (for example, “maybe require anomaly to last 3 consecutive points before red alert to reduce noise”), open an issue or PR describing the change. Because these rules affect how the system will react in production, they might be debated – that’s fine, we encourage healthy discussion on thresholds and policies in the issues forum. Ultimately, if the project maintainers agree, we’ll merge changes that make the system safer and more reliable.
Documentation: We welcome improvements to documentation as well. If you find something unclear in the README or want to add a tutorial (e.g., “How to add a new pool to monitor” or “Interpreting the nightly report”), you can contribute to the Wiki or docs folder. Clear, comprehensive docs help everyone.
Community Support: Even if you don’t write code, you can contribute by helping others. Answer questions that appear in issues (maybe you faced the same issue and solved it), share your usage experiences, or suggest new ideas.
Development Setup: For contributing code, it’s useful to install the package in editable mode and run in a dev environment:
pip install -e .[dev]
(This would install the package and any dev dependencies like pytest.) Then you can run pytest to execute tests. If you add dependencies, update requirements.txt and optionally a requirements-dev.txt for dev-only packages. When submitting a PR, ensure you sign the Contributor License Agreement (CLA) if prompted (for MIT, usually not an issue, but some orgs require DCO/CLA for contributions). We’re excited to build a strong community around open-source DeFi monitoring. By contributing, you’ll be helping make the stablecoin ecosystem safer and more transparent. Thank you in advance for your involvement!
License & Credits
This project is open source under the MIT License. That means you’re free to use, modify, and distribute the code, provided you include the license notice in any copies or substantial portions of the software. Refer to the LICENSE file in the repository for the full text of the license. In short, it provides the software “as is” without warranty, and the authors aren’t liable for any damages or issues arising from its use. Credits & Acknowledgments:
The Sentinel Depeg Forecasting System was initially developed by the contributors of the [Your Org or Team Name] in 2025, inspired by the need for better real-time risk management in DeFi stablecoin platforms.
We acknowledge the open-source libraries that made this project possible: pandas, numpy, scikit-learn, XGBoost, PyTorch, FastAPI, Streamlit, Web3.py, WeasyPrint, FAISS, and many others. Their developers and communities deserve huge thanks.
Special thanks to anyone who has provided feedback, testing, or early contributions to this project. Your input has shaped the system’s features and stability.
If you use Sentinel in your research or project, we kindly ask you to give credit in your documentation or academic papers. This not only recognizes the work but also helps others find this tool.
Financial Disclaimer: This software is for informational purposes and does not constitute financial advice. While it aims to predict and warn of potential depeg events, it cannot guarantee accuracy. Users should not rely solely on this system for decision-making and should manage financial risks accordingly.
We hope Sentinel proves useful to developers, risk analysts, and DeFi participants in safeguarding stablecoin ecosystems. Happy monitoring! If you have any questions or need help, feel free to open an issue or join our community chat (if available). Together, let’s keep an eye on those pegs!
