DeFi Depeg Sentinel with Tokenized Reward System
1. Project Overview
DeFi Depeg Sentinel is a real-time anomaly detection and risk forecasting system for decentralized finance (DeFi) stablecoin pools. Its goal is to monitor stablecoin liquidity pools (such as Curve or Uniswap pools) and detect early signs of a depeg event – when a stablecoin strays from its intended price (usually $1.00) – or other anomalous behavior. The system provides short-term risk forecasts (over 10-minute and 30-minute horizons) to predict potential depeg incidents before they fully materialize. Key features of the Depeg Sentinel include:
Real-Time Analysis: Continuously ingests live stablecoin pool data (prices, liquidity, outflows, etc.) and computes anomaly scores in real time to flag irregular behavior in the pool.
Ensemble Anomaly Detectors: Utilizes multiple anomaly detection models (statistical and machine learning based) to identify outliers or shifts in the pool’s metrics. These detectors are combined into a fused anomaly score for robust detection.
Risk Forecasting: Trained predictive models estimate the probability of a depeg or severe anomaly occurring in the near future (looking 10 minutes and 30 minutes ahead). This helps differentiate between a momentary glitch and a developing crisis.
Calibrated Alerts: The raw forecasts are calibrated to ensure that the reported risk probabilities correspond to real-world frequencies. The system generates a severity level (on a scale of 1 to 5) for the current situation, where higher severity indicates a more critical threat (e.g., an imminent depeg).
Streamlit Dashboard: A user-friendly dashboard (dashboard.py) visualizes the live data, anomaly scores, forecasted risks, and any triggered alerts. This dashboard allows DeFi researchers or operators to monitor stablecoin pools and understand the model outputs at a glance.
Tokenized Reward Mechanism: Optionally, the Sentinel can emit on-chain rewards when significant anomalies are detected. A Vyper smart contract called SentinelDataToken can be called to record the event on the Ethereum blockchain and mint reward tokens to a specified recipient based on the severity of the anomaly. This creates a decentralized, tamper-evident log of depeg risk events and can incentivize those who run or respond to the sentinel.
In summary, this project serves as an early warning system for stablecoin depegging and abnormal pool behavior, blending off-chain analytics with on-chain actions. It is useful for protocol developers, risk analysts, or automated systems looking to hedge or mitigate risks in real time.
2. Architecture
The system is composed of several components and steps that together form the end-to-end pipeline from data input to on-chain output. Below is an overview of the architecture and each major component:
Data Input Pipeline: The pipeline begins with live data ingestion. For demonstration and testing, the repository provides a file live_dataset.csv which simulates a stream of time-series data from a stablecoin pool. In a real deployment, this would be replaced by an API or on-chain data fetch (e.g., pulling pool reserve balances, prices, and exchange rates at intervals). The data includes features such as:
Price deviation (dev): how far the stablecoin’s price is from $1 (could be absolute or percentage).
Rolling volatility (dev_roll_std): short-term volatility of the price deviation.
Liquidity outflow rate (tvl_outflow_rate): the rate at which liquidity (total value locked) is leaving the pool.
Price gaps (spot_twap_gap_bps): the gap between spot price and time-weighted average price (TWAP) in basis points.
Oracle price ratio (oracle_ratio): ratio of on-chain oracle price to pool price (to detect oracle divergence).
Others: any additional relevant metrics.
These features are updated continuously. The pipeline reads the latest sample (or batch of samples) and feeds them to the anomaly detectors.
Anomaly Detectors (Score Zoo): The project implements a variety of anomaly detection algorithms in the module score_zoo.py (the “zoo” of detectors). Each detector examines the incoming feature data and produces an anomaly score (typically, higher = more likely anomaly) for the current state. The detectors include:
z_if (Isolation Forest): An isolation forest model that isolates outliers in the feature space. It’s an ensemble of decision trees that assigns an anomaly score based on how early a data point is isolated in the trees. This is good for general anomaly detection on numerical data.
z_lof (Local Outlier Factor): A model that computes the local density deviation of a given data point with respect to its neighbors. If a point’s density is significantly lower than its neighbors, it’s an outlier.
z_ocsvm (One-Class SVM): A one-class support vector machine trained on “normal” data to detect novel patterns. It tries to learn the boundary of normal data in feature space; points outside that boundary are flagged as novelties.
z_ae (Autoencoder): A neural network autoencoder that compresses and reconstructs data. If the reconstruction error is high for a new data point, it suggests the point is unlike the normal training data (an anomaly).
z_cusum (CUSUM Detector): A Cumulative Sum algorithm that detects shifts in the mean of a series. It’s effective for detecting abrupt changes or drift in a time-series (for example, a sudden jump in price deviation).
z_ae_seq (Sequential Autoencoder): An extension of the autoencoder that considers sequences (e.g., an LSTM-based autoencoder). It looks at a window of time and tries to reconstruct it; large reconstruction error indicates an anomalous sequence of events (even if single data points seem normal in isolation).
Each detector outputs a score (often normalized between 0 and 1, or as a z-score). For example, z_if might output 0.95 if the Isolation Forest strongly believes the point is an outlier, whereas z_cusum might output 0.10 if no change is detected.
Score Fusion: Rather than relying on a single model, the Sentinel fuses these anomaly scores into one unified fused anomaly score (anom_fused). This can be done either by a simple average or a weighted combination of the detector outputs. Weighted fusion allows giving more importance to detectors that historically perform better. In our configuration, weights could be derived from each detector’s precision/recall performance (for instance, Isolation Forest (z_if) was found to be the top performer in our tests, so it might get a higher weight). The fusion step helps reduce false positives and false negatives by leveraging the strengths of each method. The fused score is scaled between 0 and 1 (with 1 indicating maximum anomaly as per the ensemble).
Forecasting Models (10m & 30m): While the anomaly detectors focus on the current state, we also want to predict the future risk. The project includes two pre-trained forecasting models:
10-minute Forecaster: An XGBoost model saved as forecast_10m_xgb.joblib that predicts the likelihood of a critical anomaly or depeg event in the next 10 minutes.
30-minute Forecaster: Another XGBoost model (forecast_30m_xgb.joblib) predicting the risk over a 30-minute horizon.
These models are trained on historical episodes of pool behavior. They take as input features derived from the current and recent data (e.g., the latest fused anomaly score, the trend of the deviation, recent volume changes, etc.). The models output a probability (0.0 to 1.0) that an incident (depeg or severe anomaly) will occur within that horizon. For example, a 30m risk of 0.86 means an 86% chance of a depeg-level event in the next 30 minutes given current conditions.
Forecast Calibration: Raw model outputs from machine learning can be overconfident or underconfident. We apply isotonic regression calibration to adjust the forecast probabilities. Calibration aligns the predicted probabilities with the observed frequency of events in the validation data. For instance, if out of all instances where the model predicted ~0.8 probability, only 60% actually experienced an incident, the calibrator will downscale 0.8 predictions closer to 0.6. Each horizon has its own calibrator:
forecast_10m_calib.joblib – isotonic regression model for 10m forecasts.
forecast_30m_calib.joblib – isotonic regression for 30m forecasts.
The calibration artifacts (also summarized in JSON like calibration_10m.json and calibration_30m.json) contain the mapping of binned prediction scores to true outcome rates. For example, a portion of a calibration report might say:
10m forecast calibration:
- bin ≈ 0.56: observed frequency 1.00 (n=16)
- bin ≈ 0.70: observed frequency 1.00 (n=20)
- bin ≈ 0.87: observed frequency 1.00 (n=24)
(This indicates in one calibration run, all predictions in those bins corresponded to 100% incident rate, which might imply the model only saw positive cases – a sign to gather more negative samples for better calibration.) After calibration, the forecast outputs are written to files (e.g., forecast_10m.parquet and forecast_30m.parquet) for use by the dashboard and policy logic.
Severity Scoring (NLP & Meta-Model): In parallel to numeric analysis, the system assesses the severity of the situation on a scale from 1 to 5 (1 = low severity, 5 = highest severity). The severity score is determined by a separate model that can incorporate both the on-chain metrics and any off-chain context:
The project includes a basic NLP model that looks at text-based events or news (for example, monitoring a stablecoin issuer’s status page or governance forum for warnings). This model (a logistic regression over sentence embeddings) classifies text events into severity levels. For instance, an update like “Curve pool imbalance rising rapidly” might be severity 2, while “Major hack draining liquidity” could be severity 5. The model is initially trained on sample events and can be fine-tuned or calibrated with real incident reports (the file events.json logs fetched events and their severities).
Separately, the current on-chain anomaly indicators can influence severity. If the fused anomaly score is extreme or the stablecoin price deviation is very large, the system will assign a higher baseline severity even if external news is quiet.
These inputs are combined into an overall severity rating. A simple approach is to take the max of the on-chain severity suggestion and off-chain event severity. This severity model is also calibrated so that severity correlates well with actual outcomes. For instance, an internal calibrator (severity_calibrator_meta.json) might ensure that severity 5 is reserved for truly extreme situations (like known depeg events).
Policy Engine (Meta-Controller): The meta-controller is the brains that decides how to act on the model outputs. It implements the business logic and rules such as:
Drift Detection: It monitors if the input data distribution has drifted significantly from what the models were trained on. This is done by comparing recent data to the training baseline using statistics like the Kolmogorov–Smirnov (KS) test and Population Stability Index (PSI). If drift is detected (flag in feature_drift.json is set to true), the policy might raise a caution flag and suggest model retraining. For example, if a feature like oracle_ratio suddenly has a PSI of 4.9 which exceeds the threshold of 0.25, it indicates the relationship between pool price and oracle price is in a regime not seen in training data.
Alert Thresholds: The policy defines what constitutes a warning versus a critical alert. For instance, it might classify any fused anomaly score above 0.9 or price deviation above 0.5% as at least a warning. If in addition the 30m forecast risk is above, say, 60%, it might escalate to a critical alert. Multiple consecutive high anomalies (“3 consecutive reds”) is also a condition for escalation.
Actions and Escalation: Based on the above, the policy decides whether to simply log/monitor, to display a warning on the dashboard, or to trigger on-chain reporting. It can also formulate mitigation actions (for example, an analyst note might suggest “widen slippage tolerance or reroute trades” when a critical risk is identified). The policy logic ensures that the smart contract is only called for significant events, to avoid spam or gas waste.
Logging and Metadata: The policy compiles a summary of each run (saved in RUN_META.json) and any event triggered (saved in events.json). This includes timestamps, what conditions were met, and what actions were taken. This record-keeping helps with debugging and for a historical log of anomalies and responses.
Streamlit Dashboard (dashboard.py): The dashboard ties it all together for the end user. It is built with Streamlit and serves as both a monitor and controller interface:
Visualization: It displays charts for the stablecoin pool metrics (price vs peg over time, liquidity changes, etc.), the anomaly scores from each detector, and the fused anomaly score in real-time. It will highlight when the fused score crosses the warning threshold.
Forecast Display: The dashboard shows the 10m and 30m risk forecasts (after calibration) perhaps as gauges or trend lines. It also might display a calibrated risk level (e.g., Low/Medium/High) and the numeric probability.
Severity and Alerts: If the system has issued a warning or critical alert, the dashboard will prominently show it (e.g., a banner or colored indicator). It may list active incidents or the latest triggered event from events.json, including the severity level and a short description (like “Unusual outflows detected”).
Explainability: The app can present interpretability info from explain.json – for example, a list of the top contributing features to the 10m risk prediction. This might be shown as a breakdown like: “Top drivers of risk: 1) Price deviation (dev) +0.37, 2) Fused anomaly +0.01, 3) Volatility (dev_roll_std) +0.00” (these values could be permutation importance or SHAP values indicating which features are pushing the risk prediction up).
Interactivity: On the dashboard, one can likely select different pools (if multiple are configured), adjust certain parameters for analysis (like toggle a threshold), or manually refresh the data.
The dashboard is designed to run both locally and in the cloud. It reads the output artifact files (mentioned above) to get the latest scores, forecasts, and events. If those artifact files are not found, the dashboard will attempt to either run in a demo mode (using embedded sample data) or prompt the user to run the pipeline to generate them. This ensures a first-time user can still see the interface without having to fully execute the backend.
Smart Contract Interface (On-Chain Integration): A unique aspect of this project is connecting the off-chain analytics to an on-chain contract for immutable logging and rewarding of detected events. The contract is written in Vyper (source in SentinelDataToken.vy) and provides a tokenized reward system:
SentinelDataToken: This contract behaves like an ERC-20 token with additional functionality. It has a fixed supply mechanics where new tokens can only be minted by the authorized minter (intended to be the Depeg Sentinel system itself or its owner). Key features of the contract include:
Data Submission (submit_data_block): This external function accepts a data payload of an anomaly event and mints reward tokens accordingly. The function signature is:
submit_data_block(bytes32 block_hash, uint256 anomaly_bps, uint256 novelty_bps, uint256 severity, address recipient) -> uint256
When the Sentinel calls this, it should provide:
block_hash: a unique identifier for the data block or event. This could be the hash of the latest block in the blockchain at the time of detection (to timestamp it), or a hash of the anomaly data. The contract ensures the same block_hash can’t be submitted twice (preventing duplicate rewards).
anomaly_bps: the anomaly severity in basis points. This is a numerical representation (0 to 10,000) of how severe the anomaly was in terms of deviation. For example, if the stablecoin dipped 2% from the peg, anomaly_bps might be 200 (i.e., 200 bps = 2.00%).
novelty_bps: the novelty or unexpectedness of the event in basis points. This could represent distribution shift or how out-of-training-distribution the event was. If unsure, you can think of novelty_bps as an additional anomaly indicator – for instance, if a completely new type of anomaly occurred, you might set this high. In many cases, novelty_bps may be calculated similarly to anomaly but focusing on features like oracle_ratio or others. (If there’s no separate notion of novelty, this could simply be set equal to anomaly or reserved for future use.)
severity: an integer from 1 to 5 indicating the severity tier of the incident (the same severity determined by the pipeline). The contract expects 1 ≤ severity ≤ 5.
recipient: the Ethereum address that should receive the reward tokens for this event. This could be an address that is being rewarded for running the sentinel or an address associated with the affected protocol for record-keeping. (It’s up to your use case – you could even set it to the pool’s governing address or an insurance fund.)
Reward Calculation: The contract computes the reward tokens to mint based on the inputs. The formula (visible in the Vyper source) is:
reward = 1e18 * (anomaly_bps + novelty_bps) * severity / (10000 * 5)
Here, 1e18 represents one token unit (since the token has 18 decimals like ETH). The denominator 10000 * 5 effectively normalizes the maximum reward to 1 token per event if anomaly_bps + novelty_bps is 10000 (i.e., 100%) and severity is 5 (max). In simpler terms:
A minor anomaly (say anomaly_bps 50 = 0.5%) with severity 2 will yield a tiny fraction of a token.
A full depeg (anomaly_bps ~10000) with severity 5 would yield ~1 token.
The sum of anomaly and novelty means if both are significant, it adds up. For example, anomaly 3000 (30% deviation) + novelty 3000, severity 5 => reward = 1e18 * 6000 * 5 / 50000 = 0.6 token.
The contract also splits the reward between a treasury and the recipient: a portion of the reward (configured by treasury_bps) will go to the treasury address, and the rest to the recipient. By default, if treasury_bps is set to, say, 2000 (20%), then 20% of the tokens are minted to the treasury and 80% to the recipient. This is useful if, for example, the protocol wants to retain some tokens (for governance or burning) every time a reward is given out.
Security and Roles: Only the address with the minter role can call submit_data_block. The contract’s owner (deployer) sets the minter address in the constructor (and can change it via set_minter). In practice, you would set the Sentinel’s Ethereum address (or the address that will send the transactions on behalf of the Sentinel, e.g., a bot’s key) as the minter. Any other address calling submit_data_block will be rejected. The contract is also pausable by the owner in case of emergencies.
Event Logging: When submit_data_block is called successfully, the contract emits a DataBlockSubmitted event. This event contains the block_hash, anomaly_bps, novelty_bps, severity, reward, and to (recipient) fields. This on-chain event log serves as an immutable record that at a certain time an anomaly of given severity was detected and a reward of X tokens was minted to a certain address. Community members or automated agents could monitor these events (for example, via a Dune Analytics dashboard or a script) to get a live feed of depeg warnings coming from the Sentinel.
In essence, the architecture spans from data collection and processing (off-chain) to decision-making and optional blockchain interaction. The off-chain components are where the heavy computation happens (machine learning models), and the on-chain component ensures transparency and aligns incentives (rewarding addresses that run the sentinel or contribute to catching incidents). Below is a simplified flow of the pipeline for clarity:
Data Sampling → (live data feed or live_dataset.csv provides latest pool metrics)
Anomaly Scoring → (compute z_if, z_lof, z_ocsvm, z_cusum, z_ae, z_ae_seq scores)
Fusion → (aggregate scores into one anom_fused score)
Forecasting → (predict 10m and 30m risk using XGBoost models)
Calibration → (adjust forecast probabilities using isotonic models)
Severity & Policy → (assign severity 1–5; check drift and rules; decide actions)
Outputs:
Save outputs to files (forecast_*.parquet, explain.json, etc.)
Update dashboard visuals/alerts
If critical, call smart contract submit_data_block (mint token and log event)
Each of these pieces is further detailed in the repository’s code and can be individually tested or modified.
3. Installation
Follow these steps to set up the project environment and install all dependencies:
Clone the Repository:
git clone https://github.com/yourusername/defi-depeg-sentinel.git
cd defi-depeg-sentinel
(Replace the URL with the actual repository URL.)
Create a Virtual Environment: It’s recommended to use a Python virtual environment to avoid package conflicts. For example, using Python’s built-in venv:
python3 -m venv venv
source venv/bin/activate   # On Windows, use "venv\\Scripts\\activate"
Install Dependencies: The required Python packages are listed in requirements.txt. Install them using pip:
pip install -r requirements.txt
This will install libraries such as:
pandas, numpy for data manipulation
scikit-learn, xgboost for the ML models and detectors
pyod or other anomaly detection libs (if used by score_zoo.py)
streamlit for the dashboard
web3 for blockchain interactions
vyper (optional, only if you plan to recompile the contract)
and others listed in the file.
(Optional) Configure Environment Variables:
If you intend to use the on-chain reward feature, set up the following environment variables:
ETH_RPC: The HTTP RPC URL of an Ethereum node. For example, you can use a public RPC or Infura/Alchemy URL. If this is not set, the system will default to a mock mode (no real chain calls) using a public endpoint if available.
PRIVATE_KEY: The private key of the Ethereum account that will send transactions (this account should have the minter role on the deployed contract and have some ETH for gas). Do not expose your private key in plain text. In a local environment, you might set it in an .env file or export in shell; on Streamlit Cloud, use Secrets management.
(Alternatively, the code may read from a Web3 wallet or prompt – but typically PRIVATE_KEY and ETH_RPC are standard).
You can also set TREASURY_ADDR if you plan to deploy your own contract and want to specify the treasury address at deployment.
If you are not using the contract integration, you can skip these. The system will run in off-chain mode by default, which does not require an RPC or key.
Download or Generate Model Artifacts: The repository should include the pre-trained model files (forecast_10m_xgb.joblib, forecast_30m_xgb.joblib, calibrators, etc.) in a directory (commonly an artifacts/ or models/ folder). If they are not present due to size, you may need to download them from a release or generate them (see Section 7 on Retraining). Ensure the files are placed in the correct locations:
e.g., models/forecast_10m_xgb.joblib, models/forecast_30m_xgb.joblib
e.g., models/forecast_10m_calib.joblib, models/forecast_30m_calib.joblib
The SentinelDataToken.abi.json and SentinelDataToken.bin (compiled bytecode) are included for contract interaction.
The live_dataset.csv (live data simulation) should be in the project root or specified path.
If you skip this, the dashboard’s demo mode might still function with baked-in data, but the full pipeline won’t run.
Verify Installation: You can do a quick check by running an example:
python -c "import xgboost; import streamlit; import web3; print('Deps OK')"
This simply checks that key libraries are importable. You can also run the pipeline in a dry-run mode (see next section) to ensure everything is set.
Now you are ready to run the Depeg Sentinel system or dive into its notebooks and scripts. The next sections explain how to execute the pipeline and use the dashboard.
4. Running the System End-to-End
This section describes how to run the entire pipeline, either step-by-step or via provided scripts/notebooks. You’ll learn what each stage produces and where to find the outputs. A. Using the Automated Pipeline Script (if available):
If the repository includes a main script (for example run_pipeline.py or a similar orchestrator), you can run that directly to execute all steps in order. For instance:
python run_pipeline.py
This would perform data ingestion, anomaly detection, forecasting, calibration, and output generation in one go. Check the repository for such a script or a Makefile task. B. Step-by-Step Manual Execution:
If you prefer to run components manually (or via Jupyter notebooks), follow these steps in sequence:
Data Preparation: Ensure live_dataset.csv is up-to-date with the data you want to analyze. If running on historical data, you might not need changes. For live deployment, the pipeline would fetch fresh data automatically. (In code, functions like sample_once() and run_anomaly_zoo_update_live() handle data fetch and updating the live dataset.)
Anomaly Detection – Score the Detectors:
Run the anomaly detectors on the latest data. This can usually be done by calling a function or script that loads the current data batch and computes all z-scores. For example, in a notebook, you might do:
from score_zoo import score_all_detectors  
df_live = load_live_data()  # load recent data, or read live_dataset.csv  
scores = score_all_detectors(df_live.tail(1))  # compute scores for the latest timestamp
This would return a dictionary or DataFrame with keys like z_if, z_lof, ..., z_ae_seq. The code likely also appends these scores to a live results file or DataFrame. After this step, you should have the individual anomaly scores and the fused anomaly score computed for the current time. The fused score might be computed as:
scores['anom_fused'] = fuse_scores(scores)
where fuse_scores applies the weighting or averaging logic. The latest fused score will also be appended to the live dataset for record-keeping.
Fusion & Logging: (If not done in step 2) Ensure that the anom_fused value is stored. The system may output an updated live dataset (including the new anomaly scores) to a file, e.g., updating live_dataset.csv or writing a new anomaly_scores.parquet. Check RUN_META.json after a run; it often contains pointers to what files were written during the run.
Run Forecast Models (10m & 30m):
Next, use the XGBoost models to predict risk:
import joblib
model_10m = joblib.load("models/forecast_10m_xgb.joblib")
model_30m = joblib.load("models/forecast_30m_xgb.joblib")
X_latest = make_feature_vector(df_live.tail(1))  # construct model features from latest data
p10_raw = model_10m.predict_proba(X_latest)[:,1]  # probability of class=1 (incident) for 10m
p30_raw = model_30m.predict_proba(X_latest)[:,1]
(The make_feature_vector function would select and arrange the appropriate features as the model expects. The features likely include recent values like dev, anom_fused, trends, etc. Refer to how the models were trained to ensure the same features are computed.) Now you have raw forecast probabilities, e.g., p10_raw = 0.65 (65% chance of incident in 10 min), p30_raw = 0.30 (30% in 30 min).
Apply Calibration to Forecasts:
Load the isotonic regression calibrators and transform the raw probabilities:
calib_10m = joblib.load("models/forecast_10m_calib.joblib")
calib_30m = joblib.load("models/forecast_30m_calib.joblib")
p10_calibrated = calib_10m.predict(p10_raw.reshape(-1, 1))[0]  # returns calibrated probability
p30_calibrated = calib_30m.predict(p30_raw.reshape(-1, 1))[0]
After this, suppose we get p10_calibrated = 0.50 and p30_calibrated = 0.25 – these would be the final probabilities that a depeg incident will occur in the respective horizon. They are saved (along with timestamp and maybe other metadata like model confidence) into the forecast results files. Typically:
forecast_10m.parquet will get a new row with columns like ts, risk_forecast_10m (0.50 in this example), etc.
forecast_30m.parquet similarly for 30m.
The system may also print these results to console or log (e.g., “[forecast10m] AP=0.602, Brier=0.216, n=109, calib=iso” meaning it evaluated performance on a test window).
Severity Scoring:
With updated anomaly info and forecasts, determine the severity level (1–5):
Automatic method: Call the severity model’s predictor. If using the provided NLP approach, you might need to update the events first (fetch latest off-chain events with update_events_from_sources() which populates events.json with any new items like Curve status updates). Then:
from severity_model import SeverityModel  # hypothetical class
sev_model = SeverityModel.load("models/severity_model.joblib")  # if pre-trained model exists
severity = sev_model.predict_current(df_live, events.json)
If no complex model is provided, severity could be assigned by rules. For example:
Start with severity = 1 (default).
If dev (price deviation) > 0.5% or fused anomaly > 0.9, set severity = 3 (critical threshold crossed).
If forecast 30m > 0.8 (80%) or dev > 3%, escalate to severity 4.
If dev > 5% or clear signs of depeg, severity = 5.
Additionally, if an off-chain event indicating a major issue is found (like “Hack” or “Custodian failure”), jump to severity 5.
These rules are just an example; adjust them to your needs.
The output severity is saved or noted. The policy might write it to RUN_META.json and use it for the next step.
Generate Explanations (explain.json):
The system can provide interpretability for the forecast. Running the explain function will perform permutation importance on the model inputs:
exp = explain_forecast_10m(n_repeats=8)  # runs permutation importance 8 times for stability
# exp is a dict with "top_contributors": list of features and scores.
This writes explain.json containing something like:
{
  "ts": "2025-09-01T21:00:00Z",
  "top_contributors": [
    {"feature": "dev", "importance": 0.3661},
    {"feature": "anom_fused", "importance": 0.0070},
    {"feature": "dev_roll_std", "importance": 0.0000}
  ],
  "confidence": "High"
}
It indicates, for example, that the price deviation (dev) had the highest impact on the 10m risk (with +0.3661 AP when included vs shuffled), while volatility had negligible impact in that scenario. This info will be used by the dashboard to display "Top drivers: dev, anom_fused, dev_roll_std..." etc. There is also an explain_30m.json if the 30m model explanation is run similarly (not always run each cycle, could be more expensive).
Policy Decisions & Event Logging:
With all the data (fused anomaly, calibrated risks, severity, explainability, drift status), the meta-controller now makes final decisions:
Determine if this constitutes an incident. For example, if severity >= 3 (our threshold for critical), mark this timestamp as an incident. The system might append a record to events.json such as:
{
  "type": "onchain_anomaly",
  "severity": 3,
  "ts": "2025-09-01T21:00:00Z",
  "summary": "Fused anomaly 1.00, 30m risk 0.86 - Critical risk",
  "source": "DepegSentinel"
}
If it’s not severe, it might still log a warning in events.json with severity 2 or so, or not log at all if everything’s normal.
The events.json may also include the off-chain events fetched (with their own severities), and the system could merge them. It deduplicates events and trims the list to recent ones.
Update RUN_META.json: This metadata file will store run-level info such as the current configuration, the paths of artifacts produced, and summary stats. For example, it may note:
{
  "ts": "2025-09-01T21:00:00Z",
  "config": {"eth_rpc": "...", "mock_mode": false, "lookback": 20, ...},
  "artifacts": {
    "live_csv": "live_dataset.csv",
    "forecast_parquet": "forecast_10m.parquet",
    "explain_json": "explain.json",
    "events_json": "events.json"
  },
  "counts": { ... }
}
This helps in debugging and ensuring all outputs are accounted for.
On-Chain Trigger: If the policy decides this is a critical incident and on-chain logging is enabled, it will prepare the data and call the smart contract. This involves:
Composing the block_hash: often taken as the latest Ethereum block hash for a reliable timestamp. The system might fetch it via web3:
latest_block = onchain.w3.eth.get_block('latest')
block_hash = latest_block.hash  # 32-byte value
If running in mock_mode, it might just use a random 32-byte or a hash of current time.
Preparing the anomaly_bps and novelty_bps: these might be derived from the data. For example:
anomaly_bps = int(min(max(dev * 10000, 0), 10000))  
novelty_bps = int(min(max(anom_fused * 10000, 0), 10000))
If dev was 0.03661 (3.661%), anomaly_bps would be ~366. If fused score is 1.0 (100%), novelty_bps = 10000 in this simplistic mapping. (This mapping can be refined as needed.)
Severity is already an integer.
Recipient: determine the recipient address. Perhaps you set this in a config (like the address of the account running the sentinel, or any address meant to collect the reward).
Call the contract function. For example, using web3.py:
contract = onchain.get_contract()  # assume OnChainTool wraps contract instance
tx = contract.functions.submit_data_block(block_hash, anomaly_bps, novelty_bps, severity, recipient_address)
tx_receipt = onchain.send_transaction(tx)
The OnChainTool in the code likely handles loading the ABI, contract address, and using the provided private key to sign the transaction. Ensure you have the contract address configured (perhaps in a config file or in onchain.get_contract() logic). If not, you can instantiate it manually with the ABI.
If successful, the transaction will be mined and emit the DataBlockSubmitted event. The console log or RUN_META might note the transaction hash or a success message.
Completion: At this point, all outputs are generated:
Anomaly scores and fused score are updated in the live dataset.
Forecast probabilities (calibrated) are appended to forecast_10m.parquet / forecast_30m.parquet.
explain.json (and possibly explain_30m.json) hold the interpretability info.
events.json has the list of recent events (combining off-chain and on-chain triggers).
RUN_META.json has the overall summary.
The smart contract may have an on-chain event if triggered.
The pipeline can repeat this periodically (every minute or block) to continuously monitor the pool. In a live setting, you would schedule this run via a cron job or a loop in the Streamlit app with a timer. After each run, the Streamlit dashboard (if running) can read the new files and update the visuals and alerts accordingly.
Note: If you prefer interactive exploration, the repository likely contains Jupyter notebooks that break down these steps. For example, there might be:
AnomalyDetection.ipynb – demonstrates how each detector works on historical data and how the fusion is done.
ForecastModelTraining.ipynb – shows how the XGBoost models were trained with labeled data and how calibration was performed.
EndToEndDemo.ipynb – runs a simulated scenario through the full pipeline, showing intermediate outputs.
SmartContractDemo.ipynb – (if included) shows how to interact with the SentinelDataToken on a test network.
Using those notebooks, you can gain a deeper understanding of each component. For production, however, running the streamlined pipeline (either via script or via the dashboard itself) is the recommended approach.
5. Dashboard Usage
The Streamlit dashboard is the primary interface for users to visualize and interact with the Depeg Sentinel system. Here’s how to use it:
Launching the Dashboard Locally:
Make sure you have followed the installation steps and have all necessary files ready. Activate your virtual environment, then run:
streamlit run dashboard.py
This will start the Streamlit server on your machine (by default at http://localhost:8501). You should see in your terminal some log messages from Streamlit, and you can open the given URL in a web browser.
Initial Load and Data Directory:
On startup, dashboard.py will attempt to load the latest data and artifacts:
It will look for files like live_dataset.csv, forecast_10m.parquet, forecast_30m.parquet, explain.json, etc., usually in the project directory or a specified output folder.
If these files exist (from previous runs of the pipeline), the dashboard will load them and display their contents.
If some or all of these files are missing, the dashboard should handle it gracefully:
In many cases, it will fall back to a demo mode. For example, it might use a small built-in sample of data (perhaps the last known dataset or a synthetic scenario) to populate the charts. You might see a message like "Demo data loaded" or a warning prompting you to run the pipeline.
If the dashboard cannot find critical files and no demo data is defined, it may show empty charts or an error message. In that case, ensure you run the pipeline (Section 4) to generate the needed data, then refresh the dashboard.
Tip: Keep the working directory consistent. If you run Streamlit from the repository root, it will find the files in relative paths as expected. If you change directories or move files, you may need to adjust paths in dashboard.py.
Dashboard Layout:
The exact layout may evolve, but typically you’ll find:
Sidebar Controls: The sidebar might contain configuration options such as selecting which pool to view (if multiple pools), toggling between real-time mode vs. historical analysis, or enabling/disabling on-chain submission. For example, you might have a dropdown of pool names or an input for the update frequency.
Main Charts: The main page will show time-series plots:
Stablecoin Price vs Peg: A line chart of the stablecoin price over time, possibly with a band highlighting the normal range (e.g., $1 ± 0.5%). Sudden drops or spikes will be clearly visible.
Anomaly Score Plot: A plot of the fused anomaly score over time. It might also show the individual detector scores in lighter lines for comparison. The plot could use a color highlight (e.g., red dots) when the anomaly score crosses the warning threshold (0.9).
Forecast Risk Plot/Gauges: It might show two gauges or bar indicators – one for 10m risk and one for 30m risk. If risk is low (green zone) or high (red zone), the color might indicate it. Alternatively, a time-series of predicted risk could be shown to see how risk predictions change over time.
Other Features: The dashboard might also plot supporting features like the oracle_ratio or tvl_outflow_rate if relevant, to show what’s happening under the hood.
Status Readouts: There could be a section displaying the latest values of key metrics (like current deviation in basis points, current volume outflow, etc.) and the current mode of the system (e.g., “MODE: CRITICAL (Severity 4)” or “Mode: Normal”).
Alerts and Events: If an event was triggered, you might see a card or table listing the event details:
E.g., "21:00 UTC – CRITICAL – Fused anomaly 1.00, 30m risk 86%. Action: monitor and prepare mitigation." (This could be derived from the policy’s analyst note and events.json).
Past events or external events could also be listed (with their timestamps and summaries). For instance, if events.json contains an entry from Curve’s status page, it might show “Curve status: Major outage reported (Severity 2)”.
Explainability/Contributors: The top features contributing to risk might be shown as a small list or bar chart (for example, Feature importances: dev (0.37), anom_fused (0.01), ...). This helps the user understand why the model is predicting a high risk.
Controls for On-Chain: If the app allows, there might be a toggle or button to manually trigger on-chain submission. However, given the pipeline automates it, the dashboard may simply reflect whether on-chain mode is active. If ETH_RPC is configured and healthy, perhaps a label like “On-Chain Logging: Enabled (Network: Mainnet)” is shown; if not, “On-Chain Logging: Off (Mock mode)”.
Demo Mode Behavior:
If you have not run any analysis yet and just launch the dashboard, expect one of two things:
The dashboard might show static example charts (to illustrate what it would show). You can use this to familiarize yourself with the interface.
Or it might actually attempt to run a quick pipeline on a sample of live_dataset.csv to generate initial results. This depends on how dashboard.py is written. In some cases, the app could call functions to produce an initial state (like a warm-up). Watch the terminal for any logs like “[run] sample → zoo → train/score → explain → reports” which indicate it executed a pipeline internally on startup.
Once you have real data flowing (from running the pipeline externally or connecting to live data), the dashboard will update accordingly.
Deployment on Streamlit Cloud:
The app can be deployed to Streamlit Cloud (share.streamlit.io) to make it accessible via the web. To do this:
Push your repository to GitHub (if not already).
On Streamlit Cloud, set up a new app, pointing to dashboard.py in your repo.
Secrets: Add any required secrets (like ETH_RPC and PRIVATE_KEY for on-chain use) in the Streamlit app’s secret management interface. The code will read ETH_RPC from environment, which Streamlit will populate from secrets.
Data and Models: Include the necessary data in the repo or have the app fetch them. Large model files can be a challenge on free tiers. If they are too big for GitHub, consider hosting them on a cloud bucket and modify the app to download them at startup (caching for future runs). Alternatively, retrain smaller models on the fly (not ideal for heavy XGBoost).
Resource Considerations: Streamlit Cloud has limited resources. The 10m/30m models and calculations are lightweight, but if you incorporate heavy tasks (like large autoencoders or frequent web requests), ensure to adjust the frequency or use caching to not hit resource limits. The default config uses a small lookback and should be fine.
Usage: Once deployed, you and others can visit the Streamlit app URL and monitor the stablecoin pool remotely. It could serve as a public dashboard for the community if you choose.
Stopping the Dashboard:
If running locally, you can stop the server by pressing Ctrl+C in the terminal where Streamlit is running. If deployed, you can manage it via the Streamlit Cloud interface.
By following these steps, you should have a fully functional dashboard that updates in real-time (or near-real-time) with anomaly scores and risk forecasts. It’s a good idea to test the system by feeding it known scenarios (if you have historical data of a depeg event, run that through and see if it catches it and how it responds).
6. Smart Contract Integration
One of the innovative parts of this project is the optional integration with an Ethereum smart contract, enabling on-chain logging and rewards. This section explains how to set up and use the SentinelDataToken contract:
Contract Overview:
The SentinelDataToken is a Vyper contract (see contracts/SentinelDataToken.vy or the file in the repo) that implements:
An ERC20-compatible token (with name, symbol, decimals, balance tracking, transfer/approve functions).
Restricted minting through the submit_data_block function, as described earlier.
Pausable functionality and an owner, treasury, and minter roles for admin control.
By deploying this contract, you create a token which can represent “sentinel rewards” or simply be a mechanism to record events on-chain. The token could be given any name and symbol (e.g., “Sentinel Alert Token”, symbol “ALERT”), which you specify during deployment.
Deployment:
If the contract is not already deployed for you, you will need to deploy it to your desired network:
Ensure you have Vyper ^0.3.9 installed, or use an online compiler for Vyper.
Compile SentinelDataToken.vy to get the bytecode and ABI (the repository already provides SentinelDataToken.bin and SentinelDataToken.abi.json which you can use directly).
Use a deployment script or tool (like Remix, Brownie, Hardhat, etc.) to deploy. You will need to supply the constructor arguments:
name_ (string): e.g., "SentinelDataToken"
symbol_ (string): e.g., "SDT"
treasury_ (address): the Ethereum address that will receive the treasury portion of rewards (can be your project’s treasury or even a burn address if not needed).
minter_ (address): the address allowed to call submit_data_block. Set this to the Ethereum address that your sentinel bot will use. If you haven’t decided, you can set it to your own address and later call set_minter() to update it.
treasury_bps_ (uint256): the basis points for treasury cut. For example, 1000 means 10% of each reward goes to treasury, 0 means no cut (100% to recipient). Use an integer between 0 and 10000.
Once you deploy, note the contract address. Update your configuration (perhaps in the code or an environment var) so the Python knows where the contract is.
If the repository maintainers already deployed an instance (e.g., on a testnet), they might provide the address. In that case, you can skip to using that contract.
Connecting in Python (Web3):
The pipeline uses the Web3.py library to interact with Ethereum. If you set ETH_RPC to a valid node URL, the system will initialize the connection on startup. In code, an OnChainTool class handles this:
from web3 import Web3
web3 = Web3(Web3.HTTPProvider(CFG.eth_rpc))
contract = web3.eth.contract(address=CONTRACT_ADDRESS, abi=json.load(open("SentinelDataToken.abi.json")))
The OnChainTool likely wraps the above and provides convenience methods. To call the contract:
Ensure web3.eth.default_account is set to the minter’s address (and that the private key for this account is loaded into Web3, via web3.eth.account.from_key() or by using a local node’s accounts).
Form the transaction for submit_data_block. Using Web3.py low-level:
tx = contract.functions.submit_data_block(block_hash, anomaly_bps, novelty_bps, severity, recipient).buildTransaction({
    'from': web3.eth.default_account,
    'nonce': web3.eth.getTransactionCount(web3.eth.default_account),
    'gas': 100000,  # estimate or adjust
    'gasPrice': web3.toWei('10', 'gwei')  # example gas price
})
signed_tx = web3.eth.account.sign_transaction(tx, private_key=MY_PRIVATE_KEY)
tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)
receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
print("Submitted to chain, tx hash:", tx_hash.hex())
In the project code, you might not need to write this from scratch – the OnChainTool possibly does it, given the private key.
The above will throw if something is wrong (e.g., if from is not the minter or if block_hash was used before). Catch exceptions to log errors. For example, the contract will assert not self.used_block[block_hash] and assert msg.sender == self.minter, which correspond to errors "dup" (duplicate submission) or "only minter" if violated.
Example usage:
Let’s say our pipeline determined:
block_hash = 0xabc123... (just an example 32-byte value),
anomaly_bps = 450 (meaning 4.5% anomaly),
novelty_bps = 300 (3% novelty),
severity = 4 (on a scale of 5),
recipient = 0xYourRecipientAddress.
We can call the contract as follows (assuming web3 and contract are set up):
tx = contract.functions.submit_data_block(
    bytes.fromhex("abc123..."),  # block hash as bytes32
    450,
    300,
    4,
    Web3.toChecksumAddress("0xYourRecipientAddress")
).transact({'from': web3.eth.default_account})
receipt = web3.eth.wait_for_transaction_receipt(tx)
If the transaction succeeds, receipt will contain the event. You can decode the DataBlockSubmitted event from it or just trust that it’s on chain. The event would look like:
DataBlockSubmitted(
  block_hash=0xabc123..., 
  anomaly_bps=450, 
  novelty_bps=300, 
  severity=4, 
  reward= someValue, 
  to=0xYourRecipientAddress
)
The reward value in the event is the total tokens minted (in our case, reward = 1e18 * (450+300)*4/50000 = 1e18 * 750 * 4 / 50000 = 1e18 * 3000 / 50000 = 0.06 * 1e18, so 0.06 tokens). The to will match the recipient.
Using the ABI and Bytecode:
The file SentinelDataToken.abi.json is provided so you don’t have to manually reconstruct the contract interface. In case you want to verify the contract or redeploy it, the Vyper source is available. The SentinelDataToken.bin contains the compiled bytecode which can be used for deployment (e.g., via web3.py contract.constructor(args...).transact() method, supplying the bin and abi).
Viewing On-Chain Data:
Once events are being submitted, you can use any Ethereum blockchain explorer to view them. If on a testnet or mainnet, find the contract address in Etherscan and look at the Events tab for DataBlockSubmitted. Each event entry is a record of when the sentinel reported something. This provides transparency – even parties who aren’t running the sentinel can see these alerts on chain. If the token has a utility (maybe governance or staking), those tokens could then circulate; otherwise, it’s mainly a signaling mechanism.
Security Considerations:
Make sure the minter key is secure. If someone else gets hold of it, they could spam false events or mint unlimited tokens (though they'd need an authorized block_hash each time, but they could just use new random hashes).
The contract is fairly straightforward, but note that it does not impose any fee or cost on using submit_data_block beyond gas. If running on Ethereum mainnet, consider the gas cost – you might want to set a policy to only submit truly critical events to avoid high costs for minor warnings. On a cheaper chain or L2, you could submit more frequently.
block_hash should ideally be something verifiable. Using the actual Ethereum block hash ties the event to a specific block (which has a timestamp). However, by the time your transaction is mined, that “latest” block hash might be a few blocks old. This is fine; it’s just an identifier. Alternatively, one could use a hash of the data payload (like hash of the JSON of features) to later prove off-chain what was submitted. In any case, ensure it’s unique per event (the contract will reject duplicates).
In summary, the smart contract integration allows the Depeg Sentinel to bridge to on-chain, turning an off-chain analysis into an actionable, traceable on-chain alert. It’s optional but adds a layer of trust and incentive. If you prefer not to involve blockchain, you can ignore the contract (the pipeline will run in mock mode and simply not mint tokens). If you do enable it, double-check your addresses and keys, and perhaps test on a testnet (like Goerli) before using mainnet.
7. Retraining & Drift Handling
The crypto markets evolve rapidly, and models can become stale. This project includes mechanisms to detect when the models might need retraining, and it provides guidance for how to retrain and update the pipeline. Drift Detection:
As mentioned in the Architecture section, the system computes statistical drift metrics comparing recent data to the original training data for each feature. The results are saved in feature_drift.json with fields:
drift: true/false – whether drift is detected overall.
reason – a short reason, e.g. "feature drift" or "concept drift".
thresholds – the cutoff values for metrics (e.g., KS > 0.2 or PSI > 0.25 are considered drift).
metrics – the actual values for each feature’s drift test. For example:
"dev": {"ks": 0.24, "psi": 4.88},
"oracle_ratio": {"ks": 0.24, "psi": 4.88},
"anom_fused": {"ks": 0.35, "psi": 4.77},
"spot_twap_gap_bps": {"ks": 0.06, "psi": 0.10},
...
"n_train": 1607, "n_recent": 690
In this snapshot, dev (price deviation) has a PSI of 4.88 which is far above 0.25 – a significant distribution shift. The number of training samples vs recent samples is also given, indicating they compared 1607 training points to 690 recent points. The drift reason was "feature drift", implying some features changed distribution enough to possibly undermine model assumptions.
When drift: true, the sentinel’s policy might:
Suppress high confidence in the forecasts (perhaps by widening prediction intervals or requiring a higher threshold to trigger alerts, acknowledging the model is less certain in new territory).
Signal that model retraining is required. The RUN_META.json or logs might contain a note or warning to this effect.
When to Retrain:
You should consider retraining your models in scenarios such as:
Feature Drift Detected: As soon as feature_drift.json shows drift = true consistently (not just a one-off blip), it’s an indicator the data now is quite different from the training set. For example, if a stablecoin changed its peg mechanism or a new type of arbitrage is happening, the old model might not capture it.
Performance Degradation: If you notice the model is missing incidents (false negatives) or raising too many false alarms (false positives) in current data, that’s a sign. You might measure performance by continuing to label incidents and see if the forecast AP or Brier score is worsening over time (the logs printed some metrics like AP=0.602, Brier=0.216 for a test set; if you update those test sets and see a drop, retrain).
Periodic Update: Even if not forced by drift, you might retrain on a schedule (e.g., monthly) to incorporate the latest data and any new incidents that occurred, ensuring the models stay up-to-date.
How to Retrain:
Retraining involves a few components:
Gather New Training Data: Accumulate the recent live data since the last train, including the period where drift was detected or incidents occurred. Merge this with your original training data (or use only recent data if you want the model to adapt fully to new regime – but beware of forgetting old patterns).
For anomaly detectors (like Isolation Forest, autoencoder): If these were trained on a baseline of “normal” behavior, you might need to update what "normal" means. For example, if the pool’s typical volume has grown 10x, the detector needs to know that high volume is normal now. Some detectors are unsupervised and might be recalculated on new data distribution. If they run on each new batch without learning, you might not retrain them per se, but you might update any thresholds.
For forecast models: You need to re-label the combined dataset with incidents.
Labeling approach: The original labeling criteria (e.g., dev > 0.005 or fused > 0.9 defines an incident) should be applied to the new data as well (Section 8 details this). Ensure consistency: if you adjust thresholds (say you realize 0.005 was too sensitive, maybe raise to 0.01), apply to both old and new data for training.
Create the feature matrix X and label vector y for both 10m and 30m horizons. This likely involves generating sequences: For each time t in historical data (except the last 10 or 30 min), mark y=1 if any incident condition was met within the next 10 (or 30) minutes after t.
The repository might have a training script that does this labeling and uses XGBoost’s API to train. If not, you can use pandas to create labels and then xgboost.train() or even scikit-learn’s XGBClassifier to fit.
Update drift reference: If you want the drift detection to now consider the new data as baseline “normal”, you should recompute the baseline distribution. In practice, that could mean replacing the old training stats with new ones or expanding the window.
Train New Models: Using the prepared dataset:
Train a new XGBoost model for 10m horizon. Tune hyperparameters if needed (or use the existing ones if they were good). Make sure to evaluate it (perhaps via cross-validation or a hold-out set).
Train the 30m model similarly.
Save the models as forecast_10m_xgb.joblib and forecast_30m_xgb.joblib (or new filenames, then update the code to load those).
If you changed any features or model types, also update code in pipeline that constructs features and calls the model.
Re-run Calibration: With new models (and possibly new data distribution), calibration should be redone:
Use a validation set or the tail of your training data to compute calibration curves. For each model, bucket the predictions into 10 bins and compute how many of those were actually positive.
Fit an isotonic regression model mapping raw prediction to calibrated probability (or use Platt’s logistic calibration if you prefer).
Save the new calibrators (forecast_10m_calib.joblib, etc.) and update any calibration JSON (the _save_calibration_artifacts() function in code can output a JSON and PNG of the calibration plot).
It’s helpful to inspect the calibration plots (calibration_10m.png) to ensure the calibration is making the predictions more accurate.
Update Severity Model (if applicable): If you have expanded your events.json with new events (like actual incidents that happened), consider retraining the severity NLP model:
The severity model looks at text summaries of events and tries to predict a severity 1–5. If you have new examples of high-severity events with descriptions, fine-tune the model on those.
Use train_severity_calibrator() function to recalibrate severity thresholds if needed. This function likely uses the events in events.json to adjust how the model’s probability output corresponds to severity >=3 (critical) vs <3.
Save any updated model or calibrator (e.g., severity_model.joblib or similar).
Testing: After retraining, run the pipeline on known scenarios, including some incidents from the new training period, to verify the new models indeed catch them with appropriate risk and severity. Check that false alarm rate is reasonable.
Deploy Updates: Replace the old model files with the new ones in the project. If you version-control these, keep copies of old models in case you need to roll back.
Update version.json or some metadata if provided, to reflect new model versions.
If using Streamlit Cloud or other deployment, upload the new artifacts there too (or if they fetch from an URL, update the URL).
Unlock Drift Mode: If you had suppressed or altered behavior due to drift detection, once models are retrained on the new distribution, you can reset that. The drift detector will then likely mark drift: false after retraining (since now recent = training distribution). In code, you might simply remove the old feature_drift.json or update n_train to include the new data.
Automating Retraining:
For advanced use, you could automate this process. For example, if drift is detected or every X weeks, automatically:
Spin up a training job (maybe in a separate environment to not stall the live sentinel).
Possibly use the current live data stored in live_dataset.csv as part of training.
Once done, have the sentinel reload the new models (the code might need a restart or have a mechanism to hot-swap models).
However, automation should be done carefully (you want to validate new models before trusting them). In most cases, a manual retraining cycle with human oversight is safest. Drift in Anomaly Detectors:
If drift is significant, some anomaly detectors like Isolation Forest or LOF might need to be refit as well. They might have been initially fit on some baseline period. Consider:
Isolation Forest: If explicitly trained on initial data, retrain it on a newer sample of normal data.
Autoencoder: Retrain the neural network on new data (this assumes you have the infrastructure to do so – if not, perhaps stick to tree models).
CUSUM: Typically doesn’t train; it has a predefined threshold and can adapt on its own over time. You might adjust its threshold if the variance of the series changed.
One-Class SVM: If used, you might retrain it on updated normal data too.
All these updated detectors should then be tested and their outputs recalibrated if you use any thresholds. TL;DR for Retraining: When the stablecoin or pool behavior changes (detected by drift or observed by you), don’t ignore it! Update the models with fresh data. The repository’s notebooks likely have instructions or at least code snippets to retrain models (look for sections that train XGBoost or mention calibration). Follow those as a guide. Finally, after retraining and updating, monitor the performance closely. Ideally, the new models should reduce any false alarms that drift caused and accurately predict any new type of incidents.
8. Labeling & Calibration Design
To effectively train and calibrate the models, a consistent labeling strategy and careful calibration design are crucial. This section describes how incidents are labeled in the data and how the calibration process is structured. Incident Labeling Criteria:
In the context of stablecoin pools, deciding what counts as a "positive" event (an incident/depeg) for model training isn't always straightforward. The approach taken in this project is to use a combination of rules on key indicators:
Price Deviation Threshold: If the stablecoin’s deviation from peg (dev) exceeds 0.005 (i.e., 0.5%), it is considered a significant deviation. In other words, if the pool price of the stablecoin drops below $0.995 or rises above $1.005 for USD-pegged coins, that's a notable event. This threshold (0.5%) is somewhat conservative (Moody’s, for example, might consider >3% a depeg
home.treasury.gov
, but for early warning we use 0.5%). This catches early signs of trouble.
Fused Anomaly Score Threshold: If the fused anomaly score > 0.9, that indicates our ensemble of detectors is 90% confident something is very abnormal. Even if price hasn’t moved much yet, such a high anomaly score could precede a depeg (for example, unusual outflows or oracle divergence might foreshadow price impact). So we label those cases as incidents too.
Using these criteria, we label a data point (or a time window) as an “incident” (positive class, y=1) if either condition is met. This ensures we capture both actual price depegs and precursor anomalies as positive examples. Everything else is labeled as negative (normal/no incident). When preparing training data for the forecast models:
For each time t, we look ahead H minutes (H=10 for the 10m model, 30 for the 30m model). If any time in the interval [t, t+H) meets the incident criteria (dev > 0.005 or fused > 0.9), we label the instance at time t as y=1. If none do, y=0.
This effectively means the model is trying to predict: “Will we see a 0.5% deviation or 0.9 anomaly score in the next H minutes?”
We might refine the labeling by requiring the condition to hold for a sustained period or at least two consecutive readings, to avoid spurious one-off spikes. But the simplest form is as above.
These thresholds (0.005 and 0.9) were chosen based on domain knowledge and some empirical testing. You can adjust them if needed:
Lower thresholds = more incidents labeled (sensitive, but risk more false positives).
Higher thresholds = fewer incidents (focused on only severe events, risk missing early signs).
Severity Levels and Modes:
The labeling above is binary (incident vs not). However, in the running system we also categorize severity 1–5:
Warning Mode: severity 1-2. Typically corresponds to minor anomalies that do not meet the incident threshold. For example, dev = 0.002 (0.2%) might be severity 1 (just a blip), dev = 0.004 (0.4%) with some odd volume changes might be severity 2 (elevated, but not critical). In warning mode, the system might only log or show a yellow status but not trigger on-chain events.
Critical Mode: severity 3-5. Once the incident threshold is crossed (dev > 0.5% or fused > 0.9, as above), we enter critical mode. Severity 3 is the entry level of critical. Severity 4 and 5 would indicate more extreme conditions:
Severity 4 could mean dev > 0.02 (2%) or forecast predicts a very high chance (say > 80%) of depeg.
Severity 5 might be reserved for confirmed depeg (dev > 0.05 or stablecoin breaking $0.95, or a combination of multiple factors like on-chain events confirming a hack).
External events can also jump severity to 5 instantly (e.g., an official announcement of collapse).
The exact mapping to severity 3,4,5 can be tuned. But importantly, the threshold between 2 and 3 delineates warning vs critical. In the severity calibrator code, this threshold is used when converting to binary for calibration:
y_bin = (y >= 3).astype(int)
This means they treat severity 1-2 as class 0 (non-critical) and 3-5 as class 1 (critical) for the purpose of calibrating the severity classifier. Essentially, they ensure that the distinction between warning and critical is meaningful and calibrated. Forecast Calibration Bins:
Calibration is performed by taking the raw probability outputs of the forecast models and adjusting them. The project’s approach:
Use 10 bins (by default) from 0 to 1. For example, [0.0–0.1), [0.1–0.2), ..., [0.9–1.0].
For each bin, calculate:
p = midpoint (or average) of predictions in that bin.
observed = fraction of actual incidents among those predictions.
n = number of samples in that bin.
Create a table of these (“calibration curve”). The isotonic regression essentially draws a monotonic piecewise constant or linear function fitting these points (or a smoothed version if data is scarce).
The JSON files calibration_10m.json and calibration_30m.json store the results. For instance:
{
  "horizon_min": 10,
  "bins": [
    { "pred_bin": 0.05, "observed": 0.1, "n": 50 },
    { "pred_bin": 0.15, "observed": 0.2, "n": 40 },
    ...
    { "pred_bin": 0.85, "observed": 0.9, "n": 10 }
  ],
  "counts": { "n": 300, "positives": 30 }
}
This is just an illustrative example. It shows maybe the model tended to underpredict a bit (when it said 5% it was 10% in reality, etc.). The calibrator would adjust predictions upward in that range.
In our earlier actual artifact, calibration_10m.json had bins: [] and counts 60/60, which implies perhaps all 60 samples were positive incidents so they couldn’t derive a curve (this might have been a degenerate case, possibly initial data was all incident or a placeholder). In a proper training, you’d expect a mix of positives and negatives.
The calibration process also gives metrics like Brier score (mean squared error of probabilistic predictions) which was reported in logs, and this should improve after calibration (or at least not worsen significantly).
Why Calibration Matters: If the model outputs “0.5” for risk, we want that to truly mean “50/50 chance”. If it doesn’t, decisions based on it (like whether to trigger mitigation) could be off. Calibration ensures a threshold like >0.7 for alarm is justified by historical frequency. Labels and Model Training:
During model training (XGBoost):
It optimizes a loss (likely logistic loss for binary classification). If the data is imbalanced (few incidents vs many normal points), XGBoost’s default might bias towards the majority. We probably provided it with enough incidents via the labeling scheme, but if needed, one can give a scale_pos_weight or use techniques to handle imbalance.
The average precision (AP) of the 10m model was about 0.60 in one test, meaning it’s doing substantially better than random (AP 0.5 would be random for balanced data). There is room to improve, but given the volatile nature of crypto, 0.6 AP might be acceptable for a first cut model.
Thresholds for Actions:
Even after calibration, the system needs thresholds for deciding actions:
For example, the policy might use 0.6 (60%) as a threshold on the 30m calibrated risk to move from “monitor” to “take action”.
Similarly, anomaly fused > 0.9 is a threshold for labeling, but for triggering immediate alert, perhaps an even higher threshold (like 0.95) could be used to be sure.
These thresholds are part of the policy configuration and can be adjusted as you gather more feedback. It might be beneficial to expose them as easily configurable parameters (maybe via a config file or even the Streamlit sidebar for experimentation).
Summary: The labeling strategy defines what we consider a depeg incident (ground truth), and the calibration ensures our model’s predicted probabilities align with reality. Warnings vs critical allow us to have a two-tier alerting system. All of these are tunable:
If you find the system is too noisy (too many false alarms), you might raise thresholds (e.g., require dev > 1% for labeling, or require forecast > 0.8 to alert).
If it’s missing events, you might lower thresholds or incorporate more signals (e.g., include social media alerts as severity triggers).
It’s recommended to document any changes you make to these thresholds and rationale, so that collaborators or future maintainers can understand the choices (perhaps in the repo’s Wiki or a config YAML).
9. Troubleshooting & Debugging
Even with a solid setup, you might encounter issues when running the Depeg Sentinel. Below are common problems and how to address them:
Dashboard shows no data / graphs empty:
This typically means the dashboard couldn’t find the artifact files or they contain no entries.
Ensure that you have run the pipeline at least once to generate forecast_10m.parquet, forecast_30m.parquet, etc. The dashboard does not itself create forecasts (unless it has a demo mode, which might be static).
Check the terminal running Streamlit for error messages. If it says file not found, place the file and refresh.
If you expect demo mode to kick in but nothing shows, the demo data might be missing or there’s a bug. In that case, manually run a small pipeline (maybe call sample_once() and related functions in a Python shell) to produce initial data.
Verify the working directory: if you run streamlit run dashboard.py from a different folder, the relative paths may break. Always run it from the project root (or adjust the code to use absolute paths).
Module or dependency import errors:
e.g., “No module named ‘xgboost’” or “No module named ‘web3’”. This means the pip install failed or your environment isn’t activated. Re-activate the virtualenv (source venv/bin/activate) and ensure pip install -r requirements.txt ran without errors. You can try installing missing packages individually.
If using Jupyter notebooks, make sure the kernel is using the correct environment. Otherwise, the notebook might run in a different env lacking the dependencies.
Model file not found or cannot be loaded:
If you see an error like “FileNotFoundError: forecast_10m_xgb.joblib not found”, confirm that the file exists in the path the code expects. The code might be looking in a models/ directory or artifacts/.
If the file exists but still fails to load, the issue might be version mismatch. For example, if the joblib was created with XGBoost 1.5 and you have XGBoost 1.6, compatibility should usually hold, but in some cases, pickled models break across versions. Try to keep the same XGBoost version. If needed, retrain or re-save the model with your current version.
For the calibrator .joblib (if any), ensure scikit-learn version is compatible. IsotonicRegression objects should be fine across minor versions of scikit-learn, but if there’s an error, you might retrain calibrators on your machine.
Streamlit app crashes or some components not updating:
Streamlit might crash if it runs out of memory or if a function is misbehaving (infinite loop or heavy computation on the main thread). Check for any while loops or large data being held.
If using live updates, note that Streamlit reruns the script from top whenever something changes (like a widget). Make sure expensive operations are cached or placed appropriately. The code might use st.experimental_singleton or st.cache_data for loading models to avoid reloading on each refresh.
If plots don’t update, it could be caching. Try disabling cache or using a unique key for the component that changes.
Logging & Debugging information:
In Streamlit: you can insert st.write() or st.text() calls to print debug info on the app or view the app’s logs. Alternatively, run streamlit run with -v (for verbose) or check ~/.streamlit/logs/*.log.
Pipeline logs: The pipeline prints various [note], [ok], [warn] messages to stdout. If running via a script, watch the console for these. They often pinpoint issues like:
“forecaster step skipped: Invalid classes inferred” – This indicates the forecast training or explanation step found only one class in the window (maybe no incidents in recent data). It’s a warning that it couldn’t update or explain due to lack of positive examples. This is not fatal, just means at that moment it didn’t refresh the model. It can be ignored unless it persists (in which case your labeling might be too strict, yielding no positives).
“[offchain] fetch error ... NameResolutionError ...” – This shows an attempt to fetch external data (e.g., Curve status) failed due to internet issues. On Streamlit Cloud, internet access might be restricted, so that could happen. If off-chain events are not critical, you can ignore or disable those fetches. Or run the app where it has internet.
“[events] load failed” or “[events] save failed” – Indicates issues reading/writing the events.json, possibly a file permission problem. Make sure the artifacts/ directory is writable. In a read-only environment, direct file writes might fail. On Streamlit Cloud, the app can write to /app directory since it’s ephemeral. Consider using st.session_state or a database if persistence is needed across sessions.
“[rpc] Real RPC not available/healthy → mock mode.” – This means your ETH_RPC was not set or failed the health check. If you expected it to connect, double-check the URL and that the network is accessible. If you see this on a local run and you intended mock mode, it’s fine.
Using a debugger: For deeper issues, you can open the project in an IDE and use breakpoints, or sprinkle print() statements. If the app is running continuously, logs will be your main tool, as attaching a live debugger to Streamlit can be complex.
Smart contract transaction errors:
If on-chain submission fails:
If you get a revert saying “only minter”, it means the from address in the transaction is not the contract’s minter. Ensure you loaded the correct private key and that the contract’s minter state matches that address (you can call the minter() view function via web3 to confirm).
If the error is “dup” (duplicate block hash assertion), it means you tried to submit the same block_hash twice. This could happen if you mistakenly reuse the ID or if you call multiple times in the same block with the same id. Solution: use truly unique IDs (e.g., incorporate a timestamp or a counter in the hash).
“out of gas” error: Increase the gas limit in the transact call. Submitting data isn’t very heavy (just storage and mint), but ensure you provide enough, e.g., 100k should suffice; if you had a loop or large data (not the case here), you’d need more.
If transactions are sent but not confirmed (no receipt), maybe nonce issues or no funds: Check the account has ETH for gas. On testnets, get some from a faucet. If nonce mismatch, perhaps the account is used elsewhere; you might reset the nonce or use web3 to fetch current nonce properly.
On a testnet, if you never see the event, ensure you’re connected to the correct network and that the contract address is correct on that network.
High false positive rate (too many alerts):
If the sentinel is crying wolf too often:
Possibly the threshold is too low or calibration is off. Check if the forecast probabilities are consistently high even for normal periods. If yes, perhaps the model is biased or the calibration isn’t properly applied. Re-evaluate the training data – maybe the model learned to correlate a benign feature with incidents incorrectly.
You can tighten the criteria for alert: e.g., require both dev > 0.005 AND anomaly > 0.9 (instead of OR) for labeling, or raise fused anomaly threshold to 0.95 for triggering.
Use the explain feature to see why the model is giving high risk. It might highlight a feature that is always pushing risk high due to a spurious correlation. If so, consider retraining without that feature or with more data.
High false negative rate (missed an incident):
If a depeg or big anomaly happened but the sentinel didn’t flag it:
Check if the data was ingested properly. If live_dataset.csv or the data source had an outage or delay, the system might not have seen the data in time.
Was the incident outside the feature space the models know? If something completely new happened (e.g., oracle was manipulated in a new way), the detectors might not have recognized it if not trained for it. This might require adding a new detector or retraining on that scenario.
Increase sensitivity: lower thresholds, or incorporate a direct rule (for example, if stablecoin price < 0.97 at any point, you might want an alert regardless of model).
Use the logs: did anomaly detectors fire? If yes and it still didn’t alert, maybe the fusion logic down-weighted it wrongly. Adjust the fusion weighting (e.g., if z_cusum caught it but z_if didn’t, perhaps increase weight of z_cusum or include that scenario in training so z_if learns it).
Compatibility and Misc Issues:
Ensure that the Python version you use is supported. (If Vyper 0.3.9 is needed, Python 3.9+ is fine. The rest should work on 3.8-3.11, but check any specific dependency versions).
If using Windows, some things like fork methods or certain library binaries might cause issues. All major libraries used (streamlit, web3, xgboost) support Windows, so it should be okay. But if you encounter path issues, adjust file paths (use raw strings or proper os.path.join).
If you see RuntimeWarning or UserWarning messages in output (the code shows some warnings like "[note] sklearn missing; using prior-only severity." or similar), they are informing about fallbacks:
"sentence-transformers missing; using prior-only severity." means the NLP model for severity didn’t load a transformer model, so it’s just using a prior probability (basically not actually NLP-ing). To fix, install sentence-transformers or related requirements if you want full functionality.
"LR fit failed: ... Using prior-only severity." means maybe not enough data to fit severity model, so it defaults to using a base rate. This is usually fine; it just means severity classification isn’t sophisticated in that run.
In general, the system is meant to be robust with defaults (e.g., if events fail to fetch or severity model is not present, it continues with reduced functionality). Most troubleshooting will revolve around data and configuration issues. Checking the content of the JSON outputs and logs is often the fastest way to pinpoint where something went wrong. Don’t hesitate to add more logging if needed, especially when extending the system. A well-placed print(f"Debug: variable = {value}") can save a lot of time when tracking an issue in the pipeline.
10. License and Attribution
This project is open source and is released under the MIT License. You can find the full text in the LICENSE file of the repository. In summary, MIT License allows you to use, modify, and distribute this software freely, provided you include the copyright notice. There is no warranty—use at your own risk.

Attribution

The DeFi Depeg Sentinel was developed by Anirudh Reddy Ninganpally. If you use significant portions of this project in your own work or research, please give credit by linking back to this repository.

The smart contract was written in Vyper. If you reuse it, consider acknowledging this repository in your contract documentation.

This project drew inspiration from real-world depeg incidents and research on stablecoin risk. We acknowledge the DeFi community’s contributions in identifying metrics (like using TWAP vs. oracle spread) that signal trouble.

The stablecoin pool data used for training and demo is courtesy of public data from protocols like Curve Finance. Any trademarks (e.g., names of protocols or stablecoins) are property of their respective owners and are used here for identification only.

We leveraged several open-source Python libraries: Pandas, NumPy, Scikit-learn, XGBoost, PyOD, Streamlit, Web3.py, Sentence-Transformers, and others. Huge thanks to those communities for making such powerful tools available.

If any code in this repository was adapted from examples or other projects, attribution is given via comments in the code (e.g., parts of the anomaly detection approach may be inspired by scikit-learn examples).

This project was developed as a research initiative.

By adhering to the MIT License and providing proper attribution, you help keep the spirit of open source and collaboration. We encourage you to contribute improvements or new ideas to this repository—feel free to open issues or pull requests!