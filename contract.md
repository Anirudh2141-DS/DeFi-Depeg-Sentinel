# Depeg Sentinel MCP — Contract Cheatsheet

Auth headers (send all requests with these):

```
X-API-Key: <YOUR_KEY>
X-Timestamp: <unix-seconds>
X-Signature: HMAC_SHA256( SECRET, f"{ts}.{body}" )
Content-Type: application/json
```

## /ml/score_zoo  `GET`
Returns parallel anomaly scores.

```bash
ts=$(date +%s)
sig=$(python - <<'PY'
import hmac,hashlib,os
sec=os.environ.get('HMAC_SECRET','secret').encode()
body=''
ts=os.environ.get('TS_OVERRIDE',str(int(__import__('time').time())))
msg=f"{ts}.{body}".encode()
print(hmac.new(sec,msg,hashlib.sha256).hexdigest())
PY
)
curl -sS -X GET "$BASE/ml/score_zoo" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Timestamp: $ts" \
  -H "X-Signature: $sig" \

```

## /ml/forecast  `GET`
Returns 10m/30m risk probabilities for the tail rows.

```bash
ts=$(date +%s)
sig=$(python - <<'PY'
import hmac,hashlib,os
sec=os.environ.get('HMAC_SECRET','secret').encode()
body=''
ts=os.environ.get('TS_OVERRIDE',str(int(__import__('time').time())))
msg=f"{ts}.{body}".encode()
print(hmac.new(sec,msg,hashlib.sha256).hexdigest())
PY
)
curl -sS -X GET "$BASE/ml/forecast" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Timestamp: $ts" \
  -H "X-Signature: $sig" \

```

## /ml/explain  `GET`
Top contributors for the current window.

```bash
ts=$(date +%s)
sig=$(python - <<'PY'
import hmac,hashlib,os
sec=os.environ.get('HMAC_SECRET','secret').encode()
body=''
ts=os.environ.get('TS_OVERRIDE',str(int(__import__('time').time())))
msg=f"{ts}.{body}".encode()
print(hmac.new(sec,msg,hashlib.sha256).hexdigest())
PY
)
curl -sS -X GET "$BASE/ml/explain" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Timestamp: $ts" \
  -H "X-Signature: $sig" \

```

## /ml/top_events  `GET`
High-severity recent events.

```bash
ts=$(date +%s)
sig=$(python - <<'PY'
import hmac,hashlib,os
sec=os.environ.get('HMAC_SECRET','secret').encode()
body=''
ts=os.environ.get('TS_OVERRIDE',str(int(__import__('time').time())))
msg=f"{ts}.{body}".encode()
print(hmac.new(sec,msg,hashlib.sha256).hexdigest())
PY
)
curl -sS -X GET "$BASE/ml/top_events" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Timestamp: $ts" \
  -H "X-Signature: $sig" \

```

## /ml/network  `GET`
Cross-pool network features and lead/lag.

```bash
ts=$(date +%s)
sig=$(python - <<'PY'
import hmac,hashlib,os
sec=os.environ.get('HMAC_SECRET','secret').encode()
body=''
ts=os.environ.get('TS_OVERRIDE',str(int(__import__('time').time())))
msg=f"{ts}.{body}".encode()
print(hmac.new(sec,msg,hashlib.sha256).hexdigest())
PY
)
curl -sS -X GET "$BASE/ml/network" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Timestamp: $ts" \
  -H "X-Signature: $sig" \

```

## /policy/decide  `POST`
Returns meta-controller policy decision.

```bash
ts=$(date +%s)
sig=$(python - <<'PY'
import hmac,hashlib,os
sec=os.environ.get('HMAC_SECRET','secret').encode()
body='{"feeds_fresh": true, "recent_forecasts": {"poolA": 0.62}}'
ts=os.environ.get('TS_OVERRIDE',str(int(__import__('time').time())))
msg=f"{ts}.{body}".encode()
print(hmac.new(sec,msg,hashlib.sha256).hexdigest())
PY
)
curl -sS -X POST "$BASE/policy/decide" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Timestamp: $ts" \
  -H "X-Signature: $sig" \
  -d '{"feeds_fresh": true, "recent_forecasts": {"poolA": 0.62}}'
```

## /policy/retrain_check  `GET`
Signals when to retrain models.

```bash
ts=$(date +%s)
sig=$(python - <<'PY'
import hmac,hashlib,os
sec=os.environ.get('HMAC_SECRET','secret').encode()
body=''
ts=os.environ.get('TS_OVERRIDE',str(int(__import__('time').time())))
msg=f"{ts}.{body}".encode()
print(hmac.new(sec,msg,hashlib.sha256).hexdigest())
PY
)
curl -sS -X GET "$BASE/policy/retrain_check" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Timestamp: $ts" \
  -H "X-Signature: $sig" \

```

## /policy/snapshot  `GET`
Returns analyst note and nightly report pointers.

```bash
ts=$(date +%s)
sig=$(python - <<'PY'
import hmac,hashlib,os
sec=os.environ.get('HMAC_SECRET','secret').encode()
body=''
ts=os.environ.get('TS_OVERRIDE',str(int(__import__('time').time())))
msg=f"{ts}.{body}".encode()
print(hmac.new(sec,msg,hashlib.sha256).hexdigest())
PY
)
curl -sS -X GET "$BASE/policy/snapshot" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Timestamp: $ts" \
  -H "X-Signature: $sig" \

```

## /ops/healthz  `GET`
Process heartbeat.

```bash
ts=$(date +%s)
sig=$(python - <<'PY'
import hmac,hashlib,os
sec=os.environ.get('HMAC_SECRET','secret').encode()
body=''
ts=os.environ.get('TS_OVERRIDE',str(int(__import__('time').time())))
msg=f"{ts}.{body}".encode()
print(hmac.new(sec,msg,hashlib.sha256).hexdigest())
PY
)
curl -sS -X GET "$BASE/ops/healthz" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Timestamp: $ts" \
  -H "X-Signature: $sig" \

```

## /ops/readyz  `GET`
Dependency readiness (RPC, CSV write, models on disk).

```bash
ts=$(date +%s)
sig=$(python - <<'PY'
import hmac,hashlib,os
sec=os.environ.get('HMAC_SECRET','secret').encode()
body=''
ts=os.environ.get('TS_OVERRIDE',str(int(__import__('time').time())))
msg=f"{ts}.{body}".encode()
print(hmac.new(sec,msg,hashlib.sha256).hexdigest())
PY
)
curl -sS -X GET "$BASE/ops/readyz" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Timestamp: $ts" \
  -H "X-Signature: $sig" \

```