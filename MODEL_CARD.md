# Equipment telemetry model card

## Purpose and provenance

Independent portfolio experiment built on November 14, 2026. Predict whether a simulated refrigeration unit will have a fault event in the next 24 hours. All records are generated locally; the timestamps starting January 1, 2025 are simulation coordinates, not historical store observations.

The generator gives each unit a random temperature offset and efficiency, with stochastic latent wear. Noisy temperature, vibration, power and duty readings reflect wear and ambient/door load. Crossing the wear threshold produces a fault, after which simulated maintenance resets wear. The fault row and three recovery readings are excluded from modeling. Latent wear, day, asset identity, fault status and time-to-fault are not predictors.

This design deliberately creates a learnable relationship. A strong score validates the software and experimental setup against that relationship, not a theory of actual equipment failure.

## Inputs and target

Six current features: cabinet temperature in Celsius, ambient temperature in Celsius, vibration RMS in arbitrary simulation units, power in kW, compressor duty fraction and door-open minutes. Seven derived features: trailing temperature mean and standard deviation, vibration/power/duty means, and temperature/vibration changes from four readings earlier.

The fields named `*_48h` aggregate **eight six-hourly readings, including the current reading**. Their point timestamps span 42 elapsed hours; the name denotes eight six-hour sampling intervals. No interpolation or future fill is used. The initial incomplete windows are discarded.

The binary target is any fault in the **next four samples**, excluding the current sample. Rows without a complete future horizon are discarded. Multiple positive rows can refer to the same fault event.

The API requires these engineered features. It does not independently verify timestamp cadence, measurement units, sensor ranges or equipment identity. Scores on arbitrary input remain mathematically defined but have no established operational meaning.

## Experimental protocol

| Split | Assets | Day interval | Rows | Positive rows |
|---|---:|---|---:|---:|
| Fit | 36 | 1.75–64.75 | 8,616 | 500 |
| Calibration | Same 36 | 72–85.75 | 1,914 | 107 |
| Later test, known units | Same 36 | 93–118.75 | 3,536 | 209 |
| Later test, unseen units | Other 12 | 93–118.75 | 1,163 | 81 |

Current and past readings create features; future fault flags are used only for labels. Scaling is fit on the fitting split. Seven-plus-day gaps prevent label windows from crossing evaluation partitions. The unseen units are entirely excluded from model fitting and threshold/model selection, while their past readings may form test-time features.

Two fixed candidates:

- `StandardScaler` plus logistic regression, `C=1`, balanced class weights, 1,000 iteration cap.
- Random forest, 180 trees, maximum depth 7, minimum leaf size 12, balanced class weights, one worker.

Both use seed 734. Candidate settings were fixed before the benchmark; there was no model hyperparameter search. For each candidate, thresholds from 0.05 to 0.95 in 0.01 increments maximize F2 on calibration, with higher thresholds breaking ties. The candidate with the highest calibration average precision is selected. The selected model is not refit after selection.

Calibration average precision was 0.7878 for logistic regression and 0.7284 for random forest. The selected logistic threshold was 0.81. Both test cohorts were evaluated after selection; their results did not select model settings or thresholds.

## Test performance and failures

| Model/cohort | Average precision | Precision | Recall | False positives | False negatives |
|---|---:|---:|---:|---:|---:|
| Logistic / known | 0.8107 | 0.6824 | 0.8325 | 81 | 35 |
| Logistic / unseen | 0.8230 | 0.6102 | 0.8889 | 46 | 9 |
| Forest / known | 0.7465 | 0.6169 | 0.8708 | 113 | 27 |
| Forest / unseen | 0.7833 | 0.5769 | 0.9259 | 55 | 6 |

Each candidate uses its own calibration-selected threshold (logistic 0.81; forest 0.66). Positive prevalence was 5.91% on known units and 6.96% on unseen units. Average precision summarizes ranking; it is not accuracy. The saved logistic model's Brier scores were 0.0354 and 0.0521 respectively, but no probability-calibration procedure or prospective calibration study was performed.

Results are on one seed and one fixed temporal split. They do not provide uncertainty across independent equipment populations. Overlapping feature/label windows and repeated units create correlated readings. No event-level deduplication, warning lead-time analysis or intervention effectiveness is claimed.

## Runtime, inference and limitations

The report records Python 3.10.11, NumPy 1.26.4, pandas 2.3.3, SciPy 1.14.1 and scikit-learn 1.7.2. The joblib bundle includes the selected fitted pipeline, exact feature order, threshold, seed and schema version. The CLI and FastAPI use that same bundle. Tests compare the API result to a saved test prediction and check the reported counts against saved scores.

The local API and saved artifacts are implemented. Production deployment, Azure execution, real-time ingestion, authentication, CI/CD, drift/retraining automation and physical equipment integration are not. Synthetic units follow the same generator family; unseen assets do not represent unknown equipment vendors, new sensors or real distribution shifts. This is not a maintenance or safety decision system.
