# Model Card

**Model:** Random Forest
**Purpose:** Asset-level grid failure risk scoring
**Target:** grid_failure_flag
**Features:** Pre-failure operational, condition, maintenance and weather variables available in the supplied dataset
**Excluded:** Post-event/possibly post-event business impact and outage fields; identifier-like fields
**Threshold:** Stored in `models/decision_threshold.json`
**Limitations:** No temporal validation; requires real-world calibration before production
