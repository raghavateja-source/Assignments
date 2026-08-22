from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
TARGET_COLUMN = 'grid_failure_flag'
LEAKAGE_COLUMNS = ['estimated_revenue_loss', 'regulatory_penalty_cost', 'avg_outage_duration_minutes']
IDENTIFIER_COLUMNS = ['asset_id', 'legacy_asset_code', 'monitoring_batch_id', 'administrative_reference', 'grid_cluster_id']
BUSINESS_IMPACT_COLUMNS = ['customers_served']
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
