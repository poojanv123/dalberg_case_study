# Data Dictionary (Clear Trends dataset)
This dictionary provides definitions for every column/metric in each file. All data is synthetic.

## district_profile.csv
| Column / Metric | Type | Unit | Definition | Typical range | Notes |
|---|---|---|---|---|---|
| district_id | string | n/a | Canonical district identifier (D001..D120). |  |  |
| district_code | string | n/a | Primary join key used in other files (format D-001). | D-001 to D-120 | Minor formatting issues may appear in project_history.csv. |
| district_name | string | n/a | Human-readable name. |  |  |
| country | string | n/a | Synthetic country name. |  |  |
| region | string | n/a | Synthetic region name (North/South/East/West/Central). |  |  |
| lat | float | degrees | Approximate latitude (synthetic). |  |  |
| lon | float | degrees | Approximate longitude (synthetic). |  |  |
| population | int | people | Estimated district population. | ~50k–~1.8M |  |
| poverty_index | float | 0–1 | Higher means higher poverty. | 0.15–0.90 | Used for equity constraint (top quintile). |
| urbanization_rate | float | 0–1 | Share of population living in urban areas. | 0.05–0.95 |  |
| baseline_infrastructure_index | float | ~0–1.4 | Composite infrastructure readiness (higher is better). | 0.15–1.35 |  |
| health_access_index | float | 0–1 | Proxy for access to health services (higher is better). | 0.05–0.95 | Some missing values. |
| mobile_penetration | float | 0–1 | Proxy for mobile access (higher is better). | 0.10–0.98 | Some missing values. |
| grid_reliability_index | float | 0–1 | Proxy for reliability of electricity supply. | 0.05–0.95 |  |
| historical_flood_loss_usd | int | USD | Estimated annualized historical flood-related losses (synthetic). | ~1e7–1e10 | One outlier exists. |
| historical_heat_loss_usd | int | USD | Estimated annualized historical heat-related losses (synthetic). | ~1e7–1e10 |  |
| past_programs_count | int | count | Number of prior resilience programs recorded (synthetic). | 0–10 |  |

## hazard_timeseries.csv
| Column / Metric | Type | Unit | Definition | Typical range | Notes |
|---|---|---|---|---|---|
| district_code | string | n/a | Join key to district_profile.district_code. |  |  |
| month | string (YYYY-MM) | month | Monthly timestamp. |  |  |
| flood_index | float | 0–100 | Relative flood hazard intensity index (higher = worse). | 0–100 | Seasonal; mild upward trend in flood-prone districts. |
| heat_index | float | 0–100 | Relative heat hazard intensity index (higher = worse). | 0–100 | Seasonal; clear upward trend from 2021→2024. |
| rainfall_anomaly_z | float | z-score | Standardized rainfall anomaly for the month (higher = wetter). | -3–3 | Correlated with flood_index. |
| temp_anomaly_z | float | z-score | Standardized temperature anomaly for the month (higher = hotter). | -3–3 | Correlated with heat_index. |
| ndvi | float | 0–1 | Vegetation greenness proxy (lower = drier/less green). | 0.05–0.85 | Negatively correlated with heat/flood stress. |
| flood_events_count | int | count/month | Number of flood events recorded this month (synthetic). | 0–10 | Increases with flood_index. |
| heatwave_days | int | days/month | Number of heatwave days recorded this month (synthetic). | 0–31 | Increases with heat_index. |

## interventions_catalog.csv
| Column / Metric | Type | Unit | Definition | Typical range | Notes |
|---|---|---|---|---|---|
| intervention_id | string | n/a | Intervention identifier (I01..I18). |  |  |
| intervention_name | string | n/a | Intervention name with tier (Basic/Standard/Enhanced). |  |  |
| hazard_focus | string | {flood,heat,both} | Primary hazard focus for the intervention. |  |  |
| unit_cost_usd | int | USD | Typical unit cost for one district-scale deployment. | 180k–1.5M |  |
| expected_risk_reduction_mean | float | fraction | Expected mean fractional reduction in risk. | 0.05–0.35 |  |
| expected_risk_reduction_low | float | fraction | Low bound on expected fractional reduction. | 0.03–0.30 |  |
| expected_risk_reduction_high | float | fraction | High bound on expected fractional reduction. | 0.08–0.45 |  |
| implementation_months | int | months | Typical duration to implement in a district. | 3–22 |  |
| complexity_1_5 | int | 1–5 | Implementation complexity (higher is harder). | 1–5 |  |
| prerequisites | string | n/a | Most common prerequisite or gating requirement. |  |  |

## project_history.csv
| Column / Metric | Type | Unit | Definition | Typical range | Notes |
|---|---|---|---|---|---|
| project_id | string | n/a | Project identifier (P00001..). |  |  |
| district_code | string | n/a | District code where the project was implemented. | D-001 style | Small fraction uses alternate formatting. |
| intervention_id | string | n/a | Intervention id (join to interventions_catalog). |  |  |
| partner | string | n/a | Implementing partner organization name. |  |  |
| start_date | date (YYYY-MM-DD) | date | Project start date. |  |  |
| planned_end_date | date (YYYY-MM-DD) | date | Planned end date based on typical duration. |  |  |
| actual_end_date | date (YYYY-MM-DD) | date | Observed end date. |  |  |
| cost_usd | int | USD | Total spend for the project (synthetic). | 100k–3M |  |
| baseline_risk_score | float | 0–100 | Baseline district risk score used for internal planning (synthetic). | 0–100 |  |
| total_delay_days | int | days | Total schedule delay beyond plan (synthetic). | 0–300 |  |
| achieved_outcome_score | float | 0–100 | Overall outcome score based on internal rubric (synthetic). | 0–100 |  |
| realized_risk_reduction | float | fraction | Estimated realized fractional risk reduction (synthetic). | 0–0.50 |  |
| delivered_on_time | int | 0/1 | Label: 1 if total_delay_days ≤ 45; else 0. | 0 or 1 | Recommended feasibility label. |
| achieved_min_outcome | int | 0/1 | Alternate label: 1 if achieved_outcome_score ≥ 60; else 0. | 0 or 1 |  |
| community_engagement_score | float | 0–100 | Proxy for local engagement quality (synthetic). | 0–100 |  |
| post_project_audit_score | float | 0–100 | Post-hoc audit score (available only after completion). | 0–100 | Leakage risk: do not use as predictor. |
| post_project_satisfaction_score | float | 0–100 | Post-hoc beneficiary satisfaction score. | 0–100 | Leakage risk: do not use as predictor. |

## field_notes/
| Column / Metric | Type | Unit | Definition | Typical range | Notes |
|---|---|---|---|---|---|
| Header fields | text | n/a | Each note includes date, country/region, district id and code, partner, intervention. |  |  |
| Severity rating | int | 1–5 | Operational severity recorded by field team (higher = more severe). | 1–5 |  |
| Estimated schedule impact | int | weeks | Approximate schedule impact referenced in the note. | 0–12 |  |
| Mitigation actions proposed | text | n/a | Suggested mitigation steps linked to the primary risk category. |  |  |
| Context note (trend cue) | text | n/a | Some notes include an explicit line indicating worsening heatwaves/rainfall disruptions. |  |  |

## gold_questions.json
| Column / Metric | Type | Unit | Definition | Typical range | Notes |
|---|---|---|---|---|---|
| id | string | n/a | Question identifier. |  |  |
| question | string | n/a | Question text (answerable from field notes). |  |  |
| expected_answer | string | n/a | Expected answer text. |  |  |
| evidence | array | n/a | File and matching hint to locate evidence. |  |  |

## risk_labels_seed.csv
| Column / Metric | Type | Unit | Definition | Typical range | Notes |
|---|---|---|---|---|---|
| snippet_id | string | n/a | Snippet identifier. |  |  |
| source_file | string | n/a | Field note filename where snippet appears. |  |  |
| district_id | string | n/a | Canonical district id. |  |  |
| text | string | n/a | Snippet text. |  |  |
| risk_category | string | n/a | Single risk label from the taxonomy (optional stretch). |  |  |
