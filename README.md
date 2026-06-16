# CalMU HubSpot Sales Performance Dashboard

This project is a live Streamlit dashboard for HubSpot sales performance. It connects to HubSpot through the API, profiles available fields, builds a safe field mapping, caches raw and processed data locally, and shows sales, paid lead, revenue, lead temperature, attribution, and data quality reporting.

The app does not use manual CSV uploads.

## 1. What This App Does

- Pulls contacts, deals, owners, and contact-to-deal associations from HubSpot.
- Profiles HubSpot contact and deal fields before using them.
- Maps dashboard fields only to populated HubSpot fields.
- Uses Contact Owner as the primary salesman field.
- Uses Deal Owner only when Contact Owner is missing.
- Counts each deal's revenue once, even when a deal has multiple contacts.
- Estimates full program tuition when a populated contact tuition field is available, then spreads that value over the normal degree duration.
- Uses editable CalMU published tuition defaults from `config/degree_tuition.json` for potential revenue and fallback estimates.
- Classifies paid leads and source groups.
- Scores leads as Hot, Warm, Cold, or Dead.
- Shows missing fields and data quality warnings instead of crashing.
- Uses local cached data when HubSpot refresh fails.

## 2. Create a HubSpot Private App Token

In HubSpot, create a private app and copy its access token.

Typical steps:

1. Go to HubSpot settings.
2. Open Integrations, then Private Apps.
3. Create a private app.
4. Add read scopes for contacts, deals, owners, CRM properties, and associations.
5. Copy the private app access token.

Suggested scopes include contact read, deal read, owner read, and CRM schema/property read access. HubSpot scope names can vary by account and HubSpot UI version, so use the closest matching read-only scopes available in your portal.

## 3. Set `HUBSPOT_ACCESS_TOKEN`

Create a local `.env` file:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
HUBSPOT_ACCESS_TOKEN=your_token_here
```

You can also set it in your shell:

```bash
export HUBSPOT_ACCESS_TOKEN=your_token_here
```

Never commit `.env`. The token is never hardcoded or printed by this app.

## 4. Install Requirements

```bash
pip install -r requirements.txt
```

## 5. Run the Field Profiler

```bash
python field_profiler.py
```

This creates:

- `data/audit/hubspot_field_inventory.csv`
- `data/audit/dashboard_field_mapping.json`
- `data/audit/dashboard_removed_fields.csv`

## 6. Run the First Sync

```bash
python sync.py
```

If the field mapping does not exist, `sync.py` runs the field profiler first.

Raw cache is saved to `data/raw`. Processed dashboard tables are saved to `data/processed`.

## 7. Run the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard reads cached processed data. Use the Refresh HubSpot Data button to pull fresh HubSpot data.

## 8. Deploy From GitHub

This app can be deployed from GitHub to a Streamlit-compatible host such as Streamlit Community Cloud.

Recommended Streamlit Cloud settings:

- Repository: `poojasjainmba-cell/calmu`
- Branch: `main`
- Main file path: `dashboard.py`
- Python dependencies: `requirements.txt`

Add this secret in the hosting platform's secrets manager, not in Git:

```toml
HUBSPOT_ACCESS_TOKEN = "your_hubspot_private_app_token"
```

Optional detailed activity history can be enabled with:

```toml
HUBSPOT_SYNC_ACTIVITIES = "true"
```

Leave `HUBSPOT_SYNC_ACTIVITIES` unset for Streamlit Cloud's first sync. The app will use HubSpot contact summary activity fields, which keeps the initial refresh fast enough for hosted deployment.

Streamlit Cloud refreshes the most recent 10,000 contacts by default so the first hosted sync can complete within HubSpot hosted-search limits. To sync every HubSpot contact, add this secret and refresh again:

```toml
HUBSPOT_MAX_CONTACTS = "0"
```

Runtime HubSpot cache files under `data/` are intentionally ignored by Git. Run `python field_profiler.py` and `python sync.py` locally or from the deployment environment after adding the token.

## 9. Dashboard Pages

- Executive Overview: top-level lead, enrollment, revenue, source, and degree-level performance.
- Daily Action Center: a practical follow-up queue for hot leads, stale paid leads, old open deals, and reviveable dead leads.
- Degree Revenue: total, annualized, 6-month, 12-month, and 24-month tuition estimates by degree level and program.
- Paid Lead Vendor Performance: vendor search, vendor normalization, paid lead leakage, enrollment, revenue, CPL, and ROI where cost data exists.
- Salesmen / Contact Owner Performance: owner scorecards using Contact Owner first and Deal Owner only as fallback.
- Student Journey: searchable per-student profile, timeline, activity summary, journey metrics, and recommended next action.
- Cohort Analysis: cohort summary, characteristics, what it took to enroll, and side-by-side cohort comparison.
- Pipeline Health: clear status, bottleneck, enrollment path, and cleanup tables replacing the old funnel and stage aging view.
- Hot vs Dead Leads: temperature mix, stuck lead queues, reviveable dead leads, and archive/remove candidates.
- Data Quality and Field Mapping: field mapping, removed fields, low-population fields, and attribution gaps.
- Definitions: glossary of calculated metrics plus the editable tuition configuration.

Friendly warnings appear when HubSpot fields are missing or empty. The dashboard is read-only and does not write back to HubSpot.

## 10. Salesmen / Contact Owner Page

The Salesmen / Contact Owner page attributes performance using this rule:

1. Use Contact Owner as `salesman_id`.
2. If Contact Owner is missing, use Deal Owner.
3. If both are missing, mark `attribution_type` as `missing_owner`.

The page includes lead counts, paid/organic counts, Hot/Warm/Cold/Dead leads, won deals, total estimated program revenue, annualized revenue, 6/12/24-month revenue, close rate, paid close rate, days to close, and revenue per lead.

## 11. Program Duration Revenue

Program duration defaults live in `config/degree_tuition.json`:

- Certificate: 6 months
- Associate: 24 months
- Bachelor: 48 months
- Master: 24 months
- Doctoral: 48 months, pending CalMU policy confirmation

`total_program_revenue` is the full estimated tuition value for a won/countable student record. When HubSpot has a populated contact tuition field such as `standard_tuition_total`, the dashboard uses that value. If not, it falls back to countable won deal revenue, then to the editable CalMU published tuition config. Annualized and 6/12/24-month revenue spread the full value over the configured program duration.

`potential_program_revenue` uses the published tuition config for matched programs regardless of won/lost deal status. The dashboard does not scrape CalMU on load. To manually update the local config from the CalMU tuition page, run:

```bash
python refresh_tuition.py
```

## 12. Processed Tables

The sync creates these processed tables under `data/processed`:

- `contacts_clean`
- `deals_clean`
- `lead_deal_fact`
- `student_journey_fact`
- `cohort_fact`
- `vendor_fact`
- `salesman_revenue_fact`
- `activity_events`

Detailed activity history is optional and controlled by `HUBSPOT_SYNC_ACTIVITIES=true`. If it is disabled or the token cannot read calls, emails, meetings, notes, and tasks, the dashboard falls back to summary activity fields and shows a friendly note.

## 13. Vendor Cost Data

Vendor cost and spend data is optional. Add rows to `config/vendor_costs.csv` to calculate cost per lead and estimated ROI:

```csv
vendor,month,spend,notes
Atra,2026-01,0,replace with actual spend
```

If cost data is missing, Paid Lead Vendor Performance still shows leads, contacted leads, leakage, enrollment, and revenue metrics.

## 14. Hot vs Dead Lead Calculation

Lead score adds points for:

- New leads
- Paid leads
- MQL, SQL, qualified, or opportunity status
- Open deals
- Recent activity
- Scheduled next activity
- Email or phone availability

Lead score subtracts points for:

- Inactivity
- Older leads without open deals

Dead overrides include closed lost, disqualified, unqualified, not interested, bad fit, invalid, junk, duplicate, old inactive leads, and older never-contacted leads.

## 15. Troubleshoot Missing Fields

Run:

```bash
python field_profiler.py
```

Then review:

- `data/audit/hubspot_field_inventory.csv`
- `data/audit/dashboard_field_mapping.json`
- `data/audit/dashboard_removed_fields.csv`

If a dashboard feature is hidden, the mapped HubSpot field is probably missing or empty. The app does not guess fields without data.

## 16. Refresh HubSpot Data

From the dashboard, click Refresh HubSpot Data.

From the terminal, run:

```bash
python sync.py
```

If HubSpot fails, the dashboard keeps using the latest cached processed data when available.
