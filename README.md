# CalMU Enrollment Performance Dashboard

Production-ready Streamlit dashboard for California Miramar University enrollment, lead, UDR, source, and budget performance.

The app is read-only. It never edits HubSpot records, sends emails, deletes source data, overwrites historical snapshots, or hardcodes HubSpot tokens.

## Included App

- Main Streamlit app: `app.py`
- Uploaded baseline files: `user_files/`
- Sanitized deployable baseline files: `public_data/baseline/`
- Dashboard modules: `modules/`
- Streamlit config: `.streamlit/config.toml`
- Secrets example: `.streamlit/secrets.toml.example`

The prior `dashboard.py` app remains in the repo for continuity, but the deployment target for this build is `app.py`.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

For local HubSpot access, create either `.env` or `.streamlit/secrets.toml`:

```bash
HUBSPOT_ACCESS_TOKEN=your_read_only_private_app_token
```

## Uploaded Files

Place these files in `user_files/`:

- `01-2026Budget.xlsx`
- `01-PaidleadsJune11-1-.xlsx`
- `02-Summer-2-Enrollment-Update-Week-6-June-8-12-1-.eml`
- `03-Summer2tracker-1-.xlsx`
- `04-UDRConversionsJune11-1-.xlsx`

The dashboard loads all workbook sheets, recalculates raw lead/enrollment metrics where possible, and uses pivot sheets only as reconciliation references.

`user_files/` is intentionally ignored by Git because the source files can contain names, emails, phone numbers, and other private fields. For GitHub and Streamlit Cloud, use the sanitized `public_data/baseline/` files instead. They preserve dashboard-level calculations, retain vendor/source business labels, and remove or pseudonymize private person labels.

To rebuild the sanitized baseline locally after source files change:

```bash
python3 scripts/build_public_baseline.py
```

## HubSpot Setup

Create a HubSpot private app token and add it as:

```toml
HUBSPOT_ACCESS_TOKEN = "your_read_only_private_app_token"
```

Required scopes:

- `crm.objects.contacts.read`
- `crm.schemas.contacts.read`
- `crm.lists.read` only if list membership is later added

The app fetches contact schemas before fetching contacts. It only uses exact available property internal names and exposes missing configured properties on the Assumptions / Data Notes page.

Optional secret:

```toml
HUBSPOT_CONTACT_PROPERTIES = "exact_custom_property_1,exact_custom_property_2"
```

Use this only for confirmed HubSpot fields.

Optional Streamlit-only UDR display labels:

```toml
[udr_label_map]
"UDR 01" = "Admissions Owner 1"
"UDR 02" = "Admissions Owner 2"
```

Use this when the public baseline should keep UDR names out of GitHub, but the private Streamlit app should show the real UDR names.

## Streamlit Community Cloud Deployment

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app from the repository.
3. Set the main file path to `app.py`.
4. Add secrets from `.streamlit/secrets.toml.example`.
5. Start with `HUBSPOT_MAX_CONTACTS = "1000"` for a fast hosted smoke test, then raise it after the app is stable.

## Dashboard Sections

- Executive Overview
- Enrollment Tracker
- Source Performance
- UDR Performance
- Program Mix
- Budget Performance
- Lead Status & Lifecycle Funnel
- Trends
- QA Checks
- Assumptions / Data Notes
- Raw Audit Data

## Calculation Notes

- Degree equals Program.
- Contact Owner is UDR.
- Enrollment tracker is the source of truth for actual enrollments, revenue, days to enroll, program mix, modality, student type, payment/funding, and enrollment source.
- HubSpot is the source of truth for live leads, lifecycle stage, lead status, contact owner, create date, last activity date, source/list fields, and CRM funnel metrics when the token is configured.
- Applicant equals lifecycle stage `Applicant`.
- CRM Enrolled equals lifecycle stage `Enrolled`.
- Actual Enrollment equals a tracker row with a nonblank student and matching selected term.
- Bad Lead equals lifecycle stage `Not a Lead` or lead status `Dead Lead`, `Do Not Contact`, `Duplicate Lead`, or `App Submitted - Unqualified`.
- Lead-to-contact uses explicit contact flags when present. Otherwise it uses Last Activity Date or progressed lead statuses.
- Revenue is summed from tracker `Rev` / revenue.
- The uploaded budget workbook is treated as enrollment goal/allocation data because no confirmed paid-media spend ledger fields are present.

## Privacy And Git Hygiene

- `.env`, `.streamlit/secrets.toml`, Streamlit credentials, generated exports, and PII export folders are ignored.
- `user_files/` source workbooks/emails are ignored and must stay private.
- `public_data/baseline/` is safe for GitHub because names, emails, phone numbers, raw Record IDs, notes, and UDR/contact-owner labels are removed or pseudonymized.
- Vendor/source/list labels are retained as business labels.
- Real UDR display names can be restored in Streamlit only through `[udr_label_map]` secrets.
- Executive pages do not show names, emails, phone numbers, or tracker notes.
- Raw rows are available only on the clearly separated Raw Audit Data page. Set `RAW_AUDIT_PASSWORD` to require a password.

## Validation

Run:

```bash
pytest
streamlit run app.py
```

Confirmed dashboard checks:

- All uploaded files load from `user_files/`.
- Missing HubSpot token state falls back to static uploaded baseline.
- Paid lead workbook raw totals reconcile against pivot references.
- UDR conversion workbook raw totals reconcile against pivot references.
- Enrollment tracker actual enrollment count comes from `Enrollments`.
- Budget total comes from the uploaded budget workbook summary/allocation fields.

Known limitation: OCR of embedded email images runs only if local OCR dependencies are available. If OCR is unavailable, the app reports that no practical OCR output was captured and still displays parsed plain text / HTML context.
