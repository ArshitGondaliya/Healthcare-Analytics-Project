# Healthcare-Analytics-Project

## Care Transition Efficiency & Placement Outcome Analytics

This project implements a Streamlit analytics dashboard for the `HHS_Unaccompanied_Alien_Children_Program.csv` dataset. It reframes the data from capacity monitoring to process efficiency, backlog detection, and placement outcome evaluation for the UAC care pipeline.

### What is included

- `care_transition_dashboard.py` — Streamlit application for pipeline analytics
- `requirements.txt` — Python dependencies for the dashboard
- `HHS_Unaccompanied_Alien_Children_Program.csv` — dataset used for analysis

### Dashboard features

- Care pipeline flow visualization spanning CBP custody, HHS care, and discharge outcomes
- Transfer efficiency and discharge effectiveness metrics
- Backlog detection and active care load bottleneck charts
- Outcome trend analysis for monthly volumes and weekday patterns
- Predictive regression model for future HHS discharge volume
- Threshold alerting for low efficiency and high HHS care load

### How to run

1. Open a terminal in this folder.
2. Create and activate your Python environment.
3. Install dependencies with:

```bash
pip install -r requirements.txt
```

4. Run the dashboard:

```bash
streamlit run care_transition_dashboard.py
```

### Notes

- The app expects `HHS_Unaccompanied_Alien_Children_Program.csv` to be in the same folder.
- The model uses historical transfer and care counts to estimate discharge performance.
- This project satisfies the PRD by focusing on transition efficiency, discharge outcomes, backlog analysis, and outcome stability.
