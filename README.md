\# ⚽ Global Football Talent Radar



ML-powered scouting system for identifying emerging football players across feeder leagues using historical transfer outcomes.



\## 🎯 Objective



The project analyzes historical player performance, market valuations and transfer outcomes to identify young players who historically resemble footballers that later moved into major European leagues.



The system also scores the latest available players and provides an interactive scouting dashboard.



\## 📊 Data



The project uses a public Transfermarkt-derived dataset containing:



\- Player profiles

\- Player appearances

\- Goals, assists and minutes played

\- Historical market valuations

\- Historical transfers

\- Clubs and competitions



Current scouting coverage includes six leagues with sufficient domestic-performance data:



\- Eredivisie

\- Belgian Pro League

\- Liga Portugal

\- Danish Superliga

\- Turkish Süper Lig

\- Greek Super League



Data cutoff: \*\*June 2026\*\*



\## 🧠 Methodology



```text

Historical player data

&#x20;       ↓

Player-season aggregation

&#x20;       ↓

Historical breakout labeling

&#x20;       ↓

Feature engineering

&#x20;       ↓

Walk-forward validation

&#x20;       ↓

Random Forest / Logistic Regression

&#x20;       ↓

Current player scoring

&#x20;       ↓

Emerging Talent Radar

