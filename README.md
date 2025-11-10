<p align="center">
  <img src="OpenFPL.png" alt="OpenFPL" width="150">
</p>

<h1 align="center">OpenFPL</h1>
<p align="center"><b>The accurate openly available forecasting method for Fantasy Premier League</b></p>

---

<p align="center"><em>Want to support this initiative?</em></p>

<p align="center">
  <a href="https://www.paypal.com/donate/?hosted_button_id=EKNGVA5RU2B96" target="_blank">
    <img src="https://www.paypalobjects.com/en_US/i/btn/btn_donate_SM.gif" alt="Donate with PayPal" height="24">
  </a>
</p>

---

## Get started

### 1. Plug

With [Python](https://www.python.org/downloads/) preinstalled, run: ```pip install -r plug.txt```

### 2. Play

Open and run *play.ipynb* for OpenFPL predictions on sample data

## Custom data

To use OpenFPL on custom data, you need to construct samples based on data from FPL and Understat APIs (see *data/samples.csv* and [paper](https://arxiv.org/abs/2508.09992) for inspiration):

- [FPL API](https://fantasy.premierleague.com/api/bootstrap-static/)
- [Understat API](https://understat.com/league/EPL/)

Historical FPL and Understat data can be accessed by help of [FPL Historical Dataset](https://github.com/vaastav/Fantasy-Premier-League)

## XI Optimizer Web App

The repository now includes a fully interactive FastAPI web application that connects directly to the Fantasy Premier League API, ingests the latest OpenFPL projections, and builds an optimised squad for any manager ID.

### Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then visit [http://localhost:8000](http://localhost:8000) to:

- Enter an FPL manager ID, season, and gameweek.
- Optimise the full 15-player squad and starting XI with budget, position, and club constraints.
- Review chip recommendations (Triple Captain, Bench Boost, Free Hit) based on projected gains.
- Compare the optimised team with your current squad and persist the results for later review.

## Head-to-head evaluation with state-of-the-art commercial method

| Method | RMSE<sub>Zeros*</sub> | RMSE<sub>Blanks*</sub> | RMSE<sub>Tickers*</sub> | RMSE<sub>Haulers*</sub> |
| :--  | --- | --- | --- | --- |
| OpenFPL | 0.818 | 1.291 | <b>1.517</b> | <b>5.142</b> |
| [FPL Review Massive Data Model](https://fplreview.com/) | <b>0.689</b> | <b>1.189</b> | 1.594 | 5.172 | 

<sup>*</sup> *Zeros*: Non-playing and 0 FPL points, *Blanks*: ≤ 2 FPL points, *Tickers*: 3 or 4 FPL points, *Haulers*: ≥ 5 FPL points

## Resources

- Scientific paper - [OpenFPL: An open-source forecasting method rivaling state-of-the-art Fantasy Premier League services](https://arxiv.org/abs/2508.09992)
- Model search framework - [*K*-Best Search](https://github.com/daniegr/KBestSearch)

## Citation

Should you find the work helpful in your research, please cite the following:
```
@article{groos2025openfpl,
  title={OpenFPL: An open-source forecasting method rivaling state-of-the-art Fantasy Premier League services},
  author={Groos, Daniel},
  year={2025},
  publisher={arXiv}
}
```
