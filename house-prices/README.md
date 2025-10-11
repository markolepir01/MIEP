# House Prices – Advanced Regression Techniques

Ovaj repozitorijum sadrži **eksperimente i inference pipeline** za Kaggle takmičenje  
[House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

Cilj projekta bio je da se predvidi **`SalePrice`** kuća na osnovu različitih atributa (npr. veličina kuće, kvalitet materijala, broj soba itd.), primenom više regresionih modela i poređenjem njihovih performansi.

---

## Struktura projekta

```
house-prices/
	data/                    # train.csv i test.csv
	models/                  # sačuvani modeli (*.pkl)
	submissions/             # Kaggle submission fajlovi (*.csv)
	src/
		config.py        # putanje i globalne konstante
		preprocess.py    # imputacija, skaliranje, one-hot encoding
		models.py        # svi modeli (Ridge, Lasso, ElasticNet, RF, XGB, LGBM, CatBoost)
		utils.py         # RMSE log, transformacije
		train.py         # treniranje modela (eksperimenti)
		infer.py         # inference pipeline za generisanje predikcija
	requirements.txt         # biblioteke potrebne za pokretanje
```

---

## Pokretanje projekta

### Instalacija zavisnosti

```bash
pip install -r requirements.txt
```

### Treniranje modela (eksperimenti)

Pokretanje treninga za željeni model (npr. CatBoost):

```bash
python src/train.py --model catboost
```

Dostupni modeli:
`ridge`, `lasso`, `elastic`, `rf`, `xgb`, `lgbm`, `catboost`

Model i preprocesor se automatski čuvaju u folderu `models/`.

### Generisanje predikcija (inference pipeline)

Nakon treniranja, generisanje Kaggle submission fajla:

```bash
python src/infer.py --model catboost
```

Rezultat će biti sačuvan u:
```
submissions/submission_catboost.csv
```

---

## Eksperimenti i rezultati

U okviru projekta testirano je sedam različitih modela.  
Tabela ispod prikazuje prosečne vrednosti **RMSE (log)** dobijene kroz 5-Fold Cross Validation.

| Model | Mean CV RMSE(log) |
|--------|-------------------|
| Ridge | 0.14680 |
| Lasso | 0.14428 |
| ElasticNet | 0.14285 |
| Random Forest | 0.14483 |
| XGBoost | 0.12730 |
| LightGBM | 0.13357 |
| **CatBoost** | **0.12447** |

**CatBoost** se pokazao kao najbolji model sa najmanjom greškom.

---


## 📄 Zaključak

- Linearni modeli (Ridge, Lasso, ElasticNet) dali su osnovnu tačnost i poslužili kao baseline.
- Ansambl i boosting modeli (Random Forest, XGBoost, LightGBM, CatBoost) značajno su poboljšali rezultate.
- **CatBoost** je postigao najbolju tačnost uz minimalan tuning hiperparametara i jednostavnu implementaciju.
- Rezultati su potvrđeni na Kaggle leaderboardu.

---