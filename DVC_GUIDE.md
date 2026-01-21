# DVC Workflow (Collaborator Guide)

This project uses DVC to version data and model artifacts.

## One-time setup

```powershell
# Install DVC
pip install dvc

# Initialize (already done in this repo)
dvc init

# Configure remote (DagsHub)
dvc remote add -d dagshub https://dagshub.com/Theepankumargandhi/multi-stage-two-tower-recommender.dvc
dvc remote modify --local dagshub user YourUserName
dvc remote modify --local dagshub password YourToken
dvc remote modify --local dagshub auth basic
```

## Pull data and models

```powershell
dvc pull data\raw.dvc checkpoints.dvc
```

## Track new data or models

```powershell
dvc add data/raw
dvc add checkpoints
git add data\raw.dvc data\.gitignore checkpoints.dvc .gitignore
```

## Push data to DagsHub

```powershell
dvc push
```

## Reproduce pipeline (optional)

```powershell
dvc repro
```

## Common notes

- `data/processed` and `data/features` are placeholders; populate them when you implement preprocessing/feature stages.
- If `dvc pull` complains about missing targets from `dvc.yaml`, use explicit pulls:
  - `dvc pull data\raw.dvc checkpoints.dvc`

---

Tip: Never commit tokens; always use `--local` DVC config for auth.
