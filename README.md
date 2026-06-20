# ChartVision

Candlestick **chart pattern recognition** with a CNN, plus **Grad-CAM** heatmaps that
show where on the chart the network is looking.

Real OHLC data (yfinance) is labelled with geometric rules, rendered into minimal
candlestick images, and classified into one of five pattern classes. Two models are
compared: a small from-scratch CNN and a ResNet18 with transfer learning.

## Classes

| Class | Family | Bias |
|---|---|---|
| `head_and_shoulders` | reversal | bearish |
| `double_top` | reversal | bearish |
| `double_bottom` | reversal | bullish |
| `bull_flag` | continuation | bullish |
| `no_pattern` | negative | — |

## Repository layout

```
code/      source: data pipeline, models, training, evaluation, Grad-CAM, tests
report/    final report (PDF)
slides/    final presentation (PDF)
```

### `code/`

```
src/
  data/       fetch_ohlc, rule_labelers, render_charts, build_dataset, dataset
  models/     baseline_cnn, resnet18
  training/   train, evaluate, compare_models, gradcam
  inference/  prediction helpers
  utils/      config, seeding
configs/      default.yaml — single source of truth (paths, tickers, seeds, hyper-params)
tests/        pytest unit tests for the rule labelers
make_*.py     figure / slide-asset generation scripts
```

Raw/processed data, model checkpoints, and generated figures are not committed — they are
reproduced by the pipeline.

## Setup

```bash
cd code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision
```

## Reproduce

```bash
python -m src.data.build_dataset
python -m src.training.train     --model resnet18
python -m src.training.evaluate  --checkpoint checkpoints/resnet18_best.pt
```

## Tests

```bash
pytest
```

---

Computer Vision Short Project — Master in Innovation and Research in Informatics, FIB UPC.
