import json
import os
from glob import glob
from tqdm import tqdm
import numpy as np
from functools import partial
from scipy.special import logsumexp
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cmdstanpy import CmdStanModel

import bayesblend as bb

FIGURE_PATH = "figures"
DATA = "data/forecast_data.json"
SEED = 1234

TEMPLATE = "none"

MIN_LFO_TRAIN = 5
MAX_LFO_TRAIN = 8
M = 1

STAN_CONFIG = {
    "iter_warmup": 1000,
    "iter_sampling": 1000,
    "seed": SEED,
}

MODELS = {
    os.path.basename(path).replace(".stan", ""): CmdStanModel(stan_file=path)
    for path 
    in glob("stan/forecast/*.stan")
}

BLEND_MODELS = {
    "pseudo-bma": partial(bb.PseudoBma, bootstrap=False),
    "pseudo-bma+": partial(bb.PseudoBma, seed=SEED),
    "stacking-mle": bb.MleStacking,
    "stacking-bayes": partial(bb.BayesStacking, seed=SEED),
    "stacking-hier-bayes": partial(bb.HierarchicalBayesStacking, seed=SEED),
}


COLORS = [
    "#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51",
]

TITLE_LOOKUP = {
    "constant": "Constant",
    "linear": "Linear",
    "linear-hk": "Linear-Hk",
    "ar1": "AR(1)",
    "ssm": "SSM",
    "gp": "GP",
    "pseudo-bma": "Pseudo BMA",
    "pseudo-bma+": "Pseudo BMA+",
    "stacking-mle": "Stacking (MLE)",
    "stacking-bayes": "Stacking (Bayes)",
    "stacking-hier-bayes": "Hierarchical stacking"
}

COLOR_LOOKUP = {
    k: color
    for k, color
    in zip(
        list(TITLE_LOOKUP.keys())[len(MODELS):],
        COLORS,
    )
}


def load_forecast_data():
    return json.load(open(DATA, "r"))

def plot_programs(data, show: bool = False):
    rng = np.random.default_rng(0)
    indices = sorted(rng.choice(range(data["N"]), size=10, replace=False))
    R = 2
    C = 5
    fig = make_subplots(R, C, subplot_titles=list(map(str, indices)))
    fig.update_layout(template=TEMPLATE)
    fig.update_xaxes(title="Accident year", row=2)
    fig.update_yaxes(title="Loss ratio", col=1)
    fig.update_yaxes(range=[0, 2])
    fig.update_xaxes(tickmode="array", tickvals=np.arange(1, 11, 1))

    rc = [(r, c) for r in range(R) for c in range(C)]
    for i, (r, c) in enumerate(rc):
        showlegend = not i
        idx = indices[i]
        fig.add_trace(
            go.Scatter(
                mode="markers+lines",
                x=data["AY_train"][idx],
                y=data["loss_ratio_train"][idx],
                marker_color="black",
                marker_symbol="circle-open",
                marker_size=10,
                name="Train",
                showlegend=showlegend,
            ),
            row=r+1, col=c+1,
        )
        fig.add_trace(
            go.Scatter(
                mode="markers+lines",
                x=data["AY_train"][idx][MIN_LFO_TRAIN:],
                y=data["loss_ratio_train"][idx][MIN_LFO_TRAIN:],
                marker_color="skyblue",
                line_color="black",
                marker_symbol="circle",
                marker_size=10,
                name="LFO-test",
                showlegend=showlegend,
            ),
            row=r+1, col=c+1,
        )
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=data["AY_test"][idx],
                y=data["loss_ratio_test"][idx],
                marker_color="indianred",
                marker_symbol="circle",
                marker_size=10,
                name="Validation",
                showlegend=showlegend,
            ),
            row=r+1, col=c+1,
        )

    fig.update_layout(autosize=False, width=1500, height=600, font_size=20)
    fig.update_xaxes(tickfont_size = 15)
    if show:
        return fig
    else:
        fig.write_image(f"{FIGURE_PATH}/forecast-programs.png", scale=2)

def fit_models(data):
    fits = {}
    for name, model in MODELS.items():
        print(f"Fitting the {name} model...")
        fits[name] = model.sample(data=data, **STAN_CONFIG)
    return fits

def lfo(fits, data):
    indices = range(MIN_LFO_TRAIN, MAX_LFO_TRAIN + 1)
    steps = [(i, i + M) for i in indices]
    ll = {}
    for i, j in tqdm(steps):
        print(f"Fitting LFO through accident year {i} to predict year {j}.")
        step_data = data | {
            "T_train": i,
            "T_test": M,
            "AY_train": [ays[:i] for ays in data["AY_train"]],
            "AY_test": [ays[i:j] for ays in data["AY_train"]],
            "DL_train": [dls[:i] for dls in data["DL_train"]],
            "DL_test": [dls[i:j] for dls in data["DL_train"]],
            "loss_ratio_train": [lrs[:i] for lrs in data["loss_ratio_train"]],
            "loss_ratio_test": [lrs[i:j] for lrs in data["loss_ratio_train"]],
        }
        step_fits = fit_models(step_data)
        for name, fit in step_fits.items():
            if name in ll:
                ll[name].append((step_data, np.stack(fit.log_lik)))
            else:
                ll[name] = [(step_data, np.stack(fit.log_lik))]
    return ll

def blend(data, fits, lfo_lpd):
    model_draws = {
        k: bb.Draws(np.hstack([ll for _, ll in v]))
        for k, v
        in lfo_lpd.items()
    }
    pred_draws = {
        k: bb.Draws(
            log_lik = f.log_lik,
            post_pred = f.post_pred,
        )
        for k, f
        in fits.items()
    }
    continuous_covariates = {
        "earned_premium": np.array([
            d[MIN_LFO_TRAIN:MAX_LFO_TRAIN + 1]
            for d
            in data["earned_premium_train"]
        ]).flatten()
    }
    continuous_covariates_pred = {
        "earned_premium": np.array([
            d[MIN_LFO_TRAIN:MAX_LFO_TRAIN + 1]
            for d
            in data["earned_premium_test"]
        ]).flatten(),
        "earned_premium": np.array(data["earned_premium_test"]).flatten(),
    }
    discrete_covariates = {
        "AY": np.array([
              list(map(str, d[MIN_LFO_TRAIN:MAX_LFO_TRAIN+1]))
              for d
              in data["AY_train"]
          ]).flatten(),
        "program": np.array([
            [str(i)] * (MAX_LFO_TRAIN - MIN_LFO_TRAIN + 1)
            for i
            in range(data["N"])
        ]).flatten()
    }
    discrete_covariates_pred = {
        "AY": [str(d) for d in np.array(data["AY_test"]).flatten()],
        "program": [str(i) for i in range(data["N"])],
    }
    blend_models = {}
    for blend, model in BLEND_MODELS.items():
        if blend == "stacking-hier-bayes":
            f = model(
                model_draws=model_draws, 
                discrete_covariates=discrete_covariates,
                continuous_covariates=continuous_covariates,
                partial_pooling=True,
            )
            f.fit()
            preds = f.predict(
                model_draws=pred_draws,
                discrete_covariates=discrete_covariates_pred,
                continuous_covariates=continuous_covariates_pred,
            )
        else:
            f = model(model_draws=model_draws)
            f.fit()
            preds = f.predict(model_draws=pred_draws)
        blend_models[blend] = (f, preds)
    with open("forecast-weights.json", "w") as f:
        json.dump({k: {m: w.tolist() for m, w in b.weights.items()} for k, (b, _) in blend_models.items()}, f)
    return blend_models

def score(fits, blends):
    elpds = {
        k: (
            np.sum(logsumexp(f.log_lik, axis=0) - np.log(f.log_lik.shape[0])),
            np.sqrt(f.log_lik.shape[1]) * np.std(logsumexp(f.log_lik, axis=0) - np.log(f.log_lik.shape[0])),
        )
        for k, f 
        in fits.items()
    }
    elpds = elpds | {k: (blend.lpd.sum(), blend.lpd.std() * np.sqrt(blend.shape[1])) for k, (_, blend) in blends.items()}
    with open("forecast-scores.json", "w") as f:
        json.dump(elpds, f)
    return elpds

def plot_scores(data, blends, scores, show: bool = False):
    R = 5
    C = 2
    rc = [(r, c) for r in range(R) for c in range(C)]

    lrs = np.array(data["loss_ratio_test"]).flatten()

    def density(x, steps: int = 500):
        limits = min(x), max(x)
        grid = np.linspace(*limits, steps)
        return grid, gaussian_kde(x)(grid)
   
    keys = list(blends.keys())
    single_models = sorted(set(scores.keys()) - set(keys))

    COLOR_LOOKUP = {
        k: color
        for k, color
        in zip(
            list(TITLE_LOOKUP.keys())[len(single_models):],
            COLORS,
        )
    }

    titles = [TITLE_LOOKUP[k] for k in keys]
    subplot_titles = [v for t in titles for v in [f"{t} blend", f"{t} ELPD"]]

    fig = make_subplots(R, C, shared_xaxes="columns", subplot_titles=subplot_titles)
    fig.update_layout(template=TEMPLATE, title_xanchor="left")

    for i, (r, c) in enumerate(rc):
        show_legend=False
        if c == 0:
            g, d = density(blends[keys[r]][1].post_pred.flatten())
            fig.add_trace(
                go.Scatter(
                    mode="lines",
                    x=g,
                    y=d,
                    marker_color=COLOR_LOOKUP[keys[r]],
                    fill="tozeroy",
                    showlegend=show_legend,
                    name=TITLE_LOOKUP[keys[r]],
                ),
                row=r + 1,
                col=c + 1,
            )
            fig.add_trace(
                go.Scatter(
                    mode="markers",
                    x=lrs,
                    y=np.zeros(len(lrs)),
                    marker_symbol="x",
                    marker_color="black",
                    showlegend=show_legend,
                ),
                row=r + 1,
                col=c + 1,
            )
            if r + 1 == R:
                fig.update_xaxes(title="Loss ratio %", row=r+1, col=c+1)
            fig.update_yaxes(title="Density", row=r+1, col=c+1)
        else:
            elpd, se = zip(*[scores[keys[r]], *[scores[m] for m in single_models]])
            x_labels = [TITLE_LOOKUP[keys[r]], *[TITLE_LOOKUP[m] for m in single_models]]
            fig.add_trace(
                go.Scatter(
                    x=elpd,
                    y=x_labels,
                    mode="markers",
                    error_x={
                        "type": "data",
                        "array": se,
                        "width": 0,
                        "color": "gray",
                        "thickness": 0.75,
                    },
                    marker_color=[COLOR_LOOKUP[keys[r]], *["gray"] * len(single_models)],
                    marker_symbol=["circle", *["circle-open"] * len(single_models)],
                    marker_size=12,
                    showlegend=show_legend,
                    name=TITLE_LOOKUP[keys[r]],
                ),
                row=r + 1,
                col=c + 1,
            )
            if r + 1 == R:
                fig.update_xaxes(title="ELPD", row=r+1, col=c+1)

    fig.update_xaxes(range=[0, 1.5], col=1)
    fig.update_xaxes(range=[20, 70], col=2)
    fig.update_xaxes(zeroline=False, col=2)
    fig.update_layout(autosize=False, width=1500, height=1200, font_size=20)
    fig.update_xaxes(tickfont_size = 15)
    fig.update_yaxes(tickfont_size = 15, col=1)
    fig.update_yaxes(tickfont_size = 12, col=2)
    if show:
        return fig
    else:
        fig.write_image(FIGURE_PATH + "/forecast.png", scale=2)

def plot_weights(data, blends, show: bool = False):
    R = 1
    C = 6
    rc = [(r, c) for r in range(R) for c in range(C)]
    weights = {k: b.weights for k, (b, _) in blends.items()}
    lpds = {m: d.lpd for m, d in blends["pseudo-bma"][0].model_draws.items()}
    keys = list(MODELS)

    fig = make_subplots(
        R, C, 
        subplot_titles=[TITLE_LOOKUP[k] for k in keys],
        specs=[
            [{"secondary_y": True} for _ in range(C)]
            for _ in range(R)
        ],
    )
    fig.update_layout(template=TEMPLATE)
    fig.update_yaxes(range=[-0.05, 1], secondary_y=False)
    fig.update_yaxes(title="Weights", secondary_y=False, row=1, col=1)
    fig.update_yaxes(range=[-1, 2.5], secondary_y=True)
    fig.update_yaxes(title="LPD", secondary_y=True, row=1, col=C)
    fig.update_xaxes(title="LFO accident years", zeroline=False)
    fig.update_yaxes(zeroline=False)

    for i, (r, c) in enumerate(rc):
        w = {k: v[keys[i]] for k, v in weights.items()}
        lpd_i = lpds[keys[i]].reshape((data["N"], MAX_LFO_TRAIN - MIN_LFO_TRAIN + 1))
        fig.add_trace(
            go.Scatter(
                mode="markers+lines",
                x=[6, 7, 8, 9],
                y=lpd_i.mean(axis=0),
                error_y={
                    "type": "data",
                    "array": lpd_i.std(axis=0),
                    "width": 0,
                    "color": "gray",
                    "thickness": 0.5,
                },
                marker_symbol="circle-open",
                marker_size=12, 
                marker_color="black",
                line_width=0.5,
                line_color="gray",
                showlegend=not i,
                name="LPD",
            ),
            row=r+1,
            col=c+1,
            secondary_y=True,
        )
        for b, v in w.items():
            if v.shape[1] == 1:
                y = v.repeat(MAX_LFO_TRAIN - MIN_LFO_TRAIN + 1)
            else:
                y = v.reshape((data["N"], MAX_LFO_TRAIN - MIN_LFO_TRAIN + 1)).mean(axis=0)
            fig.add_trace(
                go.Scatter(
                    mode="markers+lines",
                    x=[6, 7, 8, 9],
                    y=y,
                    marker_symbol="circle",
                    marker_size=12, 
                    marker_color=COLOR_LOOKUP[b],
                    line_width=0.5,
                    line_color="gray",
                    showlegend=not i,
                    name=TITLE_LOOKUP[b],
                ),
                row=r+1,
                col=c+1,
                secondary_y=False,
            )
    fig.update_layout(autosize=False, width=2000, height=600, font_size=20)
    fig.update_xaxes(tickfont_size = 15)
    fig.update_yaxes(tickfont_size = 15)
    if show:
        return fig
    else:
        fig.write_image(FIGURE_PATH + "/forecast-weights.png", scale=2)

def main():
    data = load_forecast_data()
    plot_programs(data)
    fits = fit_models(data)
    lfo_lpd = lfo(fits, data)
    blends = blend(data, fits, lfo_lpd)
    scores = score(fits, blends)
    plot_scores(data, blends, scores)
    plot_weights(data, blends)

if __name__ == "__main__":
    main()
