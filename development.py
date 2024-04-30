import json
import numpy as np
from functools import partial
from scipy.special import logsumexp
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cmdstanpy import CmdStanModel

import bayesblend as bb

FIGURE_PATH = "figures"
DATA = "data/development_data.json"
MODEL_PATH = "stan/development/exponential.stan"
SEED = 1

TEMPLATE = "none"

N_TEST = 20

STAN_CONFIG = {
    "chains": 4,
    "parallel_chains": 4,
    "iter_warmup": 1000,
    "iter_sampling": 2500,
    "adapt_delta": 0.99,
    "inits": 0,
    "seed": SEED,
}

ARGS_TEST = {"log_lik_name": "log_lik_test", "post_pred_name": "post_pred_test"}
ARGS_VALID = {"log_lik_name": "log_lik_valid", "post_pred_name": "post_pred_valid"}

MODEL = CmdStanModel(stan_file=MODEL_PATH)

BLEND_MODELS = {
    "pseudo-bma": partial(bb.PseudoBma, bootstrap=False),
    "pseudo-bma+": partial(bb.PseudoBma, seed=SEED),
    "stacking-mle": bb.MleStacking,
    "stacking-bayes": partial(bb.BayesStacking, seed=SEED),
    "stacking-hier-bayes": partial(bb.HierarchicalBayesStacking, seed=SEED),
}

COLORS = [
    "#264653",
    "#2a9d8f",
    "#e9c46a",
    "#f4a261",
    "#e76f51",
]

TITLE_LOOKUP = {
    "exp1": "Exponential (DL 1+)",
    "exp2": "Exponential (DL 2+)",
    "exp3": "Exponential (DL 3+)",
    "exp4": "Exponential (DL 4+)",
    "pseudo-bma": "Pseudo BMA",
    "pseudo-bma+": "Pseudo BMA+",
    "stacking-mle": "Stacking (MLE)",
    "stacking-bayes": "Stacking (Bayes)",
    "stacking-hier-bayes": "Hierarchical stacking",
}

COLOR_LOOKUP = {
    k: color
    for k, color in zip(
        list(TITLE_LOOKUP.keys())[4:],
        COLORS,
    )
}


def load_development_data():
    return json.load(open(DATA, "r"))


def fit_models(data):
    data_dict = {
        "exp1": data | {"DL_start": 2},
        "exp2": data | {"DL_start": 3},
        "exp3": data | {"DL_start": 4},
        "exp4": data | {"DL_start": 5},
    }
    fits = {}
    for name, d in data_dict.items():
        print(f"Fitting the {name} model...")
        fits[name] = MODEL.sample(data=d, **STAN_CONFIG)
    return fits


def blend(data, fits):
    model_draws = {
        model: bb.Draws.from_cmdstanpy(fit, **ARGS_TEST) for model, fit in fits.items()
    }
    pred_draws = {
        model: bb.Draws.from_cmdstanpy(fit, **ARGS_VALID) for model, fit in fits.items()
    }
    discrete_covariates = {
        "dev_lag": [str(v) for v in data["DL_test_vals"]],
        "acc_year": [str(v) for v in data["AY_test_vals"]],
    }
    discrete_covariates_pred = {
        "dev_lag": [str(v) for v in data["DL_valid_vals"]],
        "acc_year": [str(v) for v in data["AY_valid_vals"]],
    }
    blend_models = {}
    for blend, model in BLEND_MODELS.items():
        if blend == "stacking-hier-bayes":
            f = model(
                model_draws=model_draws,
                discrete_covariates=discrete_covariates,
                partial_pooling=True,
            )
            f.fit()
            preds = f.predict(
                model_draws=pred_draws,
                discrete_covariates=discrete_covariates_pred,
            )
        else:
            f = model(model_draws=model_draws)
            f.fit()
            preds = f.predict(model_draws=pred_draws)
        blend_models[blend] = (f, preds)
    with open("development-weights.json", "w") as f:
        json.dump(
            {
                k: {m: w.tolist() for m, w in b.weights.items()}
                for k, (b, _) in blend_models.items()
            },
            f,
        )
    return blend_models


def score(fits, blends):
    elpds = {
        k: (
            np.sum(
                logsumexp(f.log_lik_valid, axis=0) - np.log(f.log_lik_valid.shape[0])
            ),
            np.sqrt(np.sum(f.log_lik_valid.shape[1:]))
            * np.std(
                logsumexp(f.log_lik_valid, axis=0) - np.log(f.log_lik_valid.shape[0])
            ),
        )
        for k, f in fits.items()
    }
    elpds = elpds | {
        k: (blend.lpd.sum(), blend.lpd.std() * np.sqrt(len(blend.lpd)))
        for k, (_, blend) in blends.items()
    }
    with open("development-scores.json", "w") as f:
        json.dump(elpds, f)
    return elpds


def plot_scores(data, blends, scores, show: bool = False):
    R = 5
    C = 2
    rc = [(r, c) for r in range(R) for c in range(C)]
    lrs = np.array(data["loss_ratio_valid"]).flatten()

    def density(x, steps: int = 500):
        limits = min(x), max(x)
        grid = np.linspace(*limits, steps)
        return grid, gaussian_kde(x)(grid)

    keys = list(blends.keys())
    single_models = sorted(set(scores.keys()) - set(keys))
    COLOR_LOOKUP = {
        k: color
        for k, color in zip(
            list(TITLE_LOOKUP.keys())[len(single_models) :],
            COLORS,
        )
    }
    titles = [TITLE_LOOKUP[k] for k in keys]
    subplot_titles = [v for t in titles for v in [f"{t} blend", f"{t} ELPD"]]
    fig = make_subplots(R, C, shared_xaxes="columns", subplot_titles=subplot_titles)
    fig.update_layout(template=TEMPLATE, title_xanchor="left")
    for i, (r, c) in enumerate(rc):
        show_legend = False
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
                fig.update_xaxes(title="Loss ratio %", row=r + 1, col=c + 1)
            fig.update_yaxes(title="Density", row=r + 1, col=c + 1)
        else:
            elpd, se = zip(*[scores[keys[r]], *[scores[m] for m in single_models]])
            x_labels = [
                TITLE_LOOKUP[keys[r]],
                *[TITLE_LOOKUP[m] for m in single_models],
            ]
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
                    marker_color=[
                        COLOR_LOOKUP[keys[r]],
                        *["gray"] * len(single_models),
                    ],
                    marker_symbol=["circle", *["circle-open"] * len(single_models)],
                    marker_size=12,
                    showlegend=show_legend,
                    name=TITLE_LOOKUP[keys[r]],
                ),
                row=r + 1,
                col=c + 1,
            )
            if r + 1 == R:
                fig.update_xaxes(title="ELPD", row=r + 1, col=c + 1)
    fig.update_xaxes(range=[0, 1.5], col=1)
    fig.update_xaxes(range=[1400, 1650], col=2)
    fig.update_xaxes(zeroline=False, col=2)
    fig.update_layout(autosize=False, width=1500, height=1200, font_size=20)
    fig.update_xaxes(tickfont_size=15)
    fig.update_yaxes(tickfont_size=15, col=1)
    fig.update_yaxes(tickfont_size=12, col=2)
    if show:
        return fig
    else:
        fig.write_image(FIGURE_PATH + "/development.png", scale=2)


def plot_weights(data, blends, show: bool = False):
    R = 1
    C = 4
    rc = [(r, c) for r in range(R) for c in range(C)]
    weights = {k: b.weights for k, (b, _) in blends.items()}
    lpds = {m: d.lpd for m, d in blends["pseudo-bma"][0].model_draws.items()}
    keys = list(TITLE_LOOKUP.keys())[:4]
    fig = make_subplots(
        R,
        C,
        subplot_titles=[TITLE_LOOKUP[k] for k in keys],
        specs=[[{"secondary_y": True} for _ in range(C)] for _ in range(R)],
    )
    fig.update_layout(template=TEMPLATE)
    fig.update_yaxes(range=[-0.05, 1.05], secondary_y=False)
    fig.update_yaxes(title="Weights", secondary_y=False, row=1, col=1)
    fig.update_yaxes(range=[0, 7], secondary_y=True)
    fig.update_yaxes(title="LPD", secondary_y=True, row=1, col=C)
    fig.update_xaxes(title="Development years", zeroline=False)
    fig.update_yaxes(zeroline=False)
    for i, (r, c) in enumerate(rc):
        w = {k: v[keys[i]] for k, v in weights.items()}
        lpd_i = lpds[keys[i]].reshape((data["N"], N_TEST))
        fig.add_trace(
            go.Scatter(
                mode="markers+lines",
                x=[8, 9],
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
            row=r + 1,
            col=c + 1,
            secondary_y=True,
        )
        for b, v in w.items():
            if v.shape[1] == 1:
                y = v.repeat(N_TEST)
            else:
                y = v.reshape((data["N"], N_TEST)).mean(axis=0)
            fig.add_trace(
                go.Scatter(
                    mode="markers+lines",
                    x=[8, 9],
                    y=y,
                    marker_symbol="circle",
                    marker_size=12,
                    marker_color=COLOR_LOOKUP[b],
                    line_width=0.5,
                    line_color="gray",
                    showlegend=not i,
                    name=TITLE_LOOKUP[b],
                ),
                row=r + 1,
                col=c + 1,
                secondary_y=False,
            )
    fig.update_layout(autosize=False, width=2000, height=600, font_size=20)
    fig.update_xaxes(tickfont_size=15)
    fig.update_yaxes(tickfont_size=15)
    if show:
        return fig
    else:
        fig.write_image(FIGURE_PATH + "/development-weights.png", scale=2)


def main():
    data = load_development_data()
    fits = fit_models(data)
    blends = blend(data, fits)
    scores = score(fits, blends)
    plot_scores(data, blends, scores)
    plot_weights(data, blends, show=True)


if __name__ == "__main__":
    main()
