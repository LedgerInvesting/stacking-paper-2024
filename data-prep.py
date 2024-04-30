from typing import Dict, List, Any, Callable

import pandas as pd
import numpy as np
from collections import namedtuple
import json

DATA_URL = "https://www.casact.org/sites/default/files/2021-04/ppauto_pos.csv"

KEEP_CODES = [
    43,
    353,
    388,
    620,
    692,
    715,
    1066,
    1090,
    1538,
    1767,
    2003,
    2143,
    3240,
    4839,
    5185,
    6807,
    6947,
    7080,
    8427,
    8559,
    10022,
    13420,
    13439,
    13501,
    13528,
    13587,
    13595,
    13641,
    13889,
    14044,
    14176,
    14257,
    14311,
    14443,
    15024,
    15199,
    15393,
    15660,
    15997,
    16373,
    16799,
    18163,
    18791,
    23574,
    23876,
    25275,
    26808,
    27022,
    27065,
    27499,
]

ORIGIN_ACCIDENT_YEAR = 1988
T_DEV_TRAIN = 70
T_DEV_TEST = 20
T_DEV_VALID = 10
T_FORECAST_TRAIN = 9
T_FORECAST_TEST = 10 - T_FORECAST_TRAIN
ULTIMATE = 10

Cell = namedtuple(
    "Cell",
    (
        "code",
        "accident_year",
        "evaluation_year",
        "development_lag",
        "incurred_loss",
        "paid_loss",
        "earned_premium",
    ),
)
DataDictType = Dict[str, int | List[float] | List[int]]


def download_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL)


def get_flattened_covariate(
    cell_list: List[Cell],
    field: str,
    condition: Callable = lambda cell: True,
) -> List[Any]:
    nested_list = [
        [getattr(cell, field) for cell in cells if condition(cell)]
        for cells in cell_list
    ]
    return [item for sublist in nested_list for item in sublist]


def make_cells(data: pd.DataFrame) -> Dict:
    cols = [
        "GRCODE",
        "AccidentYear",
        "DevelopmentYear",
        "DevelopmentLag",
        "IncurLoss_B",
        "CumPaidLoss_B",
        "EarnedPremDIR_B",
    ]
    cells = [Cell(*values) for _, values in data[cols].iterrows()]
    return [cell for cell in cells if cell.code in KEEP_CODES]


def build_development_data(cells: List[Cell]) -> DataDictType:
    raw = {}
    for cell in cells:
        if cell.code in raw:
            raw[cell.code].append(cell)
        else:
            raw[cell.code] = [cell]
    train = {
        code: [cell for cell in cells if (cell.development_lag <= 7)]
        for code, cells in raw.items()
    }
    test = {
        code: [cell for cell in cells if (7 < cell.development_lag <= 9)]
        for code, cells in raw.items()
    }
    validation = {
        code: [cell for cell in cells if cell.development_lag == 10]
        for code, cells in raw.items()
    }
    ay_train_lookup = {
        year: i + 1
        for i, year in enumerate(
            set([cell.accident_year for cell in list(train.values())[0]])
        )
    }
    ay_test_lookup = {
        year: i + 1
        for i, year in enumerate(
            set([cell.accident_year for cell in list(test.values())[0]])
        )
    }
    ay_valid_lookup = {
        year: i + 1
        for i, year in enumerate(
            set([cell.accident_year for cell in list(validation.values())[0]])
        )
    }
    assert all(len(cells) == T_DEV_TRAIN for cells in train.values())
    assert all(len(cells) == T_DEV_TEST for cells in test.values())
    assert all(len(cells) == T_DEV_VALID for cells in validation.values())

    train_array = [np.full((10, T_DEV_TRAIN // 10), -1.0) for _ in range(len(train))]
    test_array = [np.full((10, T_DEV_TEST // 10), -1.0) for _ in range(len(train))]
    valid_array = [
        np.full((10, T_DEV_VALID // 10), -1.0) for _ in range(len(validation))
    ]

    for i, cells in enumerate(train.values()):
        for cell in cells:
            train_array[i][
                ay_train_lookup[cell.accident_year] - 1, cell.development_lag - 1
            ] = (cell.paid_loss / cell.earned_premium)
    for i, cells in enumerate(test.values()):
        for cell in cells:
            test_array[i][
                ay_test_lookup[cell.accident_year] - 1, cell.development_lag - 8
            ] = (cell.paid_loss / cell.earned_premium)
    for i, cells in enumerate(validation.values()):
        for cell in cells:
            valid_array[i][
                ay_valid_lookup[cell.accident_year] - 1, cell.development_lag - 10
            ] = (cell.paid_loss / cell.earned_premium)

    AY_train, DL_train = train_array[0].shape
    AY_test, DL_test = test_array[0].shape
    AY_valid, DL_valid = valid_array[0].shape

    return {
        "N": len(train),
        "AY_train": AY_train,
        "DL_train": DL_train,
        "AY_train_vals": get_flattened_covariate(
            train.values(), "accident_year", lambda cell: cell.development_lag > 1
        ),
        "DL_train_vals": get_flattened_covariate(
            train.values(), "development_lag", lambda cell: cell.development_lag > 1
        ),
        "AY_test": AY_test,
        "DL_test": DL_test,
        "AY_test_vals": get_flattened_covariate(test.values(), "accident_year"),
        "DL_test_vals": get_flattened_covariate(test.values(), "development_lag"),
        "AY_valid": AY_valid,
        "DL_valid": DL_valid,
        "AY_valid_vals": get_flattened_covariate(validation.values(), "accident_year"),
        "DL_valid_vals": get_flattened_covariate(
            validation.values(), "development_lag"
        ),
        "premium_train": get_flattened_covariate(train.values(), "earned_premium"),
        "premium_test": get_flattened_covariate(test.values(), "earned_premium"),
        "premium_valid": get_flattened_covariate(validation.values(), "earned_premium"),
        "loss_ratio_train": [array.tolist() for array in train_array],
        "loss_ratio_test": [array.tolist() for array in test_array],
        "loss_ratio_valid": [array.tolist() for array in valid_array],
    }


def build_forecast_data(cells: List[Cell]) -> DataDictType:
    raw = {}
    for cell in cells:
        if cell.development_lag == ULTIMATE:
            if cell.code in raw:
                raw[cell.code].append(cell)
            else:
                raw[cell.code] = [cell]
    TRAIN_CUTOFF = ORIGIN_ACCIDENT_YEAR + T_FORECAST_TRAIN
    train = {
        code: [cell for cell in cells if cell.accident_year < TRAIN_CUTOFF]
        for code, cells in raw.items()
    }
    test = {
        code: [cell for cell in cells if cell.accident_year >= TRAIN_CUTOFF]
        for code, cells in raw.items()
    }
    ay_lookup = {
        year: i + 1
        for i, year in enumerate(set([cell.accident_year for cell in cells]))
    }
    assert all(len(cells) == T_FORECAST_TRAIN for cells in train.values())
    assert all(len(cells) == T_FORECAST_TEST for cells in test.values())
    return {
        "N": len(train),
        "T_train": T_FORECAST_TRAIN,
        "T_test": T_FORECAST_TEST,
        "AY_train": [
            [ay_lookup[cell.accident_year] for cell in cells]
            for cells in train.values()
        ],
        "AY_test": [
            [ay_lookup[cell.accident_year] for cell in cells] for cells in test.values()
        ],
        "DL_train": [
            [cell.development_lag for cell in cells] for cells in train.values()
        ],
        "DL_test": [
            [cell.development_lag for cell in cells] for cells in test.values()
        ],
        "earned_premium_train": [
            [cell.earned_premium for cell in cells] for cells in train.values()
        ],
        "earned_premium_test": [
            [cell.earned_premium for cell in cells] for cells in test.values()
        ],
        "loss_ratio_train": [
            [cell.incurred_loss / cell.earned_premium for cell in cells]
            for cells in train.values()
        ],
        "loss_ratio_test": [
            [cell.incurred_loss / cell.earned_premium for cell in cells]
            for cells in test.values()
        ],
    }


def main():
    data = download_data()
    cells = make_cells(data)
    development_data = build_development_data(cells)
    forecast_data = build_forecast_data(cells)
    with open("data/development_data.json", "w") as outfile:
        json.dump(development_data, outfile)
    with open("data/forecast_data.json", "w") as outfile:
        json.dump(forecast_data, outfile)
    return None


if __name__ == "__main__":
    main()
