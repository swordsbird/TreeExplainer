import qlib
from qlib.contrib.data.handler import Alpha158

data_handler_config = {
    "start_time": "2022-07-21",
    "end_time": "2023-08-21",
    "fit_start_time": "2022-07-21",
    "fit_end_time": "2023-07-17",
    "instruments": ["GS"],
    "freq": 'day',
}

qlib.init(provider_uri={'day': '~/.qlib/qlib_data/my_data/' })
h = Alpha158(**data_handler_config)
print(h.fetch(col_set="feature"))
