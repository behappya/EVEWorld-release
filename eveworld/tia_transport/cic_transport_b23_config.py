"""Twin of `cic_transport_config` with transport after block 23.

`t4g_id_block` and `cic_transport_after_block` must move together.
"""

from __future__ import annotations

import copy
import os

from eveworld.tia_transport.cic_transport_config import config as _base


config = copy.deepcopy(_base)
config["models"].update(
    t4g_id_block="block23",
    cic_transport_after_block="block23",
)
config["project_dir"] = (
    f"{os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))}/eve_v2_outputs/eve_cic_transport_v1/"
    "cic_transport_b23_seed42_s300/experiments"
)
