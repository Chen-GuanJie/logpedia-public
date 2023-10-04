import time
from typing import TYPE_CHECKING, Generator

import json
from beagle.common.logging import logger
from beagle.datasources.base_datasource import DataSource
from beagle.transformers.evtx_json_transformer import WinEVTXJsonTransformer

if TYPE_CHECKING:
    from beagle.transformer.base_transformer import Transformer
    from typing import List


class WinEVTXJson(DataSource):
    """Parses Windows .evtx files. Yields events one by one using the `python-evtx` library.

    Parameters
    ----------
    evtx_log_file : str
        The path to the windows evtx file to parse.
    """

    name = "Windows EVTX json File"
    transformers = [WinEVTXJsonTransformer]  # type: List[Transformer]
    category = "Windows Event json Logs"

    def __init__(self, evtx_log_file: str) -> None:

        self.file_path = evtx_log_file

        logger.info(f"Setting up WinEVTX for {self.file_path}")

    def events(self) -> Generator[dict, None, None]:
        with open(self.file_path) as f:
            logs = json.load(f)
            for log in logs:
                yield self.parse_record(log)

    def metadata(self) -> dict:
        """Get the hostname by inspecting the first record.

        Returns
        -------
        dict
            >>> {"hostname": str}
        """
        with open(self.file_path) as f:
            logs = json.load(f)
            for log in logs:
                event = self.parse_record(log)
                break

        return {"hostname": event["Account Name"]}

    def parse_record(self,record: dict, name: str = "") -> dict:
        data = {}
        if isinstance(record, str) and name != "":
            data[name] = record
            if name == "Time":
                ts = int(time.mktime(time.strptime(data["Time"], "%m/%d/%Y %H:%M:%S %p")))
                data["Time"] = ts
        else:
            for key, value in record.items():
                data.update(self.parse_record(value, key))
        return data

