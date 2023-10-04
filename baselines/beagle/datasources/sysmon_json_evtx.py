import datetime
from typing import TYPE_CHECKING
from typing import TYPE_CHECKING, Generator
import Evtx.Evtx as evtx
from lxml import etree
import json

from beagle.datasources.win_evtx import WinEVTX
from beagle.transformers.sysmon_transformer import SysmonTransformer

if TYPE_CHECKING:
    from beagle.transformer.base_transformer import Transformer
    from typing import List
    
class SysmonJSONEVTX(WinEVTX):

    name = "Sysmon EVTX JSON File"
    transformers = [SysmonTransformer] 
    category = "SysMon"

    def __init__(self, sysmon_evtx_log_file: str) -> None:
        super().__init__(sysmon_evtx_log_file)

    def metadata(self) -> dict:
        """Returns the Hostname by inspecting the `Computer` entry of the
        first record.

        Returns
        -------
        dict
            >>> {"hostname": str}
        """
            
        with open(self.file_path,'r') as f:
            for line in f:
                if "microsoft-windows-sysmon" in line.lower():
                    event=json.loads(line)
                    event=self.parse_record_json(event)
                    break
        return {"hostname": event["Computer"]}
    
    def events(self) -> Generator[dict,None,None]:
        with open(self.file_path,'r') as f:
            for line in f:
                if "microsoft-windows-sysmon" in line.lower():
                    data=json.loads(line)
                    yield self.parse_record_json(data)
                
    def parse_record_json(self,record)->dict:
        return record
    
    

