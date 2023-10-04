from typing import Dict, Optional, Tuple, Union

from beagle.common import logger, split_path
from beagle.constants import Protocols
from beagle.nodes import URI, Domain, File, IPAddress, Node, Process, RegistryKey, Alert
from beagle.transformers.base_transformer import Transformer


class WinEVTXJsonTransformer(Transformer):
    name = "Win EVTX json"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        logger.info("Created Windows EVTX json Transformer.")

    def transform(self, event: dict) -> Optional[Tuple]:
        # Track which processese we've seen
        self.seen_procs: Dict[int, Process] = {}

        event_id = int(event["Event ID"])

        if event_id == 4688:
            return self.process_creation(event)
        # elif event_id == 4656:
        #     return self.request_object(event)
        elif event_id == 4663:
            return self.access_object(event)
        # # elif event_id == 4660:
        # #     return self.delete_object(event)
        # elif event_id == 4657:
        #     return self.modify_registry(event)

        return None

    def process_creation(self, event: dict) -> Tuple[Process, File, Process]:
        """Transformers a process creation (event ID 4688) into a set of nodes.

        https://www.ultimatewindowssecurity.com/securitylog/encyclopedia/event.aspx?eventID=4688

        Parameters
        ----------
        event : dict
            [description]

        Returns
        -------
        Optional[Tuple[Process, File, Process, File]]
            [description]
        """

        # Get the parent PID
        parent_pid = int(event["Creator Process ID"], 16)

        # Child PID
        child_pid = int(event["New Process ID"], 16)

        proc_name, proc_path = split_path(event["New Process Name"])

        child = Process(
            host=event["Account Name"],
            process_id=child_pid,
            user=event["Account Domain"],
            process_image=proc_name,
            process_image_path=proc_path,
            command_line=event.get("data_name_commandline"),
        )
        child.name = event["New Process Name"]

        child_file = child.get_file_node()
        child_file.file_of[child]

        # Map the process for later
        self.seen_procs[child_pid] = child

        parent = self.seen_procs.get(parent_pid)

        if parent is None:
            # Create a dummy proc. If we haven't already seen the parent
            parent = Process(host=event["Account Name"], process_id=parent_pid)
            # parent.name=str(parent_pid)
        parent.launched[child].append(timestamp=event["Time"])

        # Don't need to pull out the parent's file, as it will have always
        # been created before being put into seen_procs

        return (child, parent)

    def request_object(self, event: dict) -> Tuple[Process, File, Process]:
        proc_pid = int(event["Process ID"], 16)
        proc_name, proc_path = split_path(event["Process Name"])
        proc = Process(
            host=event["Account Name"],
            process_id=proc_pid,
            user=event["Account Domain"],
            process_image=proc_name,
            process_image_path=proc_path,
            command_line=event.get("data_name_commandline"),
        )

        # proc_file = proc.get_file_node()
        proc.name = event["Process Name"]
        # proc_file.file_of[proc]
        file_name, file_path = split_path(event["Object Name"])
        target_file = File(file_name=file_name, file_path=file_path)

        proc.accessed[target_file].append(timestamp=event["Time"])

        return (proc, target_file)

    def access_object(self, event: dict):
        proc_pid = int(event["Process ID"], 16)
        proc_name, proc_path = split_path(event["Process Name"])
        proc = Process(
            host=event["Account Name"],
            process_id=proc_pid,
            user=event["Account Domain"],
            process_image=proc_name,
            process_image_path=proc_path,
        )
        proc.name = event["Process Name"]
        proc_file = proc.get_file_node()
        proc_file.file_of[proc]
        file_name, file_path = split_path(event["Object Name"])
        target_file = File(file_name=file_name, file_path=file_path)
        target_file.name = event["Object Name"]
        if "ReadData" in event["Accesses"]:
            proc.loaded[target_file].append(timestamp=event["Time"])
            return (proc, target_file)
        elif "WriteData" in event["Accesses"]:
            proc.wrote[target_file].append(timestamp=event["Time"])
            return (proc, target_file)
        elif "DeleteChild" in event["Accesses"]:
            proc.deleted[target_file].append(timestamp=event["Time"])
            return (proc, target_file)
        return None

    def modify_registry(self, event: dict):
        proc_pid = int(event["Process ID"], 16)
        proc_name, proc_path = split_path(event["Process Name"])
        proc = Process(
            host=event["Account Name"],
            process_id=proc_pid,
            user=event["Account Domain"],
            process_image=proc_name,
            process_image_path=proc_path,
        )

        proc_file = proc.get_file_node()
        proc_file.file_of[proc]

        key_path = event["ObjectName"]
        hive = key_path.split("\\")[1]
        key = key_path.split("\\")[-1]
        # Always has a leading \\ so split from 2:
        key_path = "\\".join(key_path.split("\\")[2:-1])

        # RegistryKey Node Creation
        reg_node = RegistryKey(
            hive=hive,
            key_path=key_path,
            key=key,
            value=event.get(event["NewValue"]),
            value_type=event.get(event["NewValueType"]),
        )

        if reg_node.value:
            proc.changed_value[reg_node].append(value=reg_node.value)
        else:
            proc.changed_value[reg_node]

        return (proc, reg_node)
