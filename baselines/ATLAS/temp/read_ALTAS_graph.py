import networkx as nx
import os
import re
import csv

# get networkx graph object from processed format logs of ATLAS 
def get_ATLAS_G(log_file_name, IncludeExecutedEdges=True, StartTime=0):
    
    G = nx.MultiDiGraph(name=log_file_name, data=True, align='vertical')
    log_file = open(log_file_name,"r")
    lines = log_file.readlines()
    processes = {}
    hosts_ips = ['192.168.223.128']
    
    for line in lines:
        if "FMfcgxvzKb" in line:
            print(line)
        line = line.lower().replace("\\", "/")
        splitted_line = line.split(",")
        if len(splitted_line) < 15:
            continue
        
        # DNS
        if len(splitted_line[1]) > 0 and len(splitted_line[2]) > 0:
            edge_type = "resolve"
            edge_label = edge_type + "_" + str(splitted_line[0])
            domain_name = splitted_line[1]
            IP_Address = splitted_line[2] #.replace(":", "_")
            if int(splitted_line[0]) >= StartTime:
                if not G.has_node(domain_name):
                    G.add_node(domain_name, type="domain_name", timestamp=splitted_line[0])
                if not G.has_node(IP_Address):
                    G.add_node(IP_Address, type="IP_Address", timestamp=splitted_line[0])
                if not G.has_edge(domain_name, IP_Address):
                    G.add_edge(domain_name, IP_Address, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0])
        
        # web_object to domain_name (in referal)
        if len(splitted_line[15]) > 0 and not splitted_line[15].startswith("/"): #  and not splitted_line[15].startswith("/") and "/" in splitted_line[15]
            edge_type = "web_request"
            domain_name = splitted_line[15]
            if ":" in domain_name:
                domain_name = domain_name.split(":")[0]
            if "://" in domain_name:
                domain_name = domain_name.split("://")[1]
            if "/" in domain_name:
                domain_name = domain_name[:domain_name.find("/")]
            web_object = splitted_line[15] # .replace(":", "_")
            if not "/" in web_object:
                web_object += "/"
            if "//" in web_object:
                web_object = web_object.replace("//", "/")
            edge_label = edge_type + "_" + str(splitted_line[0])
            if int(splitted_line[0]) >= StartTime:
                if not G.has_node(domain_name):
                    G.add_node(domain_name, type="domain_name", timestamp=splitted_line[0])
                if not G.has_node(web_object):
                    G.add_node(web_object, type="web_object", timestamp=splitted_line[0])
                if not G.has_edge(web_object, domain_name):
                    G.add_edge(web_object, domain_name, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0])

        # web_object to domain_name
        if len(splitted_line[14]) > 0:
            edge_type = "web_request"
            domain_name = splitted_line[14]
            if ":" in domain_name:
                domain_name = domain_name[:domain_name.find(":")]
            if "/" in domain_name:
                domain_name = domain_name[:domain_name.find("/")]
            web_object = splitted_line[14]
            if not "/" in web_object:
                web_object += "/"
            web_object = web_object # .replace(":", "_")
            if len(splitted_line[11]) > 0:
                url = splitted_line[11] # .replace(":", "_")
                if url.startswith("/"):
                    web_object = splitted_line[14] + url # .replace(":", "_") splitted_line[14].replace(":", "_") 
                else:
                    #web_object = splitted_line[14].replace(":", "_") + "/" + url.replace(":", "_")
                    web_object = splitted_line[11] # .replace(":", "_")
            elif len(splitted_line[12]) > 0:
                url = splitted_line[12]
                if url.startswith("/"):
                    web_object = splitted_line[14] + url # .replace(":", "_") splitted_line[14].replace(":", "_")
                else:
                    #web_object = splitted_line[14].replace(":", "_") + "/" + url.replace(":", "_")
                    web_object = splitted_line[12] # .replace(":", "_")
            edge_label = edge_type + "_" + str(splitted_line[0])
            web_object = web_object.replace("//", "/")
            if int(splitted_line[0]) >= StartTime:
                if not G.has_node(domain_name):
                    G.add_node(domain_name, type="domain_name", timestamp=splitted_line[0])
                if not G.has_node(web_object):
                    G.add_node(web_object, type="web_object", timestamp=splitted_line[0])
                if not G.has_edge(web_object, domain_name):
                    G.add_edge(web_object, domain_name, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0])
            # web_object (from referal) to web_object in request/response
            if len(splitted_line[15]) > 0:
                edge_type = "refer"
                edge_label = edge_type + "_" + str(splitted_line[0])
                web_object0 = splitted_line[15] # .replace(":", "_")
                if int(splitted_line[0]) >= StartTime:
                    if not G.has_node(web_object0):
                        G.add_node(web_object0, type="web_object", timestamp=splitted_line[0])
                    if not G.has_node(web_object):
                        G.add_node(web_object, type="web_object", timestamp=splitted_line[0])
                    if not G.has_edge(web_object, web_object0):
                        G.add_edge(web_object, web_object0, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0])
        
        # POST web_object to domain_name
        elif len(splitted_line[12]) > 0:
            IsValidIP = False
            cleaned_ip = ""
            edge_type = "web_request"
            edge_label = edge_type + "_" + str(splitted_line[0])
            domain_name = splitted_line[14]
            if not ":" in domain_name:
                IsValidIP = re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain_name)
                if IsValidIP:
                    cleaned_ip = domain_name
                    domain_name += "_website"
            else:
                IsValidIP = re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain_name.split(":")[0])
                if IsValidIP:
                    cleaned_ip = domain_name.split(":")[0]
                    domain_name = domain_name.split(":")[0] + "_website_" + domain_name.split(":")[1]
                else:
                    domain_name = domain_name # .replace(":", "_")

            if "/" in domain_name:
                domain_name = domain_name[:domain_name.find("/")]

            web_object = domain_name + splitted_line[12]

            if not "/" in web_object:
                web_object += "/"

            if int(splitted_line[0]) >= StartTime:
                if not G.has_node(domain_name):
                    G.add_node(domain_name, type="domain_name", timestamp=splitted_line[0])
                if not G.has_node(web_object):
                    G.add_node(web_object, type="web_object", timestamp=splitted_line[0])
                if not G.has_edge(web_object, domain_name):
                    G.add_edge(web_object, domain_name, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0])
                if IsValidIP:
                    edge_type = "resolve"
                    edge_label = edge_type + "_" + str(splitted_line[0])
                    if not G.has_node(cleaned_ip):
                        G.add_node(cleaned_ip, type="IP_Address", timestamp=splitted_line[0])
                    if not G.has_edge(domain_name, cleaned_ip):
                        G.add_edge(domain_name, cleaned_ip, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0])

            if len(splitted_line[15]) > 0:
                IsValidIP = False
                cleaned_ip = ""
                edge_type = "refer"
                edge_label = edge_type + "_" + str(splitted_line[0])
                domain_name = splitted_line[15]
                if not ":" in domain_name:
                    IsValidIP = re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain_name)
                    if IsValidIP:
                        cleaned_ip = domain_name
                        domain_name += "_website"
                else:
                    IsValidIP = re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain_name.split(":")[0])
                    if IsValidIP:
                        cleaned_ip = domain_name.split(":")[0]
                        domain_name = domain_name.split(":")[0] + "_website_" + domain_name.split(":")[1]
                    else:
                        domain_name = domain_name # .replace(":", "_")

                if "/" in domain_name:
                    domain_name = domain_name[:domain_name.find("/")]

                if int(splitted_line[0]) >= StartTime:
                    if not G.has_node(domain_name):
                        G.add_node(domain_name, type="domain_name", timestamp=splitted_line[0])
                    if not G.has_node(web_object):
                        G.add_node(web_object, type="web_object", timestamp=splitted_line[0])
                    if not G.has_edge(web_object, domain_name):
                        G.add_edge(web_object, domain_name, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0])
                    if IsValidIP:
                        edge_type = "resolve"
                        edge_label = edge_type + "_" + str(splitted_line[0])
                        if not G.has_node(cleaned_ip):
                            G.add_node(cleaned_ip, type="IP_Address", timestamp=splitted_line[0])
                        if not G.has_edge(domain_name, cleaned_ip):
                            G.add_edge(domain_name, cleaned_ip, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0])

        # GET
        elif len(splitted_line[11]) > 0:
            IsValidIP = False
            cleaned_ip = ""
            edge_type = "web_request"
            edge_label = edge_type + "_" + str(splitted_line[0])
            domain_name = splitted_line[11]
            if not "/" in splitted_line[11]:
                domain_name = splitted_line[11]
            else:
                domain_name = splitted_line[11][:splitted_line[11].find("/")]
            
            if not ":" in domain_name:
                IsValidIP = re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain_name)
                if IsValidIP:
                    cleaned_ip = domain_name
                    domain_name += "_website"
            else:
                IsValidIP = re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain_name.split(":")[0])
                if IsValidIP:
                    cleaned_ip = domain_name.split(":")[0]
                    domain_name = domain_name.split(":")[0] + "_website_" + domain_name.split(":")[1]
                else:
                    domain_name = domain_name # .replace(":", "_")

            if "/" in domain_name:
                domain_name = domain_name[:domain_name.find("/")]

            web_object = domain_name + splitted_line[11][splitted_line[11].find("/"):] # .replace(":", "_")

            if not "/" in web_object:
                web_object += "/"

            if int(splitted_line[0]) >= StartTime:
                if not G.has_node(domain_name):
                    G.add_node(domain_name, type="domain_name", timestamp=splitted_line[0])
                if not G.has_node(web_object):
                    G.add_node(web_object, type="web_object", timestamp=splitted_line[0])
                if not G.has_edge(web_object, domain_name):
                    G.add_edge(web_object, domain_name, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0])
                if IsValidIP:
                    edge_type = "resolve"
                    edge_label = edge_type + "_" + str(splitted_line[0])
                    if not G.has_node(cleaned_ip):
                        G.add_node(cleaned_ip, type="IP_Address", timestamp=splitted_line[0])
                    if not G.has_edge(domain_name, cleaned_ip):
                        G.add_edge(domain_name, cleaned_ip, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0])


            if len(splitted_line[15]) > 0:
                IsValidIP = False
                cleaned_ip = ""
                edge_type = "refer"
                edge_label = edge_type + "_" + str(splitted_line[0])
                domain_name = splitted_line[15]
                if not ":" in domain_name:
                    IsValidIP = re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain_name)
                    if IsValidIP:
                        cleaned_ip = domain_name
                        domain_name += "_website"
                else:
                    IsValidIP = re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain_name.split(":")[0])
                    if IsValidIP:
                        cleaned_ip = domain_name.split(":")[0]
                        domain_name = domain_name.split(":")[0] + "_website_" + domain_name.split(":")[1]
                    else:
                        domain_name = domain_name  # .replace(":", "_")

                if "/" in domain_name:
                    domain_name = domain_name[:domain_name.find("/")]

                if int(splitted_line[0]) >= StartTime:
                    if not G.has_node(domain_name):
                        G.add_node(domain_name, type="domain_name", timestamp=splitted_line[0])
                    if not G.has_node(web_object):
                        G.add_node(web_object, type="web_object", timestamp=splitted_line[0])
                    if not G.has_edge(web_object, domain_name):
                        G.add_edge(web_object, domain_name, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0])
                    if IsValidIP:
                        edge_type = "resolve"
                        edge_label = edge_type + "_" + str(splitted_line[0])
                        if not G.has_node(cleaned_ip):
                            G.add_node(cleaned_ip, type="IP_Address", timestamp=splitted_line[0])
                        if not G.has_edge(domain_name, cleaned_ip):
                            G.add_edge(domain_name, cleaned_ip, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0])

        if len(splitted_line[3]) > 0:
            # create the current line process
            pid = splitted_line[3]
            program_name = splitted_line[5]
            node_name = program_name + "_" + pid
            if len(program_name) == 0 or len(pid) == 0:
                if len(pid) == 0:
                    pid = "NOPID"
                if len(program_name) == 0:
                    program_name = "NOPROCESSNAME"
                node_name = program_name + "_" + pid
            else:
                processes[pid] = program_name
            node_name = str(node_name)

            if program_name.startswith("/device/harddiskvolume1"):
                program_name = program_name.replace("/device/harddiskvolume1", "c:")
                node_name = node_name.replace("/device/harddiskvolume1", "c:")
            
            if not G.has_node(node_name) and not node_name == "NOPROCESSNAME" and not node_name == "NOPROCESSNAME_NOPID":
                #print node_name
                if int(splitted_line[0]) >= StartTime:
                    G.add_node(node_name, type="process", timestamp=splitted_line[0])
                    if program_name.endswith("/") and not program_name.endswith("//"):
                        program_name = program_name[:len(program_name)-1] + "//"
                    if not program_name == "NOPROGRAMNAME":
                        program_name = program_name.rstrip()
                        if not G.has_node(program_name):
                            G.add_node(program_name, type="file", timestamp=splitted_line[0])
                        if IncludeExecutedEdges:
                            edge_type = "executed"
                            edge_label = edge_type + "_" + str(0)
                            G.add_edge(node_name, program_name, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0])
            
            # create a direct edge from parent to current line process
            if len(splitted_line[4]) > 0:
                parent_node_name = ""
                parent_pid = splitted_line[4]
                parent_name = ""
                if parent_pid in processes.keys():
                    parent_name = processes[parent_pid]
                else:
                    parent_name = "NOPROCESSNAME"
                parent_node_name = parent_name + "_" + parent_pid
                parent_node_name = str(parent_node_name)
                if parent_node_name.startswith("/device/harddiskvolume1"):
                    parent_name = parent_name.replace("/device/harddiskvolume1", "c:")
                    parent_node_name = parent_node_name.replace("/device/harddiskvolume1", "c:")
                
                if not G.has_node(parent_node_name) and not parent_node_name == "NOPROCESSNAME" and not parent_node_name == "NOPROCESSNAME_NOPID":
                    if int(splitted_line[0]) >= StartTime:
                        G.add_node(parent_node_name, type="process", timestamp=splitted_line[0])
                        if not parent_name == "NOPROCESSNAME":
                            if not G.has_node(parent_name):
                                if parent_name.endswith("/"):
                                    parent_name = parent_name[:len(parent_name)-1] + "//"
                                G.add_node(parent_name, type="file", timestamp=splitted_line[0])
                            if IncludeExecutedEdges:
                                edge_type = "executed"
                                edge_label = edge_type + "_" + str(0)
                                G.add_edge(parent_node_name, parent_name, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0])
                
                edge_type = "fork"
                edge_label = edge_type + "_" + str(splitted_line[0])
                if int(splitted_line[0]) >= StartTime:
                    if not G.has_edge(node_name, parent_node_name): # if not parent_node_name in G.successors(node_name)
                        G.add_edge(node_name, parent_node_name, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0])
                    else:
                        ALREADY_ADDED = False
                        for e in G.edges(node_name, data=True):
                            if e[2]['label'].startswith(edge_type):
                                ALREADY_ADDED = True
                                break
                        if not ALREADY_ADDED:
                            G.add_edge(node_name, parent_node_name, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0])

            if len(splitted_line[8]) > 0:
                d_ip = splitted_line[8]
                d_port = str(0)

                if len(splitted_line[9]) > 0:
                    d_port = splitted_line[9]

                d_ip = d_ip # .replace(":", "_")

                s_ip = splitted_line[6]
                s_port = str(0)

                if len(splitted_line[7]) > 0:
                    s_port = splitted_line[7]

                s_ip = s_ip # .replace(":", "_")

                joint_ips = ""
                joint_ips1 = s_ip + "_" + d_ip
                joint_ips2 = d_ip + "_" + s_ip

                if not G.has_node(joint_ips1) and not G.has_node(joint_ips2):
                    if int(splitted_line[0]) >= StartTime:
                        joint_ips = "connection_" + joint_ips1
                        G.add_node(joint_ips, type="connection", timestamp=splitted_line[0])
                else:
                    if G.has_node(joint_ips1):
                        if int(splitted_line[0]) >= StartTime:
                            joint_ips = joint_ips1
                    else:
                        if int(splitted_line[0]) >= StartTime:
                            joint_ips = joint_ips2

                if not G.has_node(s_ip):
                    if int(splitted_line[0]) >= StartTime:
                        G.add_node(s_ip, type="IP_Address", timestamp=splitted_line[0])
                if not G.has_node(d_ip):
                    if int(splitted_line[0]) >= StartTime:
                        G.add_node(d_ip, type="IP_Address", timestamp=splitted_line[0])

                # this block is to connect the remote IP to process, joint_ips connection and local ports
                edge_type = "connected_remote_ip"
                edge_label = edge_type + "_" + str(splitted_line[0])
                if int(splitted_line[0]) >= StartTime:
                    if s_ip == hosts_ips[0]: #if s_ip == "0.0.0.0" or s_ip == "127.0.0.1" or 
                        if not G.has_edge(d_ip, node_name): # .encode('unicode_escape')
                            G.add_edge(d_ip, node_name, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0], ip=d_ip)
                        if not G.has_edge(d_ip, joint_ips): # .encode('unicode_escape')
                            G.add_edge(d_ip, joint_ips, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0], ip=d_ip)
                    elif d_ip == hosts_ips[0]:
                        if not G.has_edge(s_ip, node_name): # .encode('unicode_escape')
                            G.add_edge(s_ip, node_name, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0], ip=s_ip)
                        if not G.has_edge(s_ip, joint_ips): # .encode('unicode_escape')
                            G.add_edge(s_ip, joint_ips, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0], ip=s_ip)

                    '''
                    else:
                        ALREADY_ADDED = False
                        for e in G.edges(s_ip, data=True):
                            if e[2]['type'] == edge_type and e[2]['sip'] == s_ip and e[2]['sport'] == s_port:
                                ALREADY_ADDED = True
                                break
                        if not ALREADY_ADDED:
                            G.add_edge(s_ip, joint_ips, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0], sip=s_ip, sport=s_port)
                    '''

                    '''
                    else:
                        ALREADY_ADDED = False
                        for e in G.edges(d_ip, data=True):
                            if e[2]['type'] == edge_type and e[2]['dip'] == d_ip and e[2]['dport'] == d_port:
                                ALREADY_ADDED = True
                                break
                        if not ALREADY_ADDED:
                            G.add_edge(d_ip, joint_ips, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0], dip=d_ip, dport=d_port)
                    '''

                edge_type = "connect"
                edge_label = edge_type + "_" + str(splitted_line[0])
                if int(splitted_line[0]) >= StartTime:
                    if not G.has_edge(joint_ips, node_name): # .encode('unicode_escape')
                        G.add_edge(joint_ips, node_name, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0], sip=s_ip, sport=s_port, dip=d_ip, dport=d_port)
                    else:
                        ALREADY_ADDED = False
                        for e in G.edges(joint_ips, data=True):
                            if e[2]['type'] == edge_type and e[2]['sip'] == s_ip and e[2]['sport'] == s_port and e[2]['dip'] == d_ip and e[2]['dport'] == d_port:
                                ALREADY_ADDED = True
                                break
                        if not ALREADY_ADDED:
                            G.add_edge(joint_ips, node_name, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0], sip=s_ip, sport=s_port, dip=d_ip, dport=d_port)
                
                edge_type = "sock_send"
                edge_label = edge_type + "_" + str(splitted_line[0])
                sender = "session_"+s_ip+"_"+s_port
                if not G.has_node(sender):
                    if int(splitted_line[0]) >= StartTime:
                        G.add_node(sender, type="session", timestamp=splitted_line[0], ip=s_ip, port=s_port)
                
                receiver = "session_"+d_ip+"_"+d_port
                if not G.has_node(receiver):
                    if int(splitted_line[0]) >= StartTime:
                        G.add_node(receiver, type="session", timestamp=splitted_line[0], ip=d_ip, port=d_port)

                if not G.has_edge(receiver, sender): # .encode('unicode_escape')
                    G.add_edge(receiver, sender, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0], sip=s_ip, sport=s_port, dip=d_ip, dport=d_port)
                
                edge_type = "bind"
                edge_label = edge_type + "_" + str(splitted_line[0])

                if s_ip == hosts_ips[0]: #s_ip == "0.0.0.0" or s_ip == "127.0.0.1" or 
                    if not G.has_edge(sender, node_name): # .encode('unicode_escape')
                        G.add_edge(sender, node_name, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0], ip=s_ip, port=s_port)
                    edge_type = "connected_session"
                    edge_label = edge_type + "_" + str(splitted_line[0])
                    if not G.has_edge(d_ip, sender): # .encode('unicode_escape')
                        G.add_edge(d_ip, sender, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0], ip=s_ip, port=s_port)
                elif d_ip == hosts_ips[0]:
                    if not G.has_edge(receiver, node_name): # .encode('unicode_escape')
                        G.add_edge(receiver, node_name, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0], ip=d_ip, port=d_port)
                    edge_type = "connected_session"
                    edge_label = edge_type + "_" + str(splitted_line[0])
                    if not G.has_edge(s_ip, receiver): # .encode('unicode_escape')
                        G.add_edge(s_ip, receiver, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0], ip=d_ip, port=d_port)

            if len(splitted_line[17]) > 0 and splitted_line[17].startswith("file_") and len(splitted_line[18]) > 0:
                accesses = splitted_line[17].rstrip()
                file_name = splitted_line[18].rstrip()

                if int(splitted_line[0]) >= StartTime:
                    if not G.has_node(file_name):
                        if file_name.endswith("/") and not file_name.endswith("//"):
                            file_name = file_name[:len(file_name)-1] + "//"
                        G.add_node(file_name, type="file", timestamp=splitted_line[0])

                for edge_type in ["readdata", "write", "delete", "execute"]: #"readdata", "writedata"
                    src_node = file_name
                    dst_node = node_name
                    if edge_type in accesses and not "attribute" in accesses: 
                        if edge_type == "readdata":
                            edge_type = "read"
                        if edge_type == "write":
                            edge_type = "write"
                        edge_label = edge_type + "_" + str(splitted_line[0])

                        #"execute" is not like fork, it is more like read, as it goes for every
                        #module gets executed under every process that executes that module.
                        if edge_type == "read" or edge_type == "execute": # 
                            src_node = node_name
                            dst_node = file_name
                        if int(splitted_line[0]) >= StartTime:
                            if not G.has_edge(src_node, dst_node): # .encode('unicode_escape')
                                G.add_edge(src_node, dst_node, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0])
                            else:
                                ALREADY_ADDED = False
                                for e in G.edges(src_node, data=True):
                                    if e[2]['label'].startswith(edge_type):
                                        ALREADY_ADDED = True
                                        break
                                if not ALREADY_ADDED:
                                    G.add_edge(src_node, dst_node, capacity=1.0, label=edge_label, type=edge_type , timestamp=splitted_line[0])
                        
                        if edge_type == "write":
                            downloaded_file_name = file_name

    # print("nodes: " + str(len(G.nodes())))
    # print("edges: " + str(len(G.edges())))

    # for node, data in G.nodes(data=True):
    #     print(node)
    #     print(data)
    
    # for u,v,data in G.edges(data=True):
    #     print(u)
    #     print(v)
    #     print(data)
        
    # with open('/Users/zhanghangsheng/Documents/my_code/attack-analysis/ATLAS/paper_experiments/S1/output_test/edges.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     # 写入 CSV 文件的标题行
    #     writer.writerow(['src_name', 'dst_name', 'type','timestamp'])
    #     for u,v,data in G.edges(data=True):
    #         src_name=u
    #         dst_name=v
    #         type=data['type']
    #         timestamp=data['timestamp']
    #         writer.writerow([src_name, dst_name, type, timestamp])
        
        
    # 打开 CSV 文件以写入节点数据
    # with open('/Users/zhanghangsheng/Documents/my_code/attack-analysis/ATLAS/paper_experiments/S1/output_test/nodes.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     # 写入 CSV 文件的标题行
    #     writer.writerow(['node_name', 'node_type', 'timestamp'])
    #     # 遍历图中的所有节点，并将它们写入 CSV 文件中
    #     for node,data in G.nodes(data=True):
    #         node_name=node
    #         node_type = data['type']
    #         timestamp = data['timestamp']
    #         writer.writerow([node_name, node_type, timestamp])
            
    # 写入边到数据
    
    return G

if __name__ == "__main__":
    folder = 'ATLAS/paper_experiments/S1/output/'
    # log_file = 'testing_preprocessed_logs_zhs'
    log_file = 'testing_preprocessed_logs_S1-CVE-2015-5122_windows'
    
    process_name=[]
    
    G=get_ATLAS_G(folder+log_file)
    search_id='connection_192.168.223.3_192.168.223.255'
    sub_G=nx.ego_graph(G,search_id,radius=2)
    
    from pyvis.network import Network
    
    net = Network(
        notebook=True,
        directed=True,
        cdn_resources="remote",
        height="1000px",
        width="100%",
        select_menu=True,
        filter_menu=True,
    )
    
    net.show_buttons(filter_=['physics'])
    
    for node in sub_G.nodes:
        net.add_node(node)
    for edge in sub_G.edges:
        net.add_edge(edge[0], edge[1])
        
    # net.barnes_hut()
    net.force_atlas_2based()
    # Display the network in the browser
    net.show('/Users/zhanghangsheng/Documents/my_code/attack-analysis/data_result/atlas/'+ search_id.rsplit('/', 1)[-1]+".html")
