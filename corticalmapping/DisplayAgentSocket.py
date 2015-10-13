# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 11:04:27 2013

@author: junz

modifiec from agent module written by derricw

agent.py

This TCP server waits on connections from remote computers and allows them
    to run scripts and custom stimulus/behavior programs.  See APIReference.txt
    for instructions.

"""
import SocketServer
import socket
from datetime import datetime
import subprocess
import os
import sys
from aibs.Core import *


def getdirectories():
    """ Gets platform specific default directories for logs, etc. """
    if 'linux' in sys.platform:
        topdir = os.path.expanduser('~/data/')
    elif 'darwin' in sys.platform:
        topdir = os.path.expanduser('~/data/')
    else:
        topdir = 'C:\\data\\'
    return topdir


os.chdir("C:/")  # prevents windows from locking cwd to enable self-restart

scriptlog = os.path.join(getdirectories(), "ScriptLog")
# Ensure directory exists...
if not os.path.isdir(scriptlog):
    os.makedirs(scriptlog)
    print "Creating new path:", scriptlog

try:
    topdir = getdirectories()  # from Core
    config = os.path.join(topdir, 'config/whitelist.cfg')
    WHITELIST = open(config, 'r').readlines()
except Exception, e:
    print "No config file found as", config, "... only accepting local connections.", e
    WHITELIST = ["127.0.0.1"]

RIG_IP = "localhost"  # rig is local machine
RIG_PORT = 10002  # for sending data
LISTEN_PORT = 10001  # for receiving data

RIGSOCKET = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
RIGSOCKET.bind((RIG_IP, LISTEN_PORT))
RIGSOCKET.settimeout(0.2)


class Agent(SocketServer.BaseRequestHandler):

    """

    A request handler to be passed to a TCPServer.
            The handle() method is overwritten to handle
            incoming connections.

            This handler accepts scripts if one is currently
            not being run, and rejects them otherwise.  It can
            be polled for its current status and it can kill
            the currently running script.

    """

    def handle(self):
        self.status = 0
        print WHITELIST
        if self.client_address[0] in WHITELIST:
            print "Incoming connection from: {}".format(self.client_address[0])

            while 1:
                try:
                    self.data = self.request.recv(8192)
                    if not self.data:
                        break
                    self.data = self.data.strip()
                    print self.data

                    if self.data[:6].upper() == "SCRIPT":
                        self.startScript()

                    elif self.data[:6].upper() == "STATUS":
                        self.pollStatus()

                    elif self.data[:4].upper() == "KILL":
                        self.killScript()

                    # added by Junz, to stop corticalmapping visual stimulus
                    elif self.data[:4].upper() =='QUIT':
                        self.quit_corticalmapping_stimuli()

                    elif self.data[:6].upper() == "REWARD":
                        self.triggerReward()

                    elif self.data[:6].upper() == "PUNISH":
                        self.triggerPunishment()

                    elif self.data[:3].upper() == "SET":
                        self.setValue()

                    elif self.data[:3].upper() == "GET":
                        self.getValue()

                    elif self.data[:3].upper() == "RUN":
                        self.runMethod()

                    elif self.data[:6].upper() == "UPDATE":
                        self.triggerUpdate()
                        break

                    self.request.sendall("0\n")
                except Exception, e:
                    print e
                    self.request.sendall("1:%s\n" % (e))
        else:
            print "Incoming address not in whitelist.  Closing socket."

    def startScript(self):
        """ Starts a script. """
        if self.status == 0:
            # Message is a script
            print "SCRIPT DETECTED"
            text = self.data[6:]
            name, script = text.split("|")  # TODO:WILL THIS EVER MATTER?
            print "Name", name, "\nScript:\n", script
            if len(name) < 4:
                name = datetime.now().strftime('%y%m%d%H%M%S')
            # Create full file name and save script
            path = os.path.join(scriptlog, name + '.py')
            with open(path, 'w') as f:
                f.write(script)
            print "Script saved as:", path
            print "Starting process..."

            # Create subprocess
            execstring = "python " + path
            self.sp = subprocess.Popen(execstring.split())
            print "Process started:", self.sp.pid
            self.status = 1
        else:
            print "Incoming script cannot be started."

    def killScript(self):
        """ Kills subprocess if one exists. """
        try:
            self.sp.kill()
            self.status = 0
        except Exception, e:
            print "Process kill failed:", e

    def quit_corticalmapping_stimuli(self):
        RIGSOCKET.sendto("STOP", (RIG_IP, RIG_PORT))
        self.status = 0

    def pollStatus(self):
        """ Returns status of rig. """
        ##TODO: I DONT LIKE THE WAY THE POLLING WORKS
        if self.status == 0:
            response = "READY\n"
        elif self.sp.poll() is None:
            response = "RUNNING\n"
            self.status = 1
        else:
            response = "READY\n"
            self.status = 0
        self.request.sendall(response)

    def triggerReward(self):
        """ Trigger reward in connected rig. """
        RIGSOCKET.sendto("REWARD", (RIG_IP, RIG_PORT))

    def triggerPunishment(self):
        """ Trigger punishment in connected rig. """
        RIGSOCKET.sendto("PUNISH", (RIG_IP, RIG_PORT))

    def getValue(self):
        """ Get value from connected rig. """
        command = self.data.split(" ")
        name_str = command[1]
        RIGSOCKET.sendto("GET %s" % (name_str), (RIG_IP, RIG_PORT))

    def setValue(self):
        """ Set value in connected rig. """
        command = self.data.split(" ")
        name_str = command[1]
        value = command[2]
        RIGSOCKET.sendto("SET %s %s" % (name_str, value), (RIG_IP, RIG_PORT))
        # Check to ensure that the command was received?

    def runMethod(self):
        """ Run specific method on connected rig. """
        command = self.data.split(" ")
        name_str = command[1]
        RIGSOCKET.sendto("RUN %s" % name_str, (RIG_IP, RIG_PORT))

    def triggerUpdate(self):
        """ Starts an update script. """
        if self.status == 0:
            try:
                print "REMOVING c:\stim.cfg"
                os.remove(r'c:\stim.cfg')
            except Exception, e:
                print "Couldn't remove stim.cfg:",e
            # Message is a script
            print "UPDATE SCRIPT DETECTED"
            script = self.data[6:]
            dt = datetime.now().strftime('%y%m%d%H%M%S')

        # Create full file name and save script
            path = os.path.join(scriptlog, dt + '.py')
            with open(path, 'w') as f:
                f.write(script)
            print "Script saved as:", path
            print "Starting process..."

            # Create subprocess
            execstring = "python " + path
            self.sp = subprocess.Popen(execstring.split(),
                                       close_fds=True)  # required for self-restart

            # Send response
            self.request.sendall("0\n")

            # Shutdown all the TCP bullshit
            self.server.socket.close()
            self.server.close_request(self.request)
            self.server.server_close()
            os._exit(0)

        else:
            print "Update script cannot be started."


def main():
    HOST, PORT = "", 10001
    agent = SocketServer.TCPServer((HOST, PORT), Agent)
    agent.allow_reuse_address = True
    print "Server started at:", agent.server_address

    agent.serve_forever()


if __name__ == '__main__':
    main()
