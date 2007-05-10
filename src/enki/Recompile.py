"""
Here's our source-tree handler.  For now we will dump to shell to recompile.

@author: U{Matthew Turk<http://www.stanford.edu/~mturk/>}
@organization: U{KIPAC<http://www-group.slac.stanford.edu/KIPAC/>}
@contact: U{mturk@slac.stanford.edu<mailto:mturk@slac.stanford.edu>}
"""

from yt.enki import *
import os, commands, re, os.path

def compileNew(MachineName=None, Clean=False, nProc=1, target=None, 
               srcDir="src", exeDir="exe"):
    """
    We get the machine name, starting with the argument, then the environment,
    then falling back to config file, and then just not assuming anything.

    @keyword MachineName: the MACHINE_NAME variable to pass to make
    @type MachineName: string
    @keyword Clean: should we clean the output files first?
    @type Clean: boolean
    @keyword nProc: number of jobs to spawn
    @type nProc: int
    @keyword target: the target to make
    @type target: string
    @keyword srcDir: the subdirectory where we find the source
    @type srcDir: string
    @keyword exeDir: the place where executables get stuck
    @type exeDir: string
    """
    old_wd = os.getcwd()
    wd = ytcfg.get("yt","enzo_source_tree")
    os.chdir(os.path.join(wd,srcDir))
    if MachineName == None:
        if os.environ.has_key("ENZO_MACHINE_NAME"):
            MachineName = os.environ["ENZO_MACHINE_NAME"]
        elif ytcfg.has_option("yt","ENZO_MACHINE_NAME"):
            MachineName = ytcfg.get("yt","ENZO_MACHINE_NAME")
    command = "make -j%s" % (nProc)
    if MachineName:
        command += " MACHINE_NAME=%s" % (MachineName)
    if Clean:
        c = command + " clean"
        cl=commands.getoutput(c)
    if target:
        command += " %s" % (target)
    # Now we get the source tree
    if not ytcfg.has_option("yt","enzo_source_tree"):
        mylog.error("Section 'yt' option 'enzo_source_tree' not set in yt config file!")
        mylog.error("Returning without compiling.")
    mylog.info("Executing %s", command)
    st=commands.getoutput(command)
    os.chdir(old_wd)
    if target:
        return st # We return the entire output if there's a target
    #return st
    # Now we should parse it to figure out what was created
    r=re.compile(r"^Created (.*$)", re.M)
    m=r.search(st)
    exePath = os.path.join(os.path.join(wd,exeDir),m.group(1))
    mylog.info("Received %s", exePath)
    return exePath