"""
Handling of sets of outputs.

We don't do any instantiation, or touching of lagos, etc.

Very simple nowadays.

@author: U{Matthew Turk<http://www.stanford.edu/~mturk/>}
@organization: U{KIPAC<http://www-group.slac.stanford.edu/KIPAC/>}
@contact: U{mturk@slac.stanford.edu<mailto:mturk@slac.stanford.edu>}
"""

from yt.fido import *

class OutputCollection:
    def __init__(self, title):
        self.title = title
        self.outputNames = na.array(())
        self.outputTimeIDs = na.array((), dtype=nT.Int64)
        self.outputTimes = na.array((), dtype=nT.Float64)

    def readIn(self, filename):
        lines = open(filename).readlines()
        outputLines = filter(lambda a: a.startswith("Output:"), lines)
        outputs = map(lambda a: a.split(":")[1], outputLines)
        outputTimeIDs = map(lambda a: int(a.split(":")[2]), outputLines)
        outputTimes = map(lambda a: float(a.split(":")[3]), outputLines)

        self.outputNames = na.array(outputs)
        self.outputTimeIDs = na.array(outputTimeIDs, nT.Int64)
        self.outputTimes = na.array(outputTimes, nT.Float64)

    def writeOut(self, filename):
        f = open(filename, "w")
        for i, output in enumerate(self.outputNames):
            f.write("Output:%s:%s:%s\n" % ( \
                        os.path.abspath(output), \
                        self.outputTimeIDs[i], \
                        self.outputTimes[i]))
        f.close()

    def sortOutputs(self):
        order = na.argsort(self.outputTimes)
        self.outputNames = self.outputNames[order]
        self.outputTimes = self.outputTimes[order]
        self.outputTimeIDs = self.outputTimeIDs[order]

    def addOutput(self, filename):
        # We're passing in *just* filenames here.  So, we simply snag the 
        # two appropriate lines in the parameter file.
        time = getParameterLine(filename, "InitialTime").split()[-1]
        # Implement exception catching
        timeID = getParameterLine(filename, "CurrentTimeIdentifier").split()[-1]
        self.outputNames = \
            na.array(self.outputNames.tolist() + [filename])
        self.outputTimeIDs = \
            na.array(self.outputTimeIDs.tolist() + [int(timeID)], dtype=nT.Int64)
        self.outputTimes = \
            na.array(self.outputTimes.tolist() + [float(time)], dtype=nT.Float64)
        self.sortOutputs()

    def getBefore(self, time):
        return na.where(self.outputTimes <= time)[0]

    def getAfter(self, time):
        return na.where(self.outputTimes > time)[0]

    def __delitem__(self, key):
        id = self.index(key)
        self.outputNames = na.array(self.outputNames[:id].tolist() \
                                  + self.outputNames[id+1:].tolist())
        self.outputTimes = na.array(self.outputTimes[:id].tolist() \
                                  + self.outputTimes[id+1:].tolist())
        self.outputTimeIDs = na.array(self.outputTimeIDs[:id].tolist() \
                                  + self.outputTimeIDs[id+1:].tolist())

    def __getitem__(self, key):
        """
        Based on the type of input, we either return based on index or
        basename.
        """
        if isinstance(key, types.StringType):
            index = self.index(key)
        elif isinstance(key, types.IntType):
            index = key
        return self.outputs[index]

    def index(self, key):
        t = os.path.basename(key)
        # Find out the index
        index = None
        for i in range(self.outputNames.shape[0]):
            if self.outputNames[i] == t:
                index = i
                break
        if index == None:
            raise KeyError
        return index

    def __repr__(self):
        return self.title

    def __len__(self):
        return len(self.outputNames)

    def __iter__(self):
        for i in range(len(self)):
            yield self.outputNames[i]

    def keys(self):
        return self.outputNames.tolist()

    def exportAmira(self, basename, fields):
        a5 = basename + "-%(field)s.a5"
        sorted_times = []
        sorted_timesteps = []
        for field in fields:
            a5b=tables.openFile(a5 % {'field':field},"w")
            a5b.close()
        if (not iterable(fields)) or (isinstance(fields, types.StringType)):
            fields = [fields]
        for i in range(len(self)):
            mylog.info("Exporting %s to Amira format", self[i])
            bn = basename + "-%06i" % (i) + "-%(field)s.h5"
            numDone = 0
            self.promoteType(i)
            self[i].exportAmira(bn, fields, a5, i)
            rootDelta = na.array([self[i].grids[0].dx, self[i].grids[0].dy, self[i].grids[0].dz], dtype=nT.Float64)
            sorted_timesteps.append(i)
            sorted_times.append(self[i]["InitialTime"])
            self.demoteType(i)
            numDone += 1
        for field in fields:
            a5b=tables.openFile(a5 % {'field':field},"a")
            a5b.createGroup("/","globalMetaData")
            node=a5b.getNode("/","globalMetaData")
            node._f_setAttr("datatype",0)
            node._f_setAttr("fieldtype",1)
            node._f_setAttr("staggering",1)
            node._f_setAttr("maxTime",self.timesteps[-1])
            node._f_setAttr("maxTimeStep",len(self)-1)
            node._f_setAttr("minTime",self.timesteps[0])
            node._f_setAttr("minTimeStep",0)
            node._f_setAttr("numTimeSteps",len(self))
            node._f_setAttr("rootDelta",rootDelta)
            a5b.createArray("/","sorted_timesteps", na.array(sorted_timesteps, dtype=nT.Int32))
            a5b.createArray("/","sorted_times", na.array(sorted_times, dtype=nT.Float64))
            a5b.close()