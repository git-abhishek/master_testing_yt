"""
Initializer for reason.

@author: U{Matthew Turk<http://www.stanford.edu/~mturk/>}
@organization: U{KIPAC<http://www-group.slac.stanford.edu/KIPAC/>}
@contact: U{mturk@slac.stanford.edu<mailto:mturk@slac.stanford.edu>}
@license:
  Copyright (C) 2007 Matthew Turk.  All Rights Reserved.

  This file is part of yt.

  yt is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


from yt.config import ytcfg
from yt.logger import lagosLogger as mylog

import yt.lagos as lagos
import yt.raven as raven
import yt.enki as enki
import yt.fido as fido

from yt.arraytypes import *

from math import log10, sqrt

import os, types, Toolbars

import yt.raven.backends.MPL as be
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_agg import FigureCanvasAgg
import wx, wx.py, wx.aui
from wx.lib.pubsub import Publisher
import matplotlib.backends.backend_wx as be_wx
import matplotlib.figure

from Windows import *
from Notebook import *
from App import *