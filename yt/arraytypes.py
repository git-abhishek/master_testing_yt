"""
We want to have flexible arrays, so we do it all in here, and then import from
this module.

This is all probably overly-complicated, and should be removed at first
opportunity to ditch numarray.

@author: U{Matthew Turk<http://www.stanford.edu/~mturk/>}
@organization: U{KIPAC<http://www-group.slac.stanford.edu/KIPAC/>}
@contact: U{mturk@slac.stanford.edu<mailto:mturk@slac.stanford.edu>}
@todo: Deprecate this.
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

import numpy as na
import numpy.core.records as rec
import scipy.ndimage as nd # Moved into scipy
import scipy as sp
import scipy.weave as weave
from scipy.weave import converters

# Now define convenience functions

def blankRecordArray(desc, elements):
    """
    Accept a descriptor describing a recordarray, and return one that's full of
    zeros

    This seems like it should be in the numpy distribution...
    """
    blanks = []
    for atype in desc['formats']:
        blanks.append(na.zeros(elements, dtype=atype))
    return rec.fromarrays(blanks, **desc)