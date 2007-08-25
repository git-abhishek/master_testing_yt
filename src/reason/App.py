from yt.reason import *

class ReasonMainWindow(wx.Frame):
    def __init__(self, *args, **kwds):
        kwds["style"] = wx.DEFAULT_FRAME_STYLE
        kwds["title"] = "yt - Reason"
        kwds["size"] = (700,700)
        wx.Frame.__init__(self, *args, **kwds)

        self.windows = []
        self.outputs = []
        self.dataObjects = []
        self.locals = {'lagos':lagos,
                       'raven':raven,
                       'enki':enki,
                       'raven':raven,
                       'outputs':self.outputs,
                       'windows':self.windows,
                       'mainwindow':self,
                       'dataObjects':self.dataObjects}

        self.mainSplitter = wx.SplitterWindow(self, -1)
        self.viewPanel = wx.SplitterWindow(self.mainSplitter, -1)
        self.dataPanel = wx.Panel(self.viewPanel, -1)
        self.intPanel = wx.Panel(self.mainSplitter, -1)
        self.interpreter = ReasonInterpreterPanel(self.intPanel, -1, self.locals)
        self.dataList = wx.TreeCtrl(self.dataPanel, -1, style=wx.TR_HIDE_ROOT |
                                    wx.TR_LINES_AT_ROOT | wx.TR_HAS_BUTTONS)
        self.plotPanel = PlotPanel(parent=self.viewPanel)
        self.SliceButton = wx.Button(self.dataPanel, -1, "Slice")
        self.ProjectButton = wx.Button(self.dataPanel, -1, "Project")
        #self.button_2 = wx.Button(self.dataPanel, -1, "button_2")
        #self.button_3 = wx.Button(self.dataPanel, -1, "button_3")

        self.mainSplitter.SetMinimumPaneSize(20)
        self.mainSplitter.SplitHorizontally(self.viewPanel, self.intPanel, -100)
        self.viewPanel.SetMinimumPaneSize(20)
        self.viewPanel.SplitVertically(self.dataPanel, self.plotPanel, 200)

        self.SetupMenubar()
        self.SetupToolBar()
        self.SetupDataTree()
        
        self.statusBar = self.CreateStatusBar(4, 0)

        self.__set_properties()
        self.__do_layout()

        self.Bind(wx.EVT_BUTTON, self.AddSlice, self.SliceButton)
        self.Bind(wx.EVT_BUTTON, self.AddProj, self.ProjectButton)

        #self.AddStaticOutputFile("/Users/matthewturk/Research/data/DataDump0012.dir/DataDump0012.hierarchy")
        self.AddStaticOutputFile("/Users/matthewturk/Research/data/galaxy0398.dir/galaxy0398.hierarchy")
        # end wxGlade

    def __set_properties(self):
        # begin wxGlade: ReasonMainWindow.__set_properties
        #self.dataList.SetMinSize((300,300))
        self.toolbar.SetToolBitmapSize((24, 24))
        self.toolbar.Realize()
        self.statusBar.SetStatusWidths([-1,-1,-1,-1])
        # statusbar fields
        #statusBar_fields = ["statusBar"]
        #for i in range(len(statusBar_fields)):
            #self.statusBar.SetStatusText(statusBar_fields[i], i)
        # end wxGlade

    def __do_layout(self):
        MainWindowSizer = wx.BoxSizer(wx.VERTICAL)
        ButtonSizer = wx.BoxSizer(wx.HORIZONTAL)
        DataPanelSizer = wx.BoxSizer(wx.VERTICAL)

        IntPanelSizer = wx.BoxSizer(wx.HORIZONTAL)
        IntPanelSizer.Add(self.interpreter, 1, wx.EXPAND, 0)
        self.intPanel.SetSizer(IntPanelSizer)

        ButtonSizer.AddSpacer(10)
        ButtonSizer.Add(self.SliceButton, 0, wx.ALIGN_CENTER, 0)
        ButtonSizer.AddSpacer(10)
        ButtonSizer.Add(self.ProjectButton, 0, wx.ALIGN_CENTER, 0)
        ButtonSizer.AddSpacer(10)
        DataPanelSizer.Add(self.dataList, 1, wx.EXPAND, 0)
        DataPanelSizer.AddSpacer(5)
        DataPanelSizer.Add(ButtonSizer, 0, 0, 0)
        DataPanelSizer.AddSpacer(5)
        self.dataPanel.SetSizer(DataPanelSizer)
        self.dataPanel.Layout()

        MainWindowSizer.Add(self.mainSplitter, 1, wx.EXPAND)
        self.SetSizer(MainWindowSizer)

        self.Layout()

    def SetupMenubar(self):
        menuBar = wx.MenuBar()
        fileMenu = wx.Menu()
        menuBar.Append(fileMenu, "File")
        
        # Set up IDs for event binding

        openHierarchy = fileMenu.Append(-1, "Open Hierarchy")
        saveImage = fileMenu.Append(-1, "Save Image")
        fileMenu.AppendSeparator()
        exit = fileMenu.Append(-1, "Exit")

        self.Bind(wx.EVT_MENU, self.OnOpenHierarchy, openHierarchy)
        self.Bind(wx.EVT_MENU, self.OnSaveImage, saveImage)
        self.Bind(wx.EVT_MENU, self.OnExit, exit)

        self.SetMenuBar(menuBar)

    def SetupToolBar(self):
        # Tool Bar
        self._VMTB_FULLDOMAIN = wx.NewId()
        self._VMTB_CHANGEZOOM = wx.NewId()
        self._VMTB_REDRAW = wx.NewId()
        self._VMTB_SAVE = wx.NewId()
        self._VMTB_FIELDSWITCH = wx.NewId()
        self._VMTB_CHANGELIMITS = wx.NewId()
        self._VMTB_VIEWPF = wx.NewId()
        self._VMTB_VELPLOT = wx.NewId()

        self.toolbar = wx.ToolBar(self, -1, style=wx.TB_HORIZONTAL|wx.TB_TEXT|wx.TB_HORZ_LAYOUT)
        font = self.toolbar.GetFont()
        font.SetFamily(wx.MODERN)
        self.toolbar.SetFont(font)
        self.toolbar.SetToolBitmapSize((16,16))

        self.SetToolBar(self.toolbar)
        def AddButton(id, label, tooltip="", bitmapID=None):
            if bitmapID != None:
                bm = wx.ArtProvider.GetBitmap(bitmapID, wx.ART_TOOLBAR, (16,16))
            else: bm = wx.NullBitmap
            self.toolbar.AddLabelTool(id, label, bm, bm, wx.ITEM_NORMAL, tooltip, "")
            self.toolbar.AddSeparator()

        self.toolbar.AddSeparator()
        AddButton(self._VMTB_REDRAW,"Redraw", "Force a redraw", wx.ART_REDO)
        #AddButton(self._VMTB_FIELDSWITCH,"Change Field", "Change the displayed field")
        self.availableFields = wx.Choice(self.toolbar, id=self._VMTB_FIELDSWITCH, choices = [])
        self.toolbar.AddControl(self.availableFields)
        Publisher().subscribe(self.UpdateToolbarFieldsMessage, ('page_changed'))
        #AddButton(self._VMTB_CHANGEZOOM, "Change Width",  "Change the displayed width") # unneeded
        AddButton(self._VMTB_FULLDOMAIN, "Zoom Top",  "Zoom to the top level", wx.ART_FIND)
        AddButton(self._VMTB_CHANGELIMITS, "Change Limits", "Change the colorbar limits")
        AddButton(self._VMTB_VIEWPF, "View ParameterFile", "View the parameter file", wx.ART_NORMAL_FILE)
        cl = wx.ArtProvider.GetBitmap(wx.ART_TICK_MARK, wx.ART_TOOLBAR, (16,16))
        self.toolbar.AddCheckLabelTool(self._VMTB_VELPLOT, "VelVecs", cl, shortHelp="Plot Velocity Vectors")
        self.toolbar.AddSeparator()

        self.Bind(wx.EVT_CHOICE, self.plotPanel.OnCallSwitchField, id=self._VMTB_FIELDSWITCH)
        self.Bind(wx.EVT_MENU, self.plotPanel.OnCallSetWidth, id=self._VMTB_CHANGEZOOM)
        self.Bind(wx.EVT_MENU, self.plotPanel.OnCallRedraw, id=self._VMTB_REDRAW)
        self.Bind(wx.EVT_MENU, self.plotPanel.OnCallZoomTop, id=self._VMTB_FULLDOMAIN)
        self.Bind(wx.EVT_MENU, self.plotPanel.OnCallSetZLim, id=self._VMTB_CHANGELIMITS)
        self.Bind(wx.EVT_MENU, self.plotPanel.OnCallViewPF, id=self._VMTB_VIEWPF)

    def UpdateToolbarFieldsMessage(self, message):
        page = message.data
        self.UpdateToolbarFields(page)

    def UpdateToolbarFields(self, page):
        self.availableFields.SetItems(page.QueryFields())

    def SetupDataTree(self):

        self.root = self.dataList.AddRoot("You shouldn't see me!")
        self.fidoRoot = self.dataList.AppendItem(self.root, "Stored Outputs")
        self.outputRoot = self.dataList.AppendItem(self.root, "Selected Outputs")
        self.dataRoot = self.dataList.AppendItem(self.root, "Data Objects")

        self.dataList.Expand(self.fidoRoot)
        self.dataList.Expand(self.outputRoot)
        self.dataList.Expand(self.dataRoot)

    def OnExit(self, event):
        self.Close()

    def OnOpenHierarchy(self, event):
        wildcard = "Hierarchy (*.hierarchy)|*.hierarchy|" \
                   "All files (*,*)|*.*"
        dialog = wx.FileDialog(None, "Choose your hierarchy", os.getcwd(),
                               "", wildcard, wx.OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            file = dialog.GetPath()
            print file
            self.AddStaticOutputFile(file)
            #self.RefreshOutputs()
        dialog.Destroy()

    def OnSaveImage(self, event):
        print "Getting thingie"
        pgI = self.plotPanel.nb.Selection
        print "Got thingie"
        pg = self.plotPanel.nb.GetPage(pgI)
        print "Augh"
        if not hasattr(pg, 'SaveImage'): return
        print "Calling save"
        pg.SaveImage()

    def AddStaticOutputFile(self, filename):
        # Alright, we choose the hierarchy in the file selector,
        # so let's strip that extension off
        fn = filename[:-10]
        eso = lagos.EnzoStaticOutput(fn)
        self.outputs.append(eso)
        try:
            z = str(eso["CosmologyCurrentRedshift"])
        except:
            z = "N/A"
        tid = wx.TreeItemData((eso, str(eso["InitialTime"]), z, len(self.outputs)))
        ni = self.dataList.AppendItem(self.outputRoot, "%s" % (eso.basename), data=tid)
        self.dataList.Expand(self.outputRoot)

    def AddDataObject(self, title, object):
        self.dataObjects.append(object)
        tid = wx.TreeItemData((object, title, "", len(self.dataObjects)))
        ni = self.dataList.AppendItem(self.dataRoot, "%s" % (title), data=tid)
        self.dataList.Expand(self.dataRoot)

    def AddSphere(self, title, sphere):
        # These all get passed in
        self.AddDataObject(title, sphere)

    def AddProj(self, event=None):
        #MyID = wx.NewId()
        self.interpreter.shell.write("\n")
        for o in self.GetOutputs():
            MyID = o["CurrentTimeIdentifier"]
            #field = Toolbars.ChooseField(o)
            field = "Density"
            if not field:
                continue
            #width, unit = Toolbars.ChooseWidth(o)
            width = 1.0
            unit = "1"
            for i, ax in zip(range(3), 'xyz'):
                t = "%s - Projection - %s" % (o.basename, ax)
                self.windows.append( \
                    ProjPlotPage(parent=self.plotPanel.nb, 
                                  statusBar=self.statusBar,
                                  outputfile = o,
                                  axis=i,
                                  field = field,
                                  mw = self))
                self.interpreter.shell.write("Adding %s slice of %s\n" % (ax, o))
                self.plotPanel.AddPlot(self.windows[-1], t, MyID)
                self.outputs.append(self.windows[-1].plot.data)

    def AddSlice(self, event=None):
        #MyID = wx.NewId()
        self.interpreter.shell.write("\n")
        for o in self.GetOutputs():
            MyID = o["CurrentTimeIdentifier"]
            #field = Toolbars.ChooseField(o)
            field = "Density"
            if not field:
                continue
            #width, unit = Toolbars.ChooseWidth(o)
            width = 1.0
            unit = "1"
            for i, ax in zip(range(3), 'xyz'):
                t = "%s - Slice - %s" % (o.basename, ax)
                self.windows.append( \
                    SlicePlotPage(parent=self.plotPanel.nb, 
                                  statusBar=self.statusBar,
                                  outputfile = o,
                                  axis=i,
                                  field = field,
                                  mw = self))
                self.interpreter.shell.write("Adding %s projection of %s\n" % (ax, o))
                self.outputs.append(self.windows[-1].plot.data)
                self.plotPanel.AddPlot(self.windows[-1], t, MyID)

    def GetOutputs(self, event=None):
        # Figure out which outputs are selected
        oss = []
        #k = self.dataList.GetFirstSelected()
        k = self.dataList.GetSelections()
        for tid in k:
            oss.append(self.dataList.GetItemData(tid).GetData()[0])
        return oss

class ReasonApp(wx.App):
    def OnInit(self):
        wx.InitAllImageHandlers()
        frame_1 = ReasonMainWindow(None, -1)
        self.SetTopWindow(frame_1)
        frame_1.Show()
        return True

class ReasonParameterFileViewer(wx.Frame):
    def __init__(self, *args, **kwds):
        kwds["style"] = wx.DEFAULT_FRAME_STYLE
        kwds["title"] = "yt - Reason"
        kwds["size"] = (800,800)
        pf = kwds.pop("outputfile")
        wx.Frame.__init__(self, *args, **kwds)

        # Add the text ctrl
        self.pf = wx.TextCtrl(self, -1, style=wx.TE_READONLY | wx.TE_MULTILINE | wx.HSCROLL)
        self.pf.LoadFile(pf.parameterFilename)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.pf, 1, wx.EXPAND, 0)
        self.SetSizer(self.sizer)
        self.sizer.Fit(self)
        self.Layout()
        self.SetSize((600,600))