import wx
from src.train_tab import TrainTab
from src.generate_tab import GenerateTab
from src.settings import Settings

class MainFrame(wx.Frame):
    def __init__(self):
        self.settings = Settings()
        pos = self.settings.get_window_position()
        size = self.settings.get_window_size()
        super().__init__(None, title="Neural Network Roughness Generator", pos=wx.Point(pos[0], pos[1]), size=wx.Size(size[0], size[1]))

        self.CreateStatusBar(2)  # Create status bar with 2 fields
        self.status_bar = self.GetStatusBar()
        
        # Create progress gauge for the second field
        self.progress_gauge = wx.Gauge(self.status_bar, range=100, style=wx.GA_HORIZONTAL)
        self.progress_gauge.SetValue(0)
        
        # Position the progress gauge in the second field
        self.setup_status_bar_layout()
        
        self.notebook = wx.Notebook(self)

        self.train_tab = TrainTab(self.notebook, self, self.settings)
        self.generate_tab = GenerateTab(self.notebook, self, self.settings)

        self.notebook.AddPage(self.train_tab, "Train")
        self.notebook.AddPage(self.generate_tab, "Generate")

        # Load current tab from settings
        current_tab = self.settings.get_current_tab()
        if 0 <= current_tab < self.notebook.GetPageCount():
            self.notebook.SetSelection(current_tab)

        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.Bind(wx.EVT_SIZE, self.on_size)
        
        # Bind notebook page change event to save current tab
        self.notebook.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.on_tab_changed)

    def on_close(self, event):
        self.settings.set_window_size(self.GetSize())
        self.settings.set_window_position(self.GetPosition())
        self.settings.set_current_tab(self.notebook.GetSelection())
        self.settings.save()
        self.Destroy()

    def on_tab_changed(self, event):
        """Handle notebook tab change to save current tab"""
        current_tab = self.notebook.GetSelection()
        self.settings.set_current_tab(current_tab)
        self.settings.save()
        event.Skip()

    def hide_main_window(self):
        """Hide the main window during training"""
        self.Hide()

    def show_main_window(self):
        """Show the main window after training"""
        self.Show()
        self.Raise()  # Bring to front

    def on_size(self, event):
        """Handle window resize to reposition status bar elements"""
        if hasattr(self, 'progress_gauge') and hasattr(self, 'status_bar'):
            wx.CallAfter(self.setup_status_bar_layout)
        event.Skip()

    def setup_status_bar_layout(self):
        """Setup the status bar layout with text and progress gauge"""
        # Set the widths of the status bar fields
        # Field 0: Text (flexible), Field 1: Progress bar (fixed width)
        self.status_bar.SetStatusWidths([-1, 200])  # -1 means flexible, 200 is fixed width for progress
        
        # Position the progress gauge in the second field
        rect = self.status_bar.GetFieldRect(1)
        self.progress_gauge.SetPosition((rect.x + 2, rect.y + 2))
        self.progress_gauge.SetSize((rect.width - 4, rect.height - 4))
        
        # Initially hide the progress gauge
        self.progress_gauge.Hide()

    def set_progress(self, value, show=True):
        """Set progress bar value (0-100) and optionally show/hide it"""
        if show and not self.progress_gauge.IsShown():
            self.progress_gauge.Show()
        elif not show and self.progress_gauge.IsShown():
            self.progress_gauge.Hide()
        
        self.progress_gauge.SetValue(min(100, max(0, value)))
        self.status_bar.Refresh()

    def hide_progress(self):
        """Hide the progress bar"""
        self.set_progress(0, show=False)
