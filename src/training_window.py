import wx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class TrainingProgressWindow(wx.Frame):
    """Window to show training progress with live charts"""
    
    def __init__(self, parent, training_manager=None, title="Training Progress"):
        super().__init__(parent, title=title, size=(800, 600))
        
        self.training_manager = training_manager
        self.cancelled = False
        self.training_finished = False
        self.epoch_data = []
        self.loss_data = {
            'generator': [],
            'discriminator': [],
            'g_gan': [],
            'g_l1': []
        }
        
        self.setup_ui()
        self.Center()
        
        # Bind close event
        self.Bind(wx.EVT_CLOSE, self.on_close)
    
    def setup_ui(self):
        """Setup the user interface"""
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Progress info
        info_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.epoch_label = wx.StaticText(panel, label="Epoch: 0/0")
        self.epoch_label.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        info_sizer.Add(self.epoch_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        
        info_sizer.AddStretchSpacer()
        
        self.time_label = wx.StaticText(panel, label="Time: 00:00:00")
        info_sizer.Add(self.time_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        
        info_sizer.AddStretchSpacer()

        self.memory_label = wx.StaticText(panel, label="VRAM: 0 MB")
        info_sizer.Add(self.memory_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        sizer.Add(info_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Progress bar
        self.progress_bar = wx.Gauge(panel, range=100, style=wx.GA_HORIZONTAL)
        sizer.Add(self.progress_bar, 0, wx.EXPAND | wx.ALL, 5)
        
        # Status text
        self.status_text = wx.StaticText(panel, label="Initializing...")
        sizer.Add(self.status_text, 0, wx.ALL, 5)
        
        # Matplotlib figure
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(panel, -1, self.figure)
        sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 5)
        
        # Setup subplots
        self.ax1 = self.figure.add_subplot(121)
        self.ax2 = self.figure.add_subplot(122)
        
        self.ax1.set_title('Generator Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.grid(True)
        
        self.ax2.set_title('Discriminator Loss')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Loss')
        self.ax2.grid(True)
        
        panel.SetSizer(sizer)
    
    def update_progress(self, epoch, total_epochs, batch, total_batches, losses, elapsed_time, memory_allocated):
        """Update progress information"""
        # Update labels
        self.epoch_label.SetLabel(f"Epoch: {epoch}/{total_epochs}")
        
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        self.time_label.SetLabel(f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Update memory usage
        self.memory_label.SetLabel(f"VRAM: {memory_allocated:.2f} MB")

        # Update progress bar
        if total_epochs > 0:
            progress = int((epoch / total_epochs) * 100)
            self.progress_bar.SetValue(progress)
        
        # Update status
        self.status_text.SetLabel(f"Batch {batch}/{total_batches} - G Loss: {losses['loss_g']:.4f}, D Loss: {losses['loss_d']:.4f}")
        
        # Store data for plotting
        self.epoch_data.append(epoch + batch / total_batches)
        self.loss_data['generator'].append(losses['loss_g'])
        self.loss_data['discriminator'].append(losses['loss_d'])
        self.loss_data['g_gan'].append(losses['loss_g_gan'])
        self.loss_data['g_l1'].append(losses['loss_g_l1'])
        
        # Update plots every 10 iterations to avoid too frequent updates
        if len(self.epoch_data) % 10 == 0:
            self.update_plots()
    
    def update_plots(self):
        """Update the loss plots"""
        if not self.epoch_data:
            return
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot generator losses
        self.ax1.plot(self.epoch_data, self.loss_data['generator'], 'b-', label='Total', linewidth=2)
        self.ax1.plot(self.epoch_data, self.loss_data['g_gan'], 'r--', label='GAN', alpha=0.7)
        self.ax1.plot(self.epoch_data, self.loss_data['g_l1'], 'g--', label='L1', alpha=0.7)
        self.ax1.set_title('Generator Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Plot discriminator loss
        self.ax2.plot(self.epoch_data, self.loss_data['discriminator'], 'orange', linewidth=2)
        self.ax2.set_title('Discriminator Loss')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Loss')
        self.ax2.grid(True, alpha=0.3)
        
        # Refresh canvas
        self.canvas.draw()
    
    def on_save_model(self, event):
        """Handle save model button click"""
        # This would trigger a save in the training loop
        wx.PostEvent(self.GetParent(), wx.PyCommandEvent(wx.EVT_BUTTON.typeId, self.GetId()))
    
    def on_close(self, event):
        """Handle window close event"""
        if self.training_finished or self.cancelled:
            # Show the main window when training window is closed
            if self.training_manager and self.training_manager.parent_frame:
                self.training_manager.parent_frame.show_main_window()
            self.Destroy()
            return

        dlg = wx.MessageDialog(self, "Training is still in progress. Cancel training and close?", 
                              "Close Window", wx.YES_NO | wx.ICON_QUESTION)
        if dlg.ShowModal() == wx.ID_YES:
            self.cancelled = True
            # Show the main window when training is cancelled
            if self.training_manager and self.training_manager.parent_frame:
                self.training_manager.parent_frame.show_main_window()
            self.Destroy()
        dlg.Destroy()
    
    def is_cancelled(self):
        """Check if training was cancelled"""
        return self.cancelled
    
    def training_completed(self):
        """Call when training is completed"""
        self.status_text.SetLabel("Training completed!")
        self.training_finished = True
        # Show the main window when training is completed
        if self.training_manager and self.training_manager.parent_frame:
            self.training_manager.parent_frame.show_main_window()
