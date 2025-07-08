import wx
from PIL import Image


class ImagePanel(wx.Panel):
    """A panel that displays an image with zoom and pan capabilities"""
    
    def __init__(self, parent):
        super().__init__(parent, style=wx.SUNKEN_BORDER)
        self.SetBackgroundColour(wx.Colour(240, 240, 240))
        
        # Enable double buffering to prevent flickering
        self.SetDoubleBuffered(True)
        
        # Image data
        self.pil_image = None
        self.wx_image = None
        self.bitmap = None
        
        # Multiple image support
        self.images = {}  # Dictionary to store multiple images
        self.current_image_key = None
        self.image_labels = {}  # Labels for each image
        
        # View state
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        
        # Bind events
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_MOUSEWHEEL, self.on_mouse_wheel)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
        self.Bind(wx.EVT_LEFT_UP, self.on_left_up)
        self.Bind(wx.EVT_MOTION, self.on_motion)
        self.Bind(wx.EVT_RIGHT_DOWN, self.on_right_click)
        self.Bind(wx.EVT_ENTER_WINDOW, self.on_enter)
        self.Bind(wx.EVT_LEAVE_WINDOW, self.on_leave)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.on_erase_background)
    
    def on_erase_background(self, event):
        """Prevent flickering by not erasing background"""
        pass
    
    def set_image(self, pil_image):
        """Set the image to display"""
        self.pil_image = pil_image.copy()
        self.wx_image = wx.Image(pil_image.size[0], pil_image.size[1])
        self.wx_image.SetData(pil_image.convert('RGB').tobytes())
        
        # Reset view to fit image
        self.fit_image_to_panel()
        self.Refresh()
    
    def set_images(self, images_dict, default_key=None):
        """Set multiple images that can be switched between
        
        Args:
            images_dict: Dictionary with keys as image identifiers and values as (pil_image, label) tuples
            default_key: Key of the image to display initially
        """
        self.images = {}
        self.image_labels = {}
        
        for key, (pil_image, label) in images_dict.items():
            self.images[key] = pil_image.copy()
            self.image_labels[key] = label
        
        if default_key and default_key in self.images:
            self.current_image_key = default_key
        elif self.images:
            self.current_image_key = list(self.images.keys())[0]
        else:
            self.current_image_key = None
        
        if self.current_image_key:
            self._update_current_image()
        else:
            self.clear_image()
    
    def _update_current_image(self):
        """Update the displayed image based on current_image_key"""
        if not self.current_image_key or self.current_image_key not in self.images:
            return
        
        # Store current view state
        old_zoom = self.zoom_factor
        old_pan_x = self.pan_x
        old_pan_y = self.pan_y
        old_image_size = None
        if self.wx_image:
            old_image_size = (self.wx_image.GetWidth(), self.wx_image.GetHeight())
        
        # Set the new image
        pil_image = self.images[self.current_image_key]
        self.pil_image = pil_image.copy()
        self.wx_image = wx.Image(pil_image.size[0], pil_image.size[1])
        self.wx_image.SetData(pil_image.convert('RGB').tobytes())
        
        # Adjust view state for different image sizes
        if old_image_size:
            new_image_size = (self.wx_image.GetWidth(), self.wx_image.GetHeight())
            
            if old_image_size != new_image_size:
                # Calculate scale factor between old and new image
                scale_x = new_image_size[0] / old_image_size[0]
                scale_y = new_image_size[1] / old_image_size[1]
                
                # Adjust pan to maintain relative position
                panel_size = self.GetSize()
                center_x = panel_size.width / 2
                center_y = panel_size.height / 2
                
                # Calculate relative position from center
                rel_x = (old_pan_x - center_x) / old_zoom
                rel_y = (old_pan_y - center_y) / old_zoom
                
                # Apply scale and recalculate pan
                self.pan_x = center_x + (rel_x * scale_x * old_zoom)
                self.pan_y = center_y + (rel_y * scale_y * old_zoom)
            else:
                # Same size, keep the view state
                self.zoom_factor = old_zoom
                self.pan_x = old_pan_x
                self.pan_y = old_pan_y
        else:
            # First image, fit to panel
            self.fit_image_to_panel()
        
        self.Refresh()
    
    def switch_to_next_image(self):
        """Switch to the next image in the collection"""
        if not self.images or len(self.images) <= 1:
            return
        
        keys = list(self.images.keys())
        if self.current_image_key in keys:
            current_index = keys.index(self.current_image_key)
            next_index = (current_index + 1) % len(keys)
            self.current_image_key = keys[next_index]
        else:
            self.current_image_key = keys[0]
        
        self._update_current_image()
        
        # Show a brief tooltip indicating which image is now displayed
        if self.current_image_key in self.image_labels:
            self.SetToolTip(f"Showing: {self.image_labels[self.current_image_key]}")
            wx.CallLater(2000, lambda: self.SetToolTip(""))
    
    def fit_image_to_panel(self):
        """Reset zoom and pan to fit the image in the panel"""
        if not self.wx_image:
            return
        
        panel_size = self.GetSize()
        if panel_size.width <= 0 or panel_size.height <= 0:
            return
        
        image_width = self.wx_image.GetWidth()
        image_height = self.wx_image.GetHeight()
        
        # Calculate zoom to fit
        zoom_x = panel_size.width / image_width
        zoom_y = panel_size.height / image_height
        self.zoom_factor = min(zoom_x, zoom_y) * 0.9  # 90% to leave some margin
        
        # Center the image
        self.pan_x = (panel_size.width - image_width * self.zoom_factor) / 2
        self.pan_y = (panel_size.height - image_height * self.zoom_factor) / 2
    
    def on_size(self, event):
        """Handle panel resize"""
        if self.wx_image:
            self.fit_image_to_panel()
        self.Refresh()
        event.Skip()
    
    def on_paint(self, event):
        """Paint the image using double buffering"""
        # Use BufferedPaintDC for automatic double buffering
        dc = wx.BufferedPaintDC(self)
        self.draw_content(dc)
    
    def draw_content(self, dc):
        """Draw the actual content to the device context"""
        dc.Clear()
        
        if not self.wx_image:
            # Draw placeholder text
            dc.SetTextForeground(wx.Colour(128, 128, 128))
            dc.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
            text = "Generated image will appear here"
            text_size = dc.GetTextExtent(text)
            panel_size = self.GetSize()
            x = (panel_size.width - text_size.width) // 2
            y = (panel_size.height - text_size.height) // 2
            dc.DrawText(text, x, y)
            return
        
        # Scale the image
        scaled_width = int(self.wx_image.GetWidth() * self.zoom_factor)
        scaled_height = int(self.wx_image.GetHeight() * self.zoom_factor)
        
        if scaled_width > 0 and scaled_height > 0:
            scaled_image = self.wx_image.Scale(scaled_width, scaled_height, wx.IMAGE_QUALITY_HIGH)
            self.bitmap = wx.Bitmap(scaled_image)
            
            # Draw the bitmap at the current pan position
            dc.DrawBitmap(self.bitmap, int(self.pan_x), int(self.pan_y))
            
            # Draw image label if we have multiple images
            if len(self.images) > 1 and self.current_image_key in self.image_labels:
                dc.SetTextForeground(wx.Colour(255, 255, 255))
                dc.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
                label_text = self.image_labels[self.current_image_key]
                text_size = dc.GetTextExtent(label_text)
                
                # Draw background rectangle
                padding = 4
                bg_rect = wx.Rect(10, 10, text_size.width + padding * 2, text_size.height + padding * 2)
                dc.SetBrush(wx.Brush(wx.Colour(0, 0, 0, 128)))
                dc.SetPen(wx.Pen(wx.Colour(0, 0, 0, 0)))
                dc.DrawRectangle(bg_rect)
                
                # Draw text
                dc.DrawText(label_text, 10 + padding, 10 + padding)
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming"""
        if not self.wx_image:
            return
        
        # Get mouse position relative to panel
        mouse_pos = event.GetPosition()
        
        # Calculate zoom change
        zoom_change = 1.2 if event.GetWheelRotation() > 0 else 1.0 / 1.2
        old_zoom = self.zoom_factor
        self.zoom_factor *= zoom_change
        
        # Limit zoom range
        self.zoom_factor = max(0.1, min(10.0, self.zoom_factor))
        
        # Adjust pan to zoom towards mouse position
        if old_zoom != self.zoom_factor:
            zoom_ratio = self.zoom_factor / old_zoom
            self.pan_x = mouse_pos.x - (mouse_pos.x - self.pan_x) * zoom_ratio
            self.pan_y = mouse_pos.y - (mouse_pos.y - self.pan_y) * zoom_ratio
        
        self.Refresh()
    
    def on_left_down(self, event):
        """Start dragging"""
        if not self.wx_image:
            return
        
        self.dragging = True
        self.drag_start_x = event.GetX()
        self.drag_start_y = event.GetY()
        self.SetCursor(wx.Cursor(wx.CURSOR_HAND))
        self.CaptureMouse()
    
    def on_left_up(self, event):
        """Stop dragging"""
        if self.dragging:
            self.dragging = False
            self.SetCursor(wx.Cursor(wx.CURSOR_ARROW))
            if self.HasCapture():
                self.ReleaseMouse()
    
    def on_motion(self, event):
        """Handle mouse movement for panning and cursor changes"""
        # Update cursor based on whether we're over an image
        if self.wx_image:
            if self.dragging:
                self.SetCursor(wx.Cursor(wx.CURSOR_HAND))
            else:
                self.SetCursor(wx.Cursor(wx.CURSOR_ARROW))
        else:
            self.SetCursor(wx.Cursor(wx.CURSOR_ARROW))
        
        if not self.dragging or not self.wx_image:
            return
        
        # Calculate pan offset
        dx = event.GetX() - self.drag_start_x
        dy = event.GetY() - self.drag_start_y
        
        # Only update if there's actual movement
        if dx != 0 or dy != 0:
            # Update pan position
            self.pan_x += dx
            self.pan_y += dy
            
            # Update drag start position for next movement
            self.drag_start_x = event.GetX()
            self.drag_start_y = event.GetY()
            
            # Use Refresh() to properly trigger repaint with double buffering
            self.Refresh()
    
    def on_enter(self, event):
        """Handle mouse entering the panel"""
        if self.wx_image:
            self.SetCursor(wx.Cursor(wx.CURSOR_ARROW))
    
    def on_leave(self, event):
        """Handle mouse leaving the panel"""
        self.SetCursor(wx.Cursor(wx.CURSOR_ARROW))
    
    def on_right_click(self, event):
        """Handle right-click to switch between images"""
        self.switch_to_next_image()
    
    def clear_image(self):
        """Clear the displayed image"""
        self.pil_image = None
        self.wx_image = None
        self.bitmap = None
        self.images = {}
        self.image_labels = {}
        self.current_image_key = None
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.Refresh()
