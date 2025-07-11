import wx
import os
import torch
import numpy as np
from PIL import Image, ImageFilter
from PIL.Image import Transpose
from src.model import Pix2PixGAN
from src.dataset import tensor_to_pil, rgb_to_luminosity, LuminosityNormalizeTransform
from src.image_panel import ImagePanel
import torchvision.transforms as transforms


class LuminosityTransform:
    """Transform to convert RGB image to luminosity"""
    def __call__(self, image):
        return rgb_to_luminosity(image)


class GenerateTab(wx.Panel):
    def __init__(self, parent, frame=None, settings=None):
        super().__init__(parent)
        self.frame = frame
        self.settings = settings
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.setup_ui()
        self.load_settings()
    
    def setup_ui(self):
        """Setup the user interface"""
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Model loading panel
        model_panel = wx.StaticBox(self, label="Model")
        model_sizer = wx.StaticBoxSizer(model_panel, wx.VERTICAL)
        
        # Model file picker
        model_file_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.model_picker = wx.FilePickerCtrl(
            model_panel, 
            message="Select trained model file",
            wildcard="PyTorch model files (*.pth)|*.pth",
            style=wx.FLP_USE_TEXTCTRL | wx.FLP_OPEN | wx.FLP_FILE_MUST_EXIST
        )
        self.model_picker.Bind(wx.EVT_FILEPICKER_CHANGED, self.on_model_selected)
        
        model_file_sizer.Add(wx.StaticText(model_panel, label="Model File:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5  )
        model_file_sizer.Add(self.model_picker, 1, wx.EXPAND)
        model_sizer.Add(model_file_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Model status
        self.model_status = wx.StaticText(model_panel, label="No model loaded")
        model_sizer.Add(self.model_status, 0, wx.ALL, 5)
        
        main_sizer.Add(model_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Input image panel
        input_panel = wx.StaticBox(self, label="Input Image")
        input_sizer = wx.StaticBoxSizer(input_panel, wx.VERTICAL)
        
        # Input file picker
        input_file_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.input_picker = wx.FilePickerCtrl(
            input_panel,
            message="Select albedo image",
            wildcard="Image files (*.png;*.jpg;*.jpeg;*.tiff;*.bmp)|*.png;*.jpg;*.jpeg;*.tiff;*.bmp",
            style=wx.FLP_USE_TEXTCTRL | wx.FLP_OPEN | wx.FLP_FILE_MUST_EXIST
        )
        self.input_picker.Bind(wx.EVT_FILEPICKER_CHANGED, self.on_input_image_changed)
        
        input_file_sizer.Add(wx.StaticText(input_panel, label="Albedo Image:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        input_file_sizer.Add(self.input_picker, 1, wx.EXPAND)
        input_sizer.Add(input_file_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        main_sizer.Add(input_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Generation controls
        control_panel = wx.StaticBox(self, label="Generation")
        control_sizer = wx.StaticBoxSizer(control_panel, wx.VERTICAL)
        
        # Output folder picker
        output_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.output_picker = wx.DirPickerCtrl(
            control_panel,
            message="Select output folder",
            style=wx.DIRP_USE_TEXTCTRL
        )
        self.output_picker.Bind(wx.EVT_DIRPICKER_CHANGED, self.on_output_folder_changed)
        output_sizer.Add(wx.StaticText(control_panel, label="Output Folder:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        output_sizer.Add(self.output_picker, 1, wx.EXPAND)
        control_sizer.Add(output_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Tile size setting
        tile_size_sizer = wx.BoxSizer(wx.HORIZONTAL)
        tile_size_sizer.Add(wx.StaticText(control_panel, label="Tile Size:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.tile_size_spin = wx.SpinCtrl(control_panel, value="64", min=8, max=256, initial=64)
        self.tile_size_spin.SetToolTip("Size of processing tiles in pixels (8-256). Smaller tiles = more detail, larger tiles = faster processing.")
        self.tile_size_spin.Bind(wx.EVT_SPINCTRL, self.on_tile_size_changed)
        tile_size_sizer.Add(self.tile_size_spin, 0)
        tile_size_sizer.Add(wx.StaticText(control_panel, label=" pixels"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 3)
        control_sizer.Add(tile_size_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # GPU threads setting
        gpu_threads_sizer = wx.BoxSizer(wx.HORIZONTAL)
        gpu_threads_sizer.Add(wx.StaticText(control_panel, label="GPU Threads:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.gpu_threads_spin = wx.SpinCtrl(control_panel, value="4", min=1, max=16, initial=4)
        self.gpu_threads_spin.SetToolTip("Number of parallel GPU threads (1-16). Higher values = faster processing but more VRAM usage.")
        self.gpu_threads_spin.Bind(wx.EVT_SPINCTRL, self.on_gpu_threads_changed)
        gpu_threads_sizer.Add(self.gpu_threads_spin, 0)
        gpu_threads_sizer.Add(wx.StaticText(control_panel, label=" threads"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 3)
        control_sizer.Add(gpu_threads_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Second pass settings
        second_pass_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Enable second pass checkbox
        self.enable_second_pass_checkbox = wx.CheckBox(control_panel, label="Enable Second Pass (50% offset)")
        self.enable_second_pass_checkbox.SetToolTip("Enable a second pass with 50% tile offset to reduce seams")
        self.enable_second_pass_checkbox.Bind(wx.EVT_CHECKBOX, self.on_second_pass_settings_changed)
        second_pass_sizer.Add(self.enable_second_pass_checkbox, 0, wx.ALL, 5)
        
        # Edge wrapping checkbox
        self.edge_wrapping_checkbox = wx.CheckBox(control_panel, label="Edge Wrapping for Second Pass")
        self.edge_wrapping_checkbox.SetToolTip("Wrap edge tiles as if the image tiles seamlessly (for second pass)")
        self.edge_wrapping_checkbox.Bind(wx.EVT_CHECKBOX, self.on_second_pass_settings_changed)
        second_pass_sizer.Add(self.edge_wrapping_checkbox, 0, wx.ALL | wx.LEFT, 20)
        
        # Feathering checkbox
        self.feather_second_pass_checkbox = wx.CheckBox(control_panel, label="Feather Second Pass Blending")
        self.feather_second_pass_checkbox.SetToolTip("Use gradient feathering for smoother blending of second pass tiles")
        self.feather_second_pass_checkbox.Bind(wx.EVT_CHECKBOX, self.on_second_pass_settings_changed)
        second_pass_sizer.Add(self.feather_second_pass_checkbox, 0, wx.ALL | wx.LEFT, 20)
        
        control_sizer.Add(second_pass_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Transformation settings
        transform_panel = wx.StaticBox(control_panel, label="Additional Transformations")
        transform_sizer = wx.StaticBoxSizer(transform_panel, wx.VERTICAL)
        
        # Transformation options
        self.transform_flip_horizontal_checkbox = wx.CheckBox(transform_panel, label="Flip Horizontal")
        self.transform_flip_horizontal_checkbox.SetToolTip("Generate additional roughness map with horizontally flipped source")
        self.transform_flip_horizontal_checkbox.Bind(wx.EVT_CHECKBOX, self.on_transform_settings_changed)
        transform_sizer.Add(self.transform_flip_horizontal_checkbox, 0, wx.ALL, 5)
        
        self.transform_flip_vertical_checkbox = wx.CheckBox(transform_panel, label="Flip Vertical")
        self.transform_flip_vertical_checkbox.SetToolTip("Generate additional roughness map with vertically flipped source")
        self.transform_flip_vertical_checkbox.Bind(wx.EVT_CHECKBOX, self.on_transform_settings_changed)
        transform_sizer.Add(self.transform_flip_vertical_checkbox, 0, wx.ALL, 5)
        
        self.transform_rotate_90_checkbox = wx.CheckBox(transform_panel, label="Rotate 90°")
        self.transform_rotate_90_checkbox.SetToolTip("Generate additional roughness map with 90° rotated source")
        self.transform_rotate_90_checkbox.Bind(wx.EVT_CHECKBOX, self.on_transform_settings_changed)
        transform_sizer.Add(self.transform_rotate_90_checkbox, 0, wx.ALL, 5)
        
        self.transform_rotate_180_checkbox  = wx.CheckBox(transform_panel, label="Rotate 180°")
        self.transform_rotate_180_checkbox.SetToolTip("Generate additional roughness map with 180° rotated source")
        self.transform_rotate_180_checkbox.Bind(wx.EVT_CHECKBOX, self.on_transform_settings_changed)
        transform_sizer.Add(self.transform_rotate_180_checkbox, 0, wx.ALL, 5)
        
        self.transform_rotate_270_checkbox = wx.CheckBox(transform_panel, label="Rotate 270°")
        self.transform_rotate_270_checkbox.SetToolTip("Generate additional roughness map with 270° rotated source")
        self.transform_rotate_270_checkbox.Bind(wx.EVT_CHECKBOX, self.on_transform_settings_changed)
        transform_sizer.Add(self.transform_rotate_270_checkbox, 0, wx.ALL, 5)
        
        control_sizer.Add(transform_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Generate button
        self.generate_button = wx.Button(control_panel, label="Generate Roughness Map")
        self.generate_button.Bind(wx.EVT_BUTTON, self.on_generate)
        self.generate_button.Enable(False)
        control_sizer.Add(self.generate_button, 0, wx.ALIGN_CENTER | wx.ALL, 5)
        
        main_sizer.Add(control_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Status
        self.status_text = wx.StaticText(self, label="Load a model and select an input image to generate")
        main_sizer.Add(self.status_text, 0, wx.ALL, 5)
        
        # Preview area with image panel
        preview_panel = wx.StaticBox(self, label="Preview")
        preview_sizer = wx.StaticBoxSizer(preview_panel, wx.VERTICAL)
        
        self.preview_area = ImagePanel(preview_panel)
        self.preview_area.SetMinSize((400, 200))
        preview_sizer.Add(self.preview_area, 1, wx.EXPAND | wx.ALL, 5)
        
        main_sizer.Add(preview_sizer, 1, wx.EXPAND | wx.ALL, 5)
        
        self.SetSizer(main_sizer)
    
    def load_settings(self):
        """Load settings and populate the UI fields"""
        if not self.settings:
            return
        
        # Load model path
        model_path = self.settings.get_generate_model_path()
        if model_path and os.path.exists(model_path):
            self.model_picker.SetPath(model_path)
            self.load_model(model_path)
        
        # Load output path
        output_path = self.settings.get_generate_output_path()
        if output_path and os.path.exists(output_path):
            self.output_picker.SetPath(output_path)
        
        # Load tile size
        tile_size = self.settings.get_generate_tile_size()
        self.tile_size_spin.SetValue(tile_size)
        
        # Load GPU threads
        gpu_threads = self.settings.get_generate_gpu_threads()
        self.gpu_threads_spin.SetValue(gpu_threads)
        
        # Load second pass settings
        enable_second_pass = self.settings.get_generate_enable_second_pass()
        self.enable_second_pass_checkbox.SetValue(enable_second_pass)
        
        edge_wrapping = self.settings.get_generate_edge_wrapping()
        self.edge_wrapping_checkbox.SetValue(edge_wrapping)
        
        feather_second_pass = self.settings.get_generate_feather_second_pass()
        self.feather_second_pass_checkbox.SetValue(feather_second_pass)
        
        # Load transformation settings
        flip_horizontal = self.settings.get_transform_flip_horizontal()
        self.transform_flip_horizontal_checkbox.SetValue(flip_horizontal)
        
        flip_vertical = self.settings.get_transform_flip_vertical()
        self.transform_flip_vertical_checkbox.SetValue(flip_vertical)
        
        rotate_90 = self.settings.get_transform_rotate_90()
        self.transform_rotate_90_checkbox.SetValue(rotate_90)
        
        rotate_180 = self.settings.get_transform_rotate_180()
        self.transform_rotate_180_checkbox.SetValue(rotate_180)
        
        rotate_270 = self.settings.get_transform_rotate_270()
        self.transform_rotate_270_checkbox.SetValue(rotate_270)
        
        # Update button state after loading
        self.update_generate_button_state()
    
    def load_model(self, model_path):
        """Load a model from the given path"""
        if not model_path:
            return
        
        try:
            # Initialize model
            self.model = Pix2PixGAN(device=self.device)
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.model.generator.eval()
            
            self.model_status.SetLabel(f"Model loaded (Epoch {checkpoint.get('epoch', 'unknown')})")
            if self.frame:
                self.frame.SetStatusText(f"Model loaded successfully (Epoch {checkpoint.get('epoch', 'unknown')})")
            
        except Exception as e:
            self.model_status.SetLabel(f"Error loading model: {str(e)}")
            if self.frame:
                self.frame.SetStatusText(f"Error loading model: {str(e)}")
            self.model = None
    
    def on_model_selected(self, event):
        """Handle model file selection"""
        model_path = event.GetPath()
        if not model_path:
            return
        
        # Load the model
        self.load_model(model_path)
        
        # Save to settings
        if self.settings and self.model:
            self.settings.set_generate_model_path(model_path)
            self.settings.save()
        
        # Update generate button state
        self.update_generate_button_state()
    
    def on_input_image_changed(self, event):
        """Handle input image selection"""
        self.update_generate_button_state()
    
    def on_output_folder_changed(self, event):
        """Handle output folder selection"""
        output_path = event.GetPath()
        if self.settings and output_path:
            self.settings.set_generate_output_path(output_path)
            self.settings.save()
        self.update_generate_button_state()
    
    def on_tile_size_changed(self, event):
        """Handle tile size setting change"""
        if self.settings:
            tile_size = self.tile_size_spin.GetValue()
            self.settings.set_generate_tile_size(tile_size)
            self.settings.save()
    
    def on_gpu_threads_changed(self, event):
        """Handle GPU threads setting change"""
        if self.settings:
            gpu_threads = self.gpu_threads_spin.GetValue()
            self.settings.set_generate_gpu_threads(gpu_threads)
            self.settings.save()
    
    def on_second_pass_settings_changed(self, event):
        """Handle second pass settings change"""
        if self.settings:
            enable_second_pass = self.enable_second_pass_checkbox.GetValue()
            edge_wrapping = self.edge_wrapping_checkbox.GetValue()
            feather_second_pass = self.feather_second_pass_checkbox.GetValue()
            self.settings.set_generate_enable_second_pass(enable_second_pass)
            self.settings.set_generate_edge_wrapping(edge_wrapping)
            self.settings.set_generate_feather_second_pass(feather_second_pass)
            self.settings.save()
    
    def on_transform_settings_changed(self, event):
        """Handle transformation settings change"""
        if self.settings:
            flip_horizontal = self.transform_flip_horizontal_checkbox.GetValue()
            flip_vertical = self.transform_flip_vertical_checkbox.GetValue()
            rotate_90 = self.transform_rotate_90_checkbox.GetValue()
            rotate_180 = self.transform_rotate_180_checkbox.GetValue()
            rotate_270 = self.transform_rotate_270_checkbox.GetValue()
            
            self.settings.set_transform_flip_horizontal(flip_horizontal)
            self.settings.set_transform_flip_vertical(flip_vertical)
            self.settings.set_transform_rotate_90(rotate_90)
            self.settings.set_transform_rotate_180(rotate_180)
            self.settings.set_transform_rotate_270(rotate_270)
            self.settings.save()
    
    def update_generate_button_state(self):
        """Enable generate button only when all required fields are populated"""
        model_loaded = self.model is not None
        input_selected = bool(self.input_picker.GetPath())
        output_selected = bool(self.output_picker.GetPath())
        
        all_ready = model_loaded and input_selected and output_selected
        self.generate_button.Enable(all_ready)
        
        if all_ready:
            self.status_text.SetLabel("Ready to generate! Click the Generate button.")
        elif not model_loaded:
            self.status_text.SetLabel("Load a model to begin.")
        elif not input_selected:
            self.status_text.SetLabel("Select an albedo image.")
        elif not output_selected:
            self.status_text.SetLabel("Select an output folder.")
    
    def apply_transformation(self, image, transform_type):
        """Apply a transformation to an image"""
        if transform_type == "flip_horizontal":
            return image.transpose(Transpose.FLIP_LEFT_RIGHT)
        elif transform_type == "flip_vertical":
            return image.transpose(Transpose.FLIP_TOP_BOTTOM)
        elif transform_type == "rotate_90":
            return image.transpose(Transpose.ROTATE_90)
        elif transform_type == "rotate_180":
            return image.transpose(Transpose.ROTATE_180)
        elif transform_type == "rotate_270":
            return image.transpose(Transpose.ROTATE_270)
        else:
            return image
    
    def reverse_transformation(self, image, transform_type):
        """Reverse a transformation on an image"""
        if transform_type == "flip_horizontal":
            return image.transpose(Transpose.FLIP_LEFT_RIGHT)
        elif transform_type == "flip_vertical":
            return image.transpose(Transpose.FLIP_TOP_BOTTOM)
        elif transform_type == "rotate_90":
            return image.transpose(Transpose.ROTATE_270)  # Reverse 90° rotation
        elif transform_type == "rotate_180":
            return image.transpose(Transpose.ROTATE_180)  # 180° is its own reverse
        elif transform_type == "rotate_270":
            return image.transpose(Transpose.ROTATE_90)   # Reverse 270° rotation
        else:
            return image
    
    def get_transform_suffix(self, transform_type):
        """Get filename suffix for a transformation"""
        suffixes = {
            "flip_horizontal": "_flip_h",
            "flip_vertical": "_flip_v", 
            "rotate_90": "_rot90",
            "rotate_180": "_rot180",
            "rotate_270": "_rot270"
        }
        return suffixes.get(transform_type, "")
    
    def get_transform_display_name(self, transform_type):
        """Get display name for a transformation"""
        names = {
            "flip_horizontal": "Flip Horizontal",
            "flip_vertical": "Flip Vertical",
            "rotate_90": "Rotate 90°",
            "rotate_180": "Rotate 180°", 
            "rotate_270": "Rotate 270°"
        }
        return names.get(transform_type, transform_type)
    
    def on_generate(self, event):
        """Generate roughness map from input albedo image with optional transformations"""
        if not self.model:
            wx.MessageBox("Please load a model first.", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        input_path = self.input_picker.GetPath()
        if not input_path:
            wx.MessageBox("Please select an input albedo image.", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        output_dir = self.output_picker.GetPath()
        if not output_dir:
            wx.MessageBox("Please select an output folder.", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        try:
            # Get settings for status display
            tile_size = self.tile_size_spin.GetValue()
            gpu_threads = self.gpu_threads_spin.GetValue()
            
            # Load input image
            input_image = Image.open(input_path).convert('RGB')
            input_filename = os.path.splitext(os.path.basename(input_path))[0]
            
            # Determine which transformations to apply
            transformations = []
            if self.transform_flip_horizontal_checkbox.GetValue():
                transformations.append("flip_horizontal")
            if self.transform_flip_vertical_checkbox.GetValue():
                transformations.append("flip_vertical")
            if self.transform_rotate_90_checkbox.GetValue():
                transformations.append("rotate_90")
            if self.transform_rotate_180_checkbox.GetValue():
                transformations.append("rotate_180")
            if self.transform_rotate_270_checkbox.GetValue():
                transformations.append("rotate_270")
            
            # Include base generation (always)
            total_generations = 1 + len(transformations)
            current_generation = 0
            
            if self.frame:
                self.frame.SetStatusText(f"Generating {total_generations} roughness map(s) using {tile_size}x{tile_size} tiles with {gpu_threads} GPU threads...")
                self.frame.set_progress(0, show=True)
            wx.GetApp().Yield()
            
            # Dictionary to store all generated images for preview
            images_dict = {}
            generated_files = []
            
            # Always generate the base image first
            current_generation += 1
            if self.frame:
                self.frame.SetStatusText(f"Generating base roughness map ({current_generation}/{total_generations})...")
                self.frame.set_progress(int(100 * (current_generation - 1) / total_generations))
            wx.GetApp().Yield()
            
            # Generate base roughness map
            base_output_image = self.generate_simple_tiled(input_image)
            
            # Save base result
            base_output_path = os.path.join(output_dir, f"{input_filename}_roughness.png")
            base_output_image.save(base_output_path)
            generated_files.append(base_output_path)
            
            # Add to preview
            display_input = rgb_to_luminosity(input_image)
            images_dict['original'] = (input_image, 'Albedo (Original)')
            images_dict['luminosity'] = (display_input, 'Albedo (Luminosity)')
            images_dict['base'] = (base_output_image, 'Base Roughness')
            
            # Generate transformed versions
            for transform_type in transformations:
                current_generation += 1
                transform_name = self.get_transform_display_name(transform_type)
                
                if self.frame:
                    self.frame.SetStatusText(f"Generating {transform_name} roughness map ({current_generation}/{total_generations})...")
                    self.frame.set_progress(int(100 * (current_generation - 1) / total_generations))
                wx.GetApp().Yield()
                
                # Apply transformation to source
                transformed_input = self.apply_transformation(input_image, transform_type)
                
                # Generate roughness map from transformed input
                transformed_output = self.generate_simple_tiled(transformed_input)
                
                # Reverse transformation on output to match original orientation
                final_output = self.reverse_transformation(transformed_output, transform_type)
                
                # Save transformed result
                suffix = self.get_transform_suffix(transform_type)
                transform_output_path = os.path.join(output_dir, f"{input_filename}_roughness{suffix}.png")
                final_output.save(transform_output_path)
                generated_files.append(transform_output_path)
                
                # Add to preview
                images_dict[transform_type] = (final_output, f'{transform_name} Roughness')
            
            # Update progress to complete
            if self.frame:
                self.frame.set_progress(100)
            wx.GetApp().Yield()
            
            # Display all images in the preview area
            self.preview_area.set_images(images_dict, default_key='base')
            
            # Update status bar and local status
            if self.frame:
                files_text = '\n'.join(generated_files)
                self.frame.SetStatusText(f"Generated {len(generated_files)} roughness map(s)")
                self.frame.hide_progress()
            
            self.status_text.SetLabel("Generation complete!")
            
            if len(generated_files) == 1:
                wx.MessageBox(f"Roughness map generated successfully!\nSaved to: {generated_files[0]}", 
                             "Generation Complete", wx.OK | wx.ICON_INFORMATION)
            else:
                files_list = '\n'.join([os.path.basename(f) for f in generated_files])
                wx.MessageBox(f"{len(generated_files)} roughness maps generated successfully!\n\nFiles:\n{files_list}", 
                             "Generation Complete", wx.OK | wx.ICON_INFORMATION)
            
        except Exception as e:
            if self.frame:
                self.frame.hide_progress()
            self.status_text.SetLabel("Generation failed")
            wx.MessageBox(f"Error generating roughness map: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)

    def generate_simple_tiled(self, input_image):
        """Generate roughness map by cutting input into tiles, processing in parallel, then stitching results"""
        # Convert input to luminosity (returns PIL Image for PIL input)
        luminosity_image = rgb_to_luminosity(input_image)
        
        # Get image dimensions - ensure we have a PIL Image
        if not isinstance(luminosity_image, Image.Image):
            raise RuntimeError("rgb_to_luminosity should return PIL Image for PIL input")
        
        orig_width, orig_height = luminosity_image.size
        
        # Create output image
        output_image = Image.new('RGB', (orig_width, orig_height))
        
        # Create transform for processing tiles (after upscaling to 256x256)
        transform = transforms.Compose([
            transforms.ToTensor(),
            LuminosityNormalizeTransform()
        ])
        
        # Get user-selected settings
        tile_size = self.tile_size_spin.GetValue()
        gpu_threads = self.gpu_threads_spin.GetValue()
        enable_second_pass = self.enable_second_pass_checkbox.GetValue()
        edge_wrapping = self.edge_wrapping_checkbox.GetValue()
        feather_second_pass = self.feather_second_pass_checkbox.GetValue()
        
        # First pass: regular tiling
        self.process_tile_pass(luminosity_image, output_image, transform, tile_size, gpu_threads, 0, 0, "First pass", enable_second_pass)
        
        # Second pass: offset by 50%
        if enable_second_pass:
            offset_x = tile_size // 2
            offset_y = tile_size // 2
            self.process_tile_pass(luminosity_image, output_image, transform, tile_size, gpu_threads, 
                                 offset_x, offset_y, "Second pass", enable_second_pass, edge_wrapping, blend_mode=True, feather_blend=feather_second_pass)
        
        return output_image
    
    def process_tile_pass(self, luminosity_image, output_image, transform, tile_size, gpu_threads, 
                         offset_x, offset_y, pass_name, enable_second_pass, edge_wrapping=False, blend_mode=False, feather_blend=False):
        """Process a single pass of tile generation"""
        orig_width, orig_height = luminosity_image.size
        
        # Calculate number of tiles needed for this pass
        start_x_positions = list(range(-offset_x, orig_width, tile_size))
        start_y_positions = list(range(-offset_y, orig_height, tile_size))
        
        # Remove negative positions that don't overlap with the image
        start_x_positions = [x for x in start_x_positions if x + tile_size > 0]
        start_y_positions = [y for y in start_y_positions if y + tile_size > 0]
        
        total_tiles = len(start_x_positions) * len(start_y_positions)
        
        # Prepare all tile data
        tile_data = []
        for start_y in start_y_positions:
            for start_x in start_x_positions:
                # Calculate actual tile bounds that intersect with the image
                actual_start_x = max(0, start_x)
                actual_start_y = max(0, start_y)
                actual_end_x = min(start_x + tile_size, orig_width)
                actual_end_y = min(start_y + tile_size, orig_height)
                
                # Skip tiles that don't intersect with the image
                if actual_start_x >= actual_end_x or actual_start_y >= actual_end_y:
                    continue
                
                # Extract and prepare tile
                if edge_wrapping:
                    # Use wrapped tile extraction
                    upscaled_tile = self.create_wrapped_tile(luminosity_image, start_x, start_y, tile_size)
                    upscaled_tile = upscaled_tile.resize((256, 256), Image.Resampling.LANCZOS)
                else:
                    # Regular tile extraction with padding
                    source_tile = luminosity_image.crop((actual_start_x, actual_start_y, actual_end_x, actual_end_y))
                    padded_tile = Image.new('RGB', (tile_size, tile_size), (0, 0, 0))
                    
                    # Calculate paste position within the padded tile
                    paste_x = actual_start_x - start_x
                    paste_y = actual_start_y - start_y
                    padded_tile.paste(source_tile, (paste_x, paste_y))
                    
                    upscaled_tile = padded_tile.resize((256, 256), Image.Resampling.LANCZOS)
                
                tile_data.append({
                    'tile': upscaled_tile,
                    'start_x': start_x,
                    'start_y': start_y,
                    'actual_start_x': actual_start_x,
                    'actual_start_y': actual_start_y,
                    'actual_end_x': actual_end_x,
                    'actual_end_y': actual_end_y,
                    'index': len(tile_data)
                })
        
        # Process tiles in parallel batches
        processed_tiles = 0
        batch_size = gpu_threads
        results = {}
        
        def process_tile_batch(batch):
            """Process a batch of tiles on GPU"""
            batch_results = {}
            
            if not self.model or not self.model.generator:
                raise RuntimeError("Model not properly loaded")
            
            # Prepare batch tensors
            batch_tensors = []
            for tile_info in batch:
                tile_tensor = transform(tile_info['tile'])
                if not isinstance(tile_tensor, torch.Tensor):
                    raise RuntimeError("Transform should return a tensor")
                batch_tensors.append(tile_tensor)
            
            # Stack into batch and move to device
            if batch_tensors:
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                
                # Process entire batch through network in one GPU call
                with torch.no_grad():
                    output_batch = self.model.generator(batch_tensor)
                
                # Convert results back to PIL images
                for i, tile_info in enumerate(batch):
                    tile_result_256 = tensor_to_pil(output_batch[i])
                    tile_result = tile_result_256.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
                    
                    # Calculate the portion of the tile to use
                    crop_start_x = tile_info['actual_start_x'] - tile_info['start_x']
                    crop_start_y = tile_info['actual_start_y'] - tile_info['start_y']
                    crop_end_x = crop_start_x + (tile_info['actual_end_x'] - tile_info['actual_start_x'])
                    crop_end_y = crop_start_y + (tile_info['actual_end_y'] - tile_info['actual_start_y'])
                    
                    result_crop = tile_result.crop((crop_start_x, crop_start_y, crop_end_x, crop_end_y))
                    
                    batch_results[tile_info['index']] = {
                        'image': result_crop,
                        'actual_start_x': tile_info['actual_start_x'],
                        'actual_start_y': tile_info['actual_start_y']
                    }
            
            return batch_results
        
        # Process all batches
        for i in range(0, len(tile_data), batch_size):
            batch = tile_data[i:i + batch_size]
            
            try:
                batch_results = process_tile_batch(batch)
                results.update(batch_results)
                
                # Update progress
                processed_tiles += len(batch_results)
                progress = (processed_tiles / len(tile_data)) * 100
                
                if self.frame:
                    self.frame.SetStatusText(f"{pass_name}: Processing {tile_size}x{tile_size} tiles ({gpu_threads} GPU threads)... {processed_tiles}/{len(tile_data)} ({progress:.1f}%)")
                    # For second pass, show progress from 50% to 100%
                    if blend_mode:
                        total_progress = 50 + (progress * 0.5)
                    else:
                        total_progress = progress * 0.5 if enable_second_pass else progress
                    self.frame.set_progress(int(total_progress))
                
                pass_text = f"{pass_name}: {processed_tiles}/{len(tile_data)} ({progress:.0f}%)"
                self.status_text.SetLabel(pass_text)
                wx.GetApp().Yield()
                
            except Exception as e:
                raise RuntimeError(f"Error processing tile batch in {pass_name}: {str(e)}")
        
        # Apply results to output image
        for i in range(len(tile_data)):
            if i in results:
                result_info = results[i]
                if blend_mode:
                    # Blend with existing image
                    if feather_blend:
                        self.feather_blend_images(output_image, result_info['image'], 
                                                result_info['actual_start_x'], result_info['actual_start_y'], tile_size)
                    else:
                        self.blend_images(output_image, result_info['image'], 
                                        result_info['actual_start_x'], result_info['actual_start_y'], 0.5)
                else:
                    # Direct paste
                    output_image.paste(result_info['image'], 
                                     (result_info['actual_start_x'], result_info['actual_start_y']))
    
    def create_wrapped_tile(self, luminosity_image, start_x, start_y, tile_size):
        """Create a tile with edge wrapping for seamless tiling"""
        orig_width, orig_height = luminosity_image.size
        
        # Create the base tile
        tile = Image.new('RGB', (tile_size, tile_size), (0, 0, 0))
        
        for y in range(tile_size):
            for x in range(tile_size):
                # Calculate source coordinates with wrapping
                src_x = (start_x + x) % orig_width
                src_y = (start_y + y) % orig_height
                
                # Get pixel from source image
                pixel = luminosity_image.getpixel((src_x, src_y))
                tile.putpixel((x, y), pixel)
        
        return tile
    
    def blend_images(self, base_image, overlay_image, start_x, start_y, blend_factor=0.5):
        """Blend overlay image into base image at specified position"""
        overlay_width, overlay_height = overlay_image.size
        base_width, base_height = base_image.size
        
        # Calculate the region to blend
        end_x = min(start_x + overlay_width, base_width)
        end_y = min(start_y + overlay_height, base_height)
        
        # Get the overlapping region from base image
        base_region = base_image.crop((start_x, start_y, end_x, end_y))
        
        # Crop overlay to match the actual region size
        actual_width = end_x - start_x
        actual_height = end_y - start_y
        overlay_region = overlay_image.crop((0, 0, actual_width, actual_height))
        
        # Blend the images
        blended_region = Image.blend(base_region, overlay_region, blend_factor)
        
        # Paste back into base image
        base_image.paste(blended_region, (start_x, start_y))
    
    def feather_blend_images(self, base_image, overlay_image, start_x, start_y, tile_size):
        """Blend overlay image into base image with feathered edges for smoother transitions"""
        import numpy as np
        
        overlay_width, overlay_height = overlay_image.size
        base_width, base_height = base_image.size
        
        # Calculate the region to blend
        end_x = min(start_x + overlay_width, base_width)
        end_y = min(start_y + overlay_height, base_height)
        
        # Get the overlapping region from base image
        base_region = base_image.crop((start_x, start_y, end_x, end_y))
        
        # Crop overlay to match the actual region size
        actual_width = end_x - start_x
        actual_height = end_y - start_y
        overlay_region = overlay_image.crop((0, 0, actual_width, actual_height))
        
        # Create feathering mask
        mask = self.create_feather_mask(actual_width, actual_height, tile_size)
        
        # Convert images to numpy arrays for pixel-level manipulation
        base_array = np.array(base_region, dtype=np.float32)
        overlay_array = np.array(overlay_region, dtype=np.float32)
        
        # Apply feathered blending
        blended_array = base_array * (1.0 - mask) + overlay_array * mask
        
        # Convert back to PIL Image
        blended_region = Image.fromarray(np.uint8(blended_array))
        
        # Paste back into base image
        base_image.paste(blended_region, (start_x, start_y))
    
    def create_feather_mask(self, width, height, tile_size):
        """Create a feathering mask with gradual transitions at tile edges"""
        import numpy as np
        
        # Create base mask (full opacity in center)
        mask = np.ones((height, width, 3), dtype=np.float32)
        
        # Calculate feather distance (percentage of tile size)
        feather_distance = max(1, tile_size // 8)  # 12.5% of tile size for feathering
        
        # Apply horizontal feathering (left and right edges)
        for x in range(min(feather_distance, width)):
            # Left edge fade in
            fade_factor = x / feather_distance
            mask[:, x] *= fade_factor
            
            # Right edge fade out
            if width - 1 - x >= 0:
                mask[:, width - 1 - x] *= fade_factor
        
        # Apply vertical feathering (top and bottom edges)
        for y in range(min(feather_distance, height)):
            # Top edge fade in
            fade_factor = y / feather_distance
            mask[y, :] *= fade_factor
            
            # Bottom edge fade out
            if height - 1 - y >= 0:
                mask[height - 1 - y, :] *= fade_factor
        
        return mask
