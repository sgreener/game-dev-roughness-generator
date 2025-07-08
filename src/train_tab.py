import wx
import os
from PIL import Image
from src.training_manager import TrainingManager
from src.constants import (
    TOOLTIP_SOURCE_FOLDER,
    TOOLTIP_EPOCHS,
    TOOLTIP_BATCH_SIZE,
    TOOLTIP_LEARNING_RATE,
    TOOLTIP_SAVE_FREQ,
    TOOLTIP_OUTPUT_FOLDER,
    TOOLTIP_MODEL_FILENAME,
    TOOLTIP_LAMBDA_L1,
    TOOLTIP_BETA1,
    TOOLTIP_CONTINUE_TRAINING
)

class TrainTab(wx.Panel):
    def __init__(self, parent, frame, settings):
        super().__init__(parent)
        self.settings = settings
        self.frame = frame
        self.training_manager = TrainingManager(self)

        # Main sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Source Data Panel
        source_data_panel = self.create_source_data_panel()
        main_sizer.Add(source_data_panel, 0, wx.EXPAND | wx.ALL, 5)

        # Properties Panel
        properties_panel = self.create_properties_panel()
        main_sizer.Add(properties_panel, 0, wx.EXPAND | wx.ALL, 5)

        # Progress Panel
        progress_panel = self.create_progress_panel()
        main_sizer.Add(progress_panel, 1, wx.EXPAND | wx.ALL, 5)

        self.SetSizer(main_sizer)

    def create_source_data_panel(self):
        panel = wx.StaticBox(self, label="Source Data")
        sizer = wx.StaticBoxSizer(panel, wx.VERTICAL)

        # Folder selection
        folder_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.folder_picker = wx.DirPickerCtrl(panel, message="Select Training Data Folder", style=wx.DIRP_USE_TEXTCTRL)
        self.folder_picker.SetToolTip(wx.ToolTip(TOOLTIP_SOURCE_FOLDER))
        self.folder_picker.SetPath(self.settings.get_training_source_path())
        folder_sizer.Add(wx.StaticText(panel, label="Source Folder:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        folder_sizer.Add(self.folder_picker, 1, wx.EXPAND)
        sizer.Add(folder_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Validation info
        self.validation_text = wx.StaticText(panel, label="Please select a source folder.")
        sizer.Add(self.validation_text, 0, wx.ALL, 5)

        self.folder_picker.Bind(wx.EVT_DIRPICKER_CHANGED, self.on_folder_selected)
        if self.settings.get_training_source_path():
            self.validate_data(self.settings.get_training_source_path())

        return sizer

    def create_properties_panel(self):
        panel = wx.StaticBox(self, label="Training Properties")
        sizer = wx.StaticBoxSizer(panel, wx.VERTICAL)
        grid_sizer = wx.FlexGridSizer(6, 2, 5, 5)

        # Epochs
        self.epochs_ctrl = wx.SpinCtrl(panel, value=str(self.settings.get_epochs()), min=1, max=9999999)
        self.epochs_ctrl.SetToolTip(wx.ToolTip(TOOLTIP_EPOCHS))
        self.epochs_ctrl.Bind(wx.EVT_SPINCTRL, self.on_settings_changed)
        grid_sizer.Add(wx.StaticText(panel, label="Epochs:"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid_sizer.Add(self.epochs_ctrl, 1, wx.EXPAND)

        # Batch Size
        self.batch_size_ctrl = wx.SpinCtrl(panel, value=str(self.settings.get_batch_size()), min=1, max=9999999)
        self.batch_size_ctrl.SetToolTip(wx.ToolTip(TOOLTIP_BATCH_SIZE))
        self.batch_size_ctrl.Bind(wx.EVT_SPINCTRL, self.on_settings_changed)
        grid_sizer.Add(wx.StaticText(panel, label="Batch Size:"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid_sizer.Add(self.batch_size_ctrl, 1, wx.EXPAND)

        # Learning Rate
        self.learning_rate_ctrl = wx.TextCtrl(panel, value=str(self.settings.get_learning_rate()))
        self.learning_rate_ctrl.SetToolTip(wx.ToolTip(TOOLTIP_LEARNING_RATE))
        self.learning_rate_ctrl.Bind(wx.EVT_TEXT, self.on_settings_changed)
        grid_sizer.Add(wx.StaticText(panel, label="Learning Rate:"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid_sizer.Add(self.learning_rate_ctrl, 1, wx.EXPAND)

        # Save Frequency
        self.save_freq_ctrl = wx.SpinCtrl(panel, value=str(self.settings.get_save_freq()))
        self.save_freq_ctrl.SetToolTip(wx.ToolTip(TOOLTIP_SAVE_FREQ))
        self.save_freq_ctrl.Bind(wx.EVT_SPINCTRL, self.on_settings_changed)
        grid_sizer.Add(wx.StaticText(panel, label="Save Frequency:"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid_sizer.Add(self.save_freq_ctrl, 1, wx.EXPAND)

        # Lambda L1
        self.lambda_l1_ctrl = wx.SpinCtrl(panel, value=str(self.settings.get_lambda_l1()))
        self.lambda_l1_ctrl.SetToolTip(wx.ToolTip(TOOLTIP_LAMBDA_L1))
        self.lambda_l1_ctrl.Bind(wx.EVT_SPINCTRL, self.on_settings_changed)
        grid_sizer.Add(wx.StaticText(panel, label="Lambda L1:"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid_sizer.Add(self.lambda_l1_ctrl, 1, wx.EXPAND)

        # Beta 1
        self.beta1_ctrl = wx.TextCtrl(panel, value=str(self.settings.get_beta1()))
        self.beta1_ctrl.SetToolTip(wx.ToolTip(TOOLTIP_BETA1))
        self.beta1_ctrl.Bind(wx.EVT_TEXT, self.on_settings_changed)
        grid_sizer.Add(wx.StaticText(panel, label="Beta 1:"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid_sizer.Add(self.beta1_ctrl, 1, wx.EXPAND)

        grid_sizer.AddGrowableCol(1, 1)
        sizer.Add(grid_sizer, 1, wx.EXPAND | wx.ALL, 5)
        return sizer

    def create_progress_panel(self):
        panel = wx.StaticBox(self, label="Training Progress")
        sizer = wx.StaticBoxSizer(panel, wx.VERTICAL)

        # Output folder
        output_folder_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.output_folder_picker = wx.DirPickerCtrl(panel, message="Select Output Folder", style=wx.DIRP_USE_TEXTCTRL)
        self.output_folder_picker.SetToolTip(wx.ToolTip(TOOLTIP_OUTPUT_FOLDER))
        self.output_folder_picker.SetPath(self.settings.get_training_output_path())
        self.output_folder_picker.Bind(wx.EVT_DIRPICKER_CHANGED, self.on_output_folder_changed)
        output_folder_sizer.Add(wx.StaticText(panel, label="Output Folder:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        output_folder_sizer.Add(self.output_folder_picker, 1, wx.EXPAND)
        sizer.Add(output_folder_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Model Filename
        model_filename_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.model_filename_ctrl = wx.TextCtrl(panel, value=self.settings.get_model_filename())
        self.model_filename_ctrl.SetToolTip(wx.ToolTip(TOOLTIP_MODEL_FILENAME))
        self.model_filename_ctrl.Bind(wx.EVT_TEXT, self.on_settings_changed)
        model_filename_sizer.Add(wx.StaticText(panel, label="Model Filename:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        model_filename_sizer.Add(self.model_filename_ctrl, 1, wx.EXPAND)
        sizer.Add(model_filename_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Continue training checkbox
        self.continue_checkbox = wx.CheckBox(panel, label="Continue training from existing model")
        self.continue_checkbox.SetToolTip(wx.ToolTip(TOOLTIP_CONTINUE_TRAINING))
        self.continue_checkbox.SetValue(self.settings.get_continue_training())
        self.continue_checkbox.Bind(wx.EVT_CHECKBOX, self.on_toggle_continue_training)
        sizer.Add(self.continue_checkbox, 0, wx.ALL, 5)

        # Model file picker
        model_path_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.model_path_picker = wx.FilePickerCtrl(
            panel, 
            message="Select Model Checkpoint", 
            wildcard="PyTorch models (*.pth)|*.pth",
            style=wx.FLP_USE_TEXTCTRL | wx.FLP_OPEN
        )
        self.model_path_picker.SetToolTip(wx.ToolTip("Select a model checkpoint (.pth) to continue training from."))
        self.model_path_picker.SetPath(self.settings.get_load_model_path())
        self.model_path_picker.Bind(wx.EVT_FILEPICKER_CHANGED, self.on_model_path_changed)
        model_path_sizer.Add(wx.StaticText(panel, label="Load Model:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        model_path_sizer.Add(self.model_path_picker, 1, wx.EXPAND)
        sizer.Add(model_path_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        self.model_path_picker.Enable(self.continue_checkbox.GetValue())

        # Train button
        train_button = wx.Button(panel, label="Train")
        train_button.Bind(wx.EVT_BUTTON, self.on_train)
        sizer.Add(train_button, 0, wx.ALIGN_CENTER | wx.ALL, 5)

        # Summary
        summary_label = wx.StaticText(panel, label="Training Summary:")
        sizer.Add(summary_label, 0, wx.LEFT | wx.TOP, 5)
        self.summary_ctrl = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.summary_ctrl.SetMinSize((-1, 100))
        sizer.Add(self.summary_ctrl, 1, wx.EXPAND | wx.ALL, 5)


        return sizer

    def on_folder_selected(self, event):
        path = event.GetPath()
        self.settings.set_training_source_path(path)
        self.settings.save()
        self.validate_data(path)

    def on_toggle_continue_training(self, event):
        is_checked = self.continue_checkbox.GetValue()
        self.model_path_picker.Enable(is_checked)
        self.settings.set_continue_training(is_checked)
        self.settings.save()

    def on_model_path_changed(self, event):
        path = event.GetPath()
        self.settings.set_load_model_path(path)
        self.settings.save()

    def on_output_folder_changed(self, event):
        path = event.GetPath()
        self.settings.set_training_output_path(path)
        self.settings.save()

    def on_settings_changed(self, event):
        self.settings.set_epochs(self.epochs_ctrl.GetValue())
        self.settings.set_batch_size(self.batch_size_ctrl.GetValue())
        self.settings.set_learning_rate(float(self.learning_rate_ctrl.GetValue()))
        self.settings.set_save_freq(self.save_freq_ctrl.GetValue())
        self.settings.set_model_filename(self.model_filename_ctrl.GetValue())
        self.settings.set_lambda_l1(self.lambda_l1_ctrl.GetValue())
        self.settings.set_beta1(float(self.beta1_ctrl.GetValue()))
        self.settings.save()

    def on_train(self, event):
        # Validate settings
        source_path = self.settings.get_training_source_path()
        output_path = self.settings.get_training_output_path()
        model_filename = self.settings.get_model_filename()
        
        if not source_path or not os.path.exists(source_path):
            wx.MessageBox("Please select a valid source folder.", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        if not output_path:
            wx.MessageBox("Please select an output folder.", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        if not model_filename:
            wx.MessageBox("Please enter a model filename.", "Error", wx.OK | wx.ICON_ERROR)
            return

        # Get continue training settings
        continue_training = self.continue_checkbox.GetValue()
        load_model_path = self.model_path_picker.GetPath() if continue_training else None

        if continue_training and (not load_model_path or not os.path.exists(load_model_path)):
            wx.MessageBox("Please select a valid model file to continue training.", "Error", wx.OK | wx.ICON_ERROR)
            return

        # Get training parameters
        epochs = self.epochs_ctrl.GetValue()
        batch_size = self.batch_size_ctrl.GetValue()
        try:
            learning_rate = float(self.learning_rate_ctrl.GetValue())
            beta1 = float(self.beta1_ctrl.GetValue())
        except ValueError:
            wx.MessageBox("Please enter a valid learning rate.", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        save_freq = self.save_freq_ctrl.GetValue()
        lambda_l1 = self.lambda_l1_ctrl.GetValue()
        
        # Validate that we have valid image pairs
        albedo_path = os.path.join(source_path, "albedo")
        roughness_path = os.path.join(source_path, "roughness")
        
        if not os.path.isdir(albedo_path) or not os.path.isdir(roughness_path):
            wx.MessageBox('Source folder must contain "albedo" and "roughness" subfolders.', "Error", wx.OK | wx.ICON_ERROR)
            return
        
        # Check if we have any common files
        albedo_files = set(os.listdir(albedo_path))
        roughness_files = set(os.listdir(roughness_path))
        common_files = albedo_files.intersection(roughness_files)
        
        if not common_files:
            wx.MessageBox("No matching image files found between albedo and roughness folders.", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        # Start training
        self.training_manager.start_training(
            source_path=source_path,
            output_path=output_path,
            model_filename=model_filename,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            save_freq=save_freq,
            lambda_l1=lambda_l1,
            beta1=beta1,
            load_model_path=load_model_path
        )

    def validate_data(self, path):
        if not path or not os.path.isdir(path):
            self.validation_text.SetLabel("Please select a valid source folder.")
            self.frame.SetStatusText("")
            return

        albedo_path = os.path.join(path, "albedo")
        roughness_path = os.path.join(path, "roughness")

        if not os.path.isdir(albedo_path) or not os.path.isdir(roughness_path):
            self.validation_text.SetLabel('Error: "albedo" and/or "roughness" subfolders not found.')
            self.frame.SetStatusText("")
            return

        albedo_files = {f for f in os.listdir(albedo_path) if os.path.isfile(os.path.join(albedo_path, f))}
        roughness_files = {f for f in os.listdir(roughness_path) if os.path.isfile(os.path.join(roughness_path, f))}

        common_files = albedo_files.intersection(roughness_files)
        total_to_scan = len(common_files)
        valid_pairs = 0
        
        # Show progress bar for scanning
        if self.frame:
            self.frame.set_progress(0, show=True)
        
        for i, f in enumerate(common_files):
            progress = ((i + 1) / total_to_scan) * 100 if total_to_scan > 0 else 100
            if self.frame:
                self.frame.SetStatusText(f"Scanning file {i+1}/{total_to_scan}: {f}")
                self.frame.set_progress(int(progress))
            try:
                albedo_img = Image.open(os.path.join(albedo_path, f))
                roughness_img = Image.open(os.path.join(roughness_path, f))
                if albedo_img.size == roughness_img.size:
                    valid_pairs += 1
            except Exception as e:
                print(f"Error processing file {f}: {e}")

        if valid_pairs > 0:
            self.validation_text.SetLabel(f"Found {valid_pairs} valid image pairs.")
        else:
            self.validation_text.SetLabel("No valid image pairs found in the selected folder.")
        
        if self.frame:
            self.frame.SetStatusText("Scan complete.")
            self.frame.hide_progress()  # Hide progress bar when complete

    def update_summary(self, summary_text):
        self.summary_ctrl.SetValue(summary_text)
