import os
import time
import threading
import torch
import wx
from src.model import Pix2PixGAN
from src.dataset import create_data_loader, tensor_to_pil
from src.training_window import TrainingProgressWindow


class TrainingManager:
    """Manages the training process and communicates with the GUI"""
    
    def __init__(self, train_tab):
        self.train_tab = train_tab
        self.parent_frame = train_tab.frame
        self.training_thread = None
        self.progress_window = None
        self.is_training = False
        
    def start_training(self, source_path, output_path, model_filename, epochs, batch_size, learning_rate, save_freq, lambda_l1, beta1, load_model_path=None):
        """Start the training process"""
        if self.is_training:
            wx.MessageBox("Training is already in progress!", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        # Validate paths
        if not os.path.exists(source_path):
            wx.MessageBox("Source path does not exist!", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path)
            except Exception as e:
                wx.MessageBox(f"Could not create output directory: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
                return
        
        # Check GPU availability
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            dlg = wx.MessageDialog(
                self.parent_frame,
                "CUDA is not available. Training will run on CPU which will be very slow. Continue?",
                "No GPU Detected",
                wx.YES_NO | wx.ICON_WARNING
            )
            if dlg.ShowModal() == wx.ID_NO:
                dlg.Destroy()
                return
            dlg.Destroy()
        
        # Create progress window
        self.progress_window = TrainingProgressWindow(self.parent_frame, self)
        self.progress_window.Show()
        
        # Hide the main window during training
        self.parent_frame.hide_main_window()
        
        # Start training in separate thread
        self.training_thread = threading.Thread(
            target=self._training_worker,
            args=(source_path, output_path, model_filename, epochs, batch_size, learning_rate, save_freq, lambda_l1, beta1, device, load_model_path),
            daemon=True
        )
        self.is_training = True
        self.training_thread.start()
    
    def _training_worker(self, source_path, output_path, model_filename, epochs, batch_size, learning_rate, save_freq, lambda_l1, beta1, device, load_model_path=None):
        """Worker function that runs the actual training"""
        try:
            start_time = time.time()
            # Initialize model
            model = Pix2PixGAN(device=device, lr=learning_rate, lambda_l1=lambda_l1, beta1=beta1)
            
            start_epoch = 0
            if load_model_path:
                try:
                    start_epoch = model.load_checkpoint(load_model_path)
                    wx.CallAfter(self._update_status, f"Resuming training from epoch {start_epoch}")
                except Exception as e:
                    wx.CallAfter(self._show_error, f"Error loading model: {str(e)}")
                    return

            # Create data loader
            wx.CallAfter(self._update_status, "Loading dataset...")
            dataloader, dataset_size = create_data_loader(
                source_path, 
                batch_size=batch_size,
                image_size=256,
                is_training=True,
                num_workers=0  # Set to 0 for Windows compatibility
            )
            
            if dataset_size == 0:
                wx.CallAfter(self._show_error, "No valid image pairs found in the dataset!")
                return
            
            wx.CallAfter(self._update_status, f"Dataset loaded: {dataset_size} image pairs")
            
            # Training loop
            start_time_training = time.time()
            total_batches = len(dataloader)
            losses = {}
            
            for epoch in range(start_epoch, epochs):
                if self.progress_window and self.progress_window.is_cancelled():
                    wx.CallAfter(self._update_status, "Training cancelled by user")
                    break
                
                epoch_start_time = time.time()
                
                for batch_idx, batch in enumerate(dataloader):
                    if self.progress_window and self.progress_window.is_cancelled():
                        break
                    
                    # Move data to device
                    real_a = batch['albedo'].to(device)  # Albedo images
                    real_b = batch['roughness'].to(device)  # Roughness images
                    
                    # Train step
                    losses = model.train_step(real_a, real_b)
                    
                    # Get memory usage
                    memory_allocated = torch.cuda.memory_allocated(device) / (1024**2) if device == 'cuda' else 0

                    # Update progress
                    elapsed_time = time.time() - start_time_training
                    wx.CallAfter(
                        self._update_progress,
                        epoch + 1, epochs, batch_idx + 1, total_batches,
                        losses, elapsed_time, memory_allocated
                    )
                
                # Save checkpoint at specified frequency
                if save_freq > 0 and (epoch + 1) % save_freq == 0:
                    base, ext = os.path.splitext(model_filename)
                    checkpoint_filename = f"{base}_epoch_{epoch + 1}{ext}"
                    checkpoint_path = os.path.join(output_path, checkpoint_filename)
                    model.save_checkpoint(checkpoint_path, epoch + 1, losses)
                    wx.CallAfter(self._update_status, f"Checkpoint saved: {checkpoint_filename}")
                
                # Save sample images periodically
                if (epoch + 1) % 10 == 0:
                    self._save_sample_images(model, dataloader, output_path, epoch + 1, device)
            
            # If training was cancelled, exit here
            if self.progress_window and self.progress_window.is_cancelled():
                return

            # Save final model
            final_model_path = os.path.join(output_path, model_filename)
            model.save_checkpoint(final_model_path, epochs, losses)
            
            # Training completed
            end_time = time.time()
            total_training_time = end_time - start_time
            wx.CallAfter(self._training_completed, total_training_time, losses)
            
        except Exception as e:
            wx.CallAfter(self._show_error, f"Training error: {str(e)}")
        finally:
            self.is_training = False
    
    def _save_sample_images(self, model, dataloader, output_path, epoch, device):
        """Save sample generated images"""
        try:
            model.generator.eval()
            with torch.no_grad():
                # Get a batch for sampling
                batch = next(iter(dataloader))
                real_a = batch['albedo'].to(device)
                real_b = batch['roughness'].to(device)
                
                # Generate fake images
                fake_b = model.generator(real_a)
                
                # Save first image from batch
                sample_dir = os.path.join(output_path, "samples")
                os.makedirs(sample_dir, exist_ok=True)
                
                # Convert tensors to PIL images
                real_albedo = tensor_to_pil(real_a[0])
                real_roughness = tensor_to_pil(real_b[0])
                fake_roughness = tensor_to_pil(fake_b[0])
                
                # Save images
                real_albedo.save(os.path.join(sample_dir, f"epoch_{epoch}_real_albedo.png"))
                real_roughness.save(os.path.join(sample_dir, f"epoch_{epoch}_real_roughness.png"))
                fake_roughness.save(os.path.join(sample_dir, f"epoch_{epoch}_fake_roughness.png"))
                
            model.generator.train()
        except Exception as e:
            print(f"Error saving sample images: {e}")
    
    def _update_progress(self, epoch, total_epochs, batch, total_batches, losses, elapsed_time, memory_allocated):
        """Update progress in the GUI thread"""
        if self.progress_window:
            self.progress_window.update_progress(epoch, total_epochs, batch, total_batches, losses, elapsed_time, memory_allocated)
    
    def _update_status(self, message):
        """Update status message in the GUI thread"""
        if self.progress_window:
            self.progress_window.status_text.SetLabel(message)
    
    def _show_error(self, message):
        """Show error message and clean up"""
        wx.MessageBox(message, "Training Error", wx.OK | wx.ICON_ERROR)
        if self.progress_window:
            self.progress_window.Close()
        self.is_training = False
        # Ensure main window is shown when there's an error
        self._ensure_main_window_visible()
    
    def _training_completed(self, total_training_time, final_losses):
        """Handle training completion"""
        if self.progress_window:
            self.progress_window.training_completed()
        wx.MessageBox("Training completed successfully!", "Training Complete", wx.OK | wx.ICON_INFORMATION)
        summary = self._generate_summary(total_training_time, final_losses)
        wx.CallAfter(self.train_tab.update_summary, summary)
    
    def stop_training(self):
        """Stop the training process"""
        if self.progress_window:
            self.progress_window.cancelled = True
        self.is_training = False
        # Ensure main window is shown when training stops
        self._ensure_main_window_visible()
    
    def _ensure_main_window_visible(self):
        """Ensure the main window is visible"""
        if self.parent_frame:
            self.parent_frame.show_main_window()

    def _generate_summary(self, total_training_time, final_losses):
        """Generate a summary of the training session."""
        summary = f"Total Training Time: {total_training_time:.2f} seconds\n"
        summary += "Final Losses:\n"
        for name, value in final_losses.items():
            summary += f"  {name}: {value:.4f}\n"
        return summary
