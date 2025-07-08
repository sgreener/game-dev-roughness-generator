import configparser

class Settings:
    def __init__(self, file_path="settings.ini"):
        self.file_path = file_path
        self.config = configparser.ConfigParser()
        self.load()

    def load(self):
        self.config.read(self.file_path)
        if not self.config.has_section('window'):
            self.config.add_section('window')
            self.config.set('window', 'x', '100')
            self.config.set('window', 'y', '100')
            self.config.set('window', 'width', '800')
            self.config.set('window', 'height', '600')
            self.config.set('window', 'current_tab', '0')
        else:
            # Ensure all required options exist in the window section
            if not self.config.has_option('window', 'current_tab'):
                self.config.set('window', 'current_tab', '0')
        if not self.config.has_section('training'):
            self.config.add_section('training')
            self.config.set('training', 'source_path', '')
            self.config.set('training', 'output_path', '')
            self.config.set('training', 'model_filename', 'final_model.pth')
            self.config.set('training', 'epochs', '200')
            self.config.set('training', 'batch_size', '1')
            self.config.set('training', 'learning_rate', '0.0002')
            self.config.set('training', 'save_freq', '50')
            self.config.set('training', 'lambda_l1', '100')
            self.config.set('training', 'beta1', '0.5')
            self.config.set('training', 'continue_training', 'False')
            self.config.set('training', 'load_model_path', '')
        if not self.config.has_section('generate'):
            self.config.add_section('generate')
            self.config.set('generate', 'model_path', '')
            self.config.set('generate', 'output_path', '')
            self.config.set('generate', 'tile_size', '64')
            self.config.set('generate', 'gpu_threads', '4')
            self.config.set('generate', 'enable_second_pass', 'False')
            self.config.set('generate', 'edge_wrapping', 'True')
            self.config.set('generate', 'feather_second_pass', 'True')
            self.config.set('generate', 'transform_flip_horizontal', 'False')
            self.config.set('generate', 'transform_flip_vertical', 'False')
            self.config.set('generate', 'transform_rotate_90', 'False')
            self.config.set('generate', 'transform_rotate_180', 'False')
            self.config.set('generate', 'transform_rotate_270', 'False')
        else:
            # Ensure all required options exist in the generate section
            if not self.config.has_option('generate', 'tile_size'):
                self.config.set('generate', 'tile_size', '64')
            if not self.config.has_option('generate', 'gpu_threads'):
                self.config.set('generate', 'gpu_threads', '4')
            if not self.config.has_option('generate', 'enable_second_pass'):
                self.config.set('generate', 'enable_second_pass', 'False')
            if not self.config.has_option('generate', 'edge_wrapping'):
                self.config.set('generate', 'edge_wrapping', 'True')
            if not self.config.has_option('generate', 'feather_second_pass'):
                self.config.set('generate', 'feather_second_pass', 'True')
            if not self.config.has_option('generate', 'transform_flip_horizontal'):
                self.config.set('generate', 'transform_flip_horizontal', 'False')
            if not self.config.has_option('generate', 'transform_flip_vertical'):
                self.config.set('generate', 'transform_flip_vertical', 'False')
            if not self.config.has_option('generate', 'transform_rotate_90'):
                self.config.set('generate', 'transform_rotate_90', 'False')
            if not self.config.has_option('generate', 'transform_rotate_180'):
                self.config.set('generate', 'transform_rotate_180', 'False')
            if not self.config.has_option('generate', 'transform_rotate_270'):
                self.config.set('generate', 'transform_rotate_270', 'False')

    def save(self, event=None):
        with open(self.file_path, 'w') as configfile:
            self.config.write(configfile)

    def get_window_position(self):
        x = self.config.getint('window', 'x')
        y = self.config.getint('window', 'y')
        return (x, y)

    def set_window_position(self, pos):
        self.config.set('window', 'x', str(pos[0]))
        self.config.set('window', 'y', str(pos[1]))

    def get_window_size(self):
        width = self.config.getint('window', 'width')
        height = self.config.getint('window', 'height')
        return (width, height)

    def set_window_size(self, size):
        self.config.set('window', 'width', str(size[0]))
        self.config.set('window', 'height', str(size[1]))

    def get_current_tab(self):
        return self.config.getint('window', 'current_tab')

    def set_current_tab(self, tab_index):
        self.config.set('window', 'current_tab', str(tab_index))

    def get_training_source_path(self):
        return self.config.get('training', 'source_path')

    def set_training_source_path(self, path):
        self.config.set('training', 'source_path', path)

    def get_training_output_path(self):
        return self.config.get('training', 'output_path')

    def set_training_output_path(self, path):
        self.config.set('training', 'output_path', path)

    def get_model_filename(self):
        return self.config.get('training', 'model_filename')

    def set_model_filename(self, filename):
        self.config.set('training', 'model_filename', filename)

    def get_epochs(self):
        return self.config.getint('training', 'epochs')

    def set_epochs(self, value):
        self.config.set('training', 'epochs', str(value))

    def get_batch_size(self):
        return self.config.getint('training', 'batch_size')

    def set_batch_size(self, value):
        self.config.set('training', 'batch_size', str(value))

    def get_learning_rate(self):
        return self.config.getfloat('training', 'learning_rate')

    def set_learning_rate(self, value):
        self.config.set('training', 'learning_rate', str(value))

    def get_save_freq(self):
        return self.config.getint('training', 'save_freq')

    def set_save_freq(self, value):
        self.config.set('training', 'save_freq', str(value))

    def get_lambda_l1(self):
        return self.config.getint('training', 'lambda_l1')

    def set_lambda_l1(self, value):
        self.config.set('training', 'lambda_l1', str(value))

    def get_beta1(self):
        return self.config.getfloat('training', 'beta1')

    def set_beta1(self, value):
        self.config.set('training', 'beta1', str(value))

    def get_continue_training(self):
        return self.config.getboolean('training', 'continue_training')

    def set_continue_training(self, value):
        self.config.set('training', 'continue_training', str(value))

    def get_load_model_path(self):
        return self.config.get('training', 'load_model_path')

    def set_load_model_path(self, path):
        self.config.set('training', 'load_model_path', path)

    # Generate tab settings
    def get_generate_model_path(self):
        return self.config.get('generate', 'model_path')

    def set_generate_model_path(self, path):
        self.config.set('generate', 'model_path', path)

    def get_generate_output_path(self):
        return self.config.get('generate', 'output_path')

    def set_generate_output_path(self, path):
        self.config.set('generate', 'output_path', path)

    def get_use_tiled_generation(self):
        return self.config.getboolean('generate', 'use_tiled_generation')

    def set_use_tiled_generation(self, value):
        self.config.set('generate', 'use_tiled_generation', str(value))

    def get_tile_size(self):
        return self.config.getint('generate', 'tile_size')

    def set_tile_size(self, value):
        self.config.set('generate', 'tile_size', str(value))

    def get_tile_overlap(self):
        return self.config.getint('generate', 'tile_overlap')

    def set_tile_overlap(self, value):
        self.config.set('generate', 'tile_overlap', str(value))

    def get_blend_edges(self):
        return self.config.getboolean('generate', 'blend_edges')

    def set_blend_edges(self, value):
        self.config.set('generate', 'blend_edges', str(value))

    def get_seamless_tiling(self):
        return self.config.getboolean('generate', 'seamless_tiling')

    def set_seamless_tiling(self, value):
        self.config.set('generate', 'seamless_tiling', str(value))

    def get_generate_tile_size(self):
        """Get the tile size for generation"""
        return self.config.getint('generate', 'tile_size')
    
    def set_generate_tile_size(self, value):
        """Set the tile size for generation"""
        self.config.set('generate', 'tile_size', str(value))

    def get_generate_gpu_threads(self):
        """Get the GPU threads count for generation"""
        return self.config.getint('generate', 'gpu_threads')
    
    def set_generate_gpu_threads(self, value):
        """Set the GPU threads count for generation"""
        self.config.set('generate', 'gpu_threads', str(value))

    def get_generate_enable_second_pass(self):
        """Get the enable second pass setting for generation"""
        return self.config.getboolean('generate', 'enable_second_pass')
    
    def set_generate_enable_second_pass(self, value):
        """Set the enable second pass setting for generation"""
        self.config.set('generate', 'enable_second_pass', str(value))
    
    def get_generate_edge_wrapping(self):
        """Get the edge wrapping setting for generation"""
        return self.config.getboolean('generate', 'edge_wrapping')
    
    def set_generate_edge_wrapping(self, value):
        """Set the edge wrapping setting for generation"""
        self.config.set('generate', 'edge_wrapping', str(value))

    def get_generate_feather_second_pass(self):
        """Get the feather second pass setting for generation"""
        return self.config.getboolean('generate', 'feather_second_pass')
    
    def set_generate_feather_second_pass(self, value):
        """Set the feather second pass setting for generation"""
        self.config.set('generate', 'feather_second_pass', str(value))

    # Transformation settings
    def get_transform_flip_horizontal(self):
        """Get the flip horizontal transformation setting"""
        return self.config.getboolean('generate', 'transform_flip_horizontal')
    
    def set_transform_flip_horizontal(self, value):
        """Set the flip horizontal transformation setting"""
        self.config.set('generate', 'transform_flip_horizontal', str(value))

    def get_transform_flip_vertical(self):
        """Get the flip vertical transformation setting"""
        return self.config.getboolean('generate', 'transform_flip_vertical')
    
    def set_transform_flip_vertical(self, value):
        """Set the flip vertical transformation setting"""
        self.config.set('generate', 'transform_flip_vertical', str(value))

    def get_transform_rotate_90(self):
        """Get the rotate 90 transformation setting"""
        return self.config.getboolean('generate', 'transform_rotate_90')
    
    def set_transform_rotate_90(self, value):
        """Set the rotate 90 transformation setting"""
        self.config.set('generate', 'transform_rotate_90', str(value))

    def get_transform_rotate_180(self):
        """Get the rotate 180 transformation setting"""
        return self.config.getboolean('generate', 'transform_rotate_180')
    
    def set_transform_rotate_180(self, value):
        """Set the rotate 180 transformation setting"""
        self.config.set('generate', 'transform_rotate_180', str(value))

    def get_transform_rotate_270(self):
        """Get the rotate 270 transformation setting"""
        return self.config.getboolean('generate', 'transform_rotate_270')
    
    def set_transform_rotate_270(self, value):
        """Set the rotate 270 transformation setting"""
        self.config.set('generate', 'transform_rotate_270', str(value))
