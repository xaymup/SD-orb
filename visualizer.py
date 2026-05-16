import pygame
import numpy as np
import cv2

class Visualizer:
    def __init__(self, width=512, height=512):
        self.width = width
        self.height = height
        
        # Warp engine grids
        Y, X = np.indices((height, width))
        self.X_NORM = (X / (width / 2)) - 1.0
        self.Y_NORM = (Y / (height / 2)) - 1.0
        self.R = np.sqrt(self.X_NORM**2 + self.Y_NORM**2)
        self.ANG = np.arctan2(self.Y_NORM, self.X_NORM)
        
        self._angle_accumulator = 0.0

    def apply_feedback(self, input_array, bands, config):
        """
        Applies feedback transformation (warp + decay + reactive geometry) 
        to an existing image array (normalized 0-1).
        """
        bass, mids, highs = bands
        zoom_base = config['zoom_base']
        zoom_sens = config['zoom_sens']
        rot_base  = config['rot_base']
        rot_sens  = config['rot_sens']

        # 1. Transform back to 0-255 uint8 for OpenCV processing
        # input_array is (H, W, 3) RGB
        frame_uint8 = (input_array * 255).astype(np.uint8)
        
        # 2. Calculate Warp
        zoom = zoom_base - (bass * 0.05 * zoom_sens)
        rot = rot_base + (mids * 0.05 * rot_sens)
        
        new_r = self.R * zoom
        new_ang = self.ANG + rot

        map_x = ((np.cos(new_ang) * new_r + 1.0) * (self.width / 2)).astype(np.float32)
        map_y = ((np.sin(new_ang) * new_r + 1.0) * (self.height / 2)).astype(np.float32)

        # 3. Apply warp using OpenCV for bilinear interpolation (much smoother)
        warped_data = cv2.remap(frame_uint8, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # 4. Decay
        decay = 0.92 + (highs * 0.06)
        warped_data = (warped_data * decay).astype(np.uint8)

        # 5. Re-create surface and draw geometry
        # Transpose for Pygame: (H, W, 3) -> (W, H, 3)
        warped_surf = pygame.surfarray.make_surface(warped_data.transpose(1, 0, 2))
        
        self._angle_accumulator += 0.02 + bass * 0.05
        cx, cy = self.width // 2, self.height // 2
        
        for i in range(2):
            dist = int(((i * 150 + self._angle_accumulator * 50) % 300) + (bass * 30))
            r_c = min(255, int(50 + bass * 200))
            color = (r_c, 100, 255)
            pygame.draw.rect(warped_surf, color, (cx - dist, cy - dist, dist * 2, dist * 2), 1)

        # 6. Convert back to normalized float32 (H, W, 3)
        return np.transpose(pygame.surfarray.array3d(warped_surf), (1, 0, 2)).astype(np.float32) / 255.0
