import numpy as np

class BoxGridProcess:
    def __init__(self, frame):
        self.frame = frame
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0

    def process_box(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        grid_size = 10
        # 计算得到每一个小格的长款
        grid_width = (x2 - x1) // grid_size
        grid_height = (y2 - y1) // grid_width

        # 初始化GRB均值的数组
        rgb_means = np.zeros((grid_size, grid_size, 3), dtype=np.float32)

        for i in range(grid_size):
            for j in range(grid_size):
                grid_x1 = int(x1) + i * grid_width
                grid_y1 = int(y1) + j * grid_height
                grid_x2 = grid_x1 + grid_size
                grid_y2 = grid_y1 + grid_size

                # 计算每一个小格的RGB并计算均值
                grid_rgb = self.frame[int(grid_y1): int(grid_y2), int(grid_x1):int(grid_x2), : ]
                grid_mean = np.mean(grid_rgb, axis=(0,1))

                rgb_means[i,j,:] = grid_mean
        return [x1, y1, x2, y2, rgb_means]
