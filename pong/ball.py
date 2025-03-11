import pygame
import math
import random


class Ball:
    MAX_VEL = 15 # Tốc độ bóng
    RADIUS = 7 # Bán kính bóng

    def __init__(self, x, y):
        # Khởi tạo vị trí ban đầu
        self.x = self.original_x = x
        self.y = self.original_y = y
        
        angle = self._get_random_angle(-30, 30, [0])
        pos = 1 if random.random() < 0.5 else -1

        # Chỉnh sửa để giảm tốc độ bóng thêm
        self.x_vel = pos * abs(math.cos(angle) * self.MAX_VEL) / 2  # Giảm tốc độ theo trục X
        self.y_vel = math.sin(angle) * self.MAX_VEL / 2  # Giảm tốc độ theo trục Y

    # Tạo góc ngẫu nhiên
    def _get_random_angle(self, min_angle, max_angle, excluded):
        angle = 0
        while angle in excluded:
            angle = math.radians(random.randrange(min_angle, max_angle))

        return angle

    # Vẽ bóng
    def draw(self, win):
        pygame.draw.circle(win, (255, 255, 255), (self.x, self.y), self.RADIUS)

    # Di chuyển
    def move(self):
        self.x += self.x_vel
        self.y += self.y_vel

    # Reset theo hàm khởi tạo
    def reset(self):
        self.x = self.original_x
        self.y = self.original_y

        # Lấy góc ngẫu nhiên
        angle = self._get_random_angle(-30, 30, [0])

        # Giảm tốc độ sau khi reset để có thể nhìn dễ hơn
        #x_vel = abs(math.cos(angle) * self.MAX_VEL) / 2  # Giảm tốc độ theo trục X
        y_vel = math.sin(angle) * self.MAX_VEL / 3  # Giảm tốc độ theo trục Y

        self.y_vel = y_vel
        self.x_vel *= -1
