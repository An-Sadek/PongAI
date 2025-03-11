import pygame


class Paddle:
    VEL = 6  # Tốc độ thanh
    WIDTH = 20 # Chiều rộng
    HEIGHT = 100 # Chiều dài

    # Khởi tạo vị trí ban đầu
    # Vị trí của paddle được tính bằng góc trên bên trái
    def __init__(self, x, y):
        self.x = self.original_x = x
        self.y = self.original_y = y

    # Vẽ thanh
    def draw(self, win):
        pygame.draw.rect(
            win, (255, 255, 255), (self.x, self.y, self.WIDTH, self.HEIGHT))

    # Di chuyển thanh dựa trên tốc độ
    def move(self, up=True):
        if up:
            self.y -= self.VEL  
        else:
            self.y += self.VEL  

    # Đặt lại vị trí ban đầu
    def reset(self):
        self.x = self.original_x
        self.y = self.original_y
