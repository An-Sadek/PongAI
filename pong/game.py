from .paddle import Paddle
from .ball import Ball
import pygame
import random
pygame.init()


class GameInformation:
    def __init__(self, left_hits, right_hits, left_score, right_score):
        self.left_hits = left_hits
        self.right_hits = right_hits
        self.left_score = left_score
        self.right_score = right_score


class Game:
    """
    To use this class simply initialize and instance and call the .loop() method
    inside of a pygame event loop (i.e while loop). Inside of your event loop
    you can call the .draw() and .move_paddle() methods according to your use case.
    Use the information returned from .loop() to determine when to end the game by calling
    .reset().
    """
    SCORE_FONT = pygame.font.SysFont("comicsans", 50) # Tuỳ chỉnh font

    # Màu RGB
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)

    def __init__(self, window, window_width, window_height):
        self.window_width = window_width
        self.window_height = window_height

        self.left_paddle = Paddle(
            10, self.window_height // 2 - Paddle.HEIGHT // 2)
        self.right_paddle = Paddle(
            self.window_width - 10 - Paddle.WIDTH, self.window_height // 2 - Paddle.HEIGHT//2)
        self.ball = Ball(self.window_width // 2, self.window_height // 2)

        self.left_score = 0
        self.right_score = 0
        self.left_hits = 0
        self.right_hits = 0
        self.window = window

    def _draw_score(self):
        left_score_text = self.SCORE_FONT.render(
            f"{self.left_score}", 1, self.WHITE)
        right_score_text = self.SCORE_FONT.render(
            f"{self.right_score}", 1, self.WHITE)
        
        # Vẽ điểm của player 1
        self.window.blit(left_score_text, (self.window_width //
                                           4 - left_score_text.get_width()//2, 20))
        
        # Vẽ điểm của player 2
        self.window.blit(right_score_text, (self.window_width * (3/4) -
                                            right_score_text.get_width()//2, 20))

    def _draw_hits(self):
        hits_text = self.SCORE_FONT.render(
            f"{self.left_hits + self.right_hits}", 1, self.RED)
        self.window.blit(hits_text, (self.window_width //
                                     2 - hits_text.get_width()//2, 10))

    # Vẽ đường chia giữa
    def _draw_divider(self):
        for i in range(10, self.window_height, self.window_height//20):
            if i % 2 == 1:
                continue
            pygame.draw.rect(
                self.window, self.WHITE, (self.window_width//2 - 5, i, 10, self.window_height//20))

    # Thanh chắn chạm với bóng
    def _handle_collision(self):
        ball = self.ball
        left_paddle = self.left_paddle
        right_paddle = self.right_paddle

        # Nếu chạm vào 2 rìa trên dưới thì đổi hướng y
        if ball.y + ball.RADIUS >= self.window_height:
            ball.y_vel *= -1 
        elif ball.y - ball.RADIUS <= 0:
            ball.y_vel *= -1 

        # Bóng di chuyển sang trái
        if ball.x_vel < 0:

            # Kiểm tra vị trí y của bóng so với paddle trái
            if ball.y >= left_paddle.y and ball.y <= left_paddle.y + Paddle.HEIGHT:

                # Kiểm tra paddle có chạm vào bóng chưa
                if ball.x - ball.RADIUS <= left_paddle.x + Paddle.WIDTH:
                    ball.x_vel *= -1 # Đổi hướng x

                    # Đánh trúng ở giữa tốc độ chậm
                    middle_y = left_paddle.y + Paddle.HEIGHT / 2 # Tính vị trí chính giữa
                    difference_in_y = middle_y - ball.y # Tính khoảng cách từ điểm chính giữa đến tâm của trái bóng
                    reduction_factor = (Paddle.HEIGHT / 2) / ball.MAX_VEL # Tính hệ số giảm tốc
                    y_vel = difference_in_y / reduction_factor # Tốc độ mới, càng gần giữa tốc độ càng chậm
                    ball.y_vel = -1 * y_vel # Đổi hướng y
                    self.left_hits += 1 # Tính điểm bến trái

        # Bóng di chuyển sang phải, còn lại tương tự
        else:
            if ball.y >= right_paddle.y and ball.y <= right_paddle.y + Paddle.HEIGHT:
                if ball.x + ball.RADIUS >= right_paddle.x:
                    ball.x_vel *= -1

                    middle_y = right_paddle.y + Paddle.HEIGHT / 2
                    difference_in_y = middle_y - ball.y
                    reduction_factor = (Paddle.HEIGHT / 2) / ball.MAX_VEL
                    y_vel = difference_in_y / reduction_factor
                    ball.y_vel = -1 * y_vel
                    self.right_hits += 1

    def draw(self, draw_score=True, draw_hits=False):
        self.window.fill(self.BLACK)

        self._draw_divider()

        if draw_score:
            self._draw_score()

        if draw_hits:
            self._draw_hits()

        for paddle in [self.left_paddle, self.right_paddle]:
            paddle.draw(self.window)

        self.ball.draw(self.window)

    def move_paddle(self, left=True, up=True):
        """
        Kiểm tra xem paddle có di chuyển được hay không

        Giá trị trả về là boolean
            Nếu paddle di chuyển ra ngoài màn hình thì trả về False
        """
        # Paddle trái
        if left:

            # Paddle vượt quá màn hình phía trên
            if up and self.left_paddle.y - Paddle.VEL < 0:
                return False
            
            # Paddle vượt quá màn hình phía dưới
            if not up and self.left_paddle.y + Paddle.HEIGHT > self.window_height:
                return False
            self.left_paddle.move(up)

        # Paddle phải, cũng tương tự
        else:
            if up and self.right_paddle.y - Paddle.VEL < 0:
                return False
            if not up and self.right_paddle.y + Paddle.HEIGHT > self.window_height:
                return False
            self.right_paddle.move(up)

        return True

    def loop(self):
        """
        Chạy vòng lặp chạy game
        """
        self.ball.move()
        self._handle_collision()

        if self.ball.x < 0:
            self.ball.reset()
            self.right_score += 1
        elif self.ball.x > self.window_width:
            self.ball.reset()
            self.left_score += 1

        # Trả về thông tin game
        game_info = GameInformation(
            self.left_hits, self.right_hits, self.left_score, self.right_score)

        return game_info

    def reset(self):
        """Reset lại game"""
        self.ball.reset()
        self.left_paddle.reset()
        self.right_paddle.reset()
        self.left_score = 0
        self.right_score = 0
        self.left_hits = 0
        self.right_hits = 0
