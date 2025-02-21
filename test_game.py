import pygame
import neat
import os
import pickle
from pong import Game  # Import lớp Game từ file pong.py


class PongGame:
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.ball = self.game.ball
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.win_score = 5  # Số điểm để thắng

    def display_end_screen(self, winner_text):
        """
        Hiển thị màn hình kết thúc với thông báo người chiến thắng
        """
        font = pygame.font.SysFont('Arial', 30)
        text = font.render(winner_text, True, (255, 255, 255))
        retry_text = font.render('Press R to Play Again', True, (255, 255, 255))
        quit_text = font.render('Press Q to Quit', True, (255, 255, 255))

        self.game.window.fill((0, 0, 0))  # Làm nền đen
        self.game.window.blit(text, (self.game.window.get_width() / 2 - text.get_width() / 2, self.game.window.get_height() / 3))
        self.game.window.blit(retry_text, (self.game.window.get_width() / 2 - retry_text.get_width() / 2, self.game.window.get_height() / 2))
        self.game.window.blit(quit_text, (self.game.window.get_width() / 2 - quit_text.get_width() / 2, self.game.window.get_height() / 1.5))
        pygame.display.update()

    def wait_for_restart_or_quit(self):
        """
        Chờ người dùng nhấn R để chơi lại hoặc Q để thoát
        """
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False  # Thoát
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Nhấn R để chơi lại
                        return True
                    elif event.key == pygame.K_q:  # Nhấn Q để thoát
                        return False

    def test_ai(self, net):
        """
        Test the AI against a human player by passing a NEAT neural network
        """
        clock = pygame.time.Clock()
        run = True
        while run:
            clock.tick(60)
            game_info = self.game.loop()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    return  # Thoát game ngay lập tức

            # Điều khiển AI
            output = net.activate((self.right_paddle.y, abs(
                self.right_paddle.x - self.ball.x), self.ball.y))
            decision = output.index(max(output))

            if decision == 1:  # AI moves up
                self.game.move_paddle(left=False, up=True)
            elif decision == 2:  # AI moves down
                self.game.move_paddle(left=False, up=False)

            # Điều khiển người chơi bằng bàn phím
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.game.move_paddle(left=True, up=True)
            elif keys[pygame.K_s]:
                self.game.move_paddle(left=True, up=False)

            # Vẽ trò chơi
            self.game.draw(draw_score=True)
            pygame.display.update()

            # Kiểm tra nếu một trong hai người chơi thắng
            if game_info.left_score >= self.win_score or game_info.right_score >= self.win_score:
                winner = "Left Player Wins!" if game_info.left_score >= self.win_score else "Right Player Wins!"
                self.display_end_screen(winner)
                if self.wait_for_restart_or_quit():
                    self.game.reset()  # Reset game thay vì thoát
                else:
                    run = False  # Thoát nếu người dùng nhấn Q

        pygame.quit()


def test_best_network(config):
    """
    Test the best network saved in best.pickle.
    """
    # Load model tốt nhất từ file best.pickle
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    # Khởi tạo cửa sổ game
    width, height = 700, 500
    win = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pong - Test AI")

    # Khởi tạo trò chơi
    pong = PongGame(win, width, height)
    pong.test_ai(winner_net)


if __name__ == '__main__':
    # Load cấu hình NEAT
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Chạy hàm test AI
    test_best_network(config)
