import pygame
import time

from game_settings import Game_Settings
from earthworm import Earthworm
import game_functions as gf


def run_game():
	""" main function of the game """
	pygame.init()
	settings = Game_Settings() 
	screen = pygame.display.set_mode((settings.screen_width, settings.screen_height))
	earthworm = Earthworm(screen,settings)
	pygame.display.set_caption("EARTHWORM")
	while True:
		gf.check_events(earthworm,settings)
		earthworm.move()
		gf.update_screen(screen, settings,earthworm)
		time.sleep(0.05)


if __name__ == '__main__':
	run_game()

