import pygame
import sys
import random


def update_screen(screen, settings, earthworm):
	""" update the screen """
	screen.fill(settings.bg_color)
	pygame.draw.rect(screen, settings.wall_color, (0,0,settings.screen_height, settings.screen_height), 1)
	
	if earthworm_hit_itself(earthworm) or earthworm_hit_wall(earthworm):
		earthworm.direction = 0
		settings.game_active = False

	check_food(settings,screen,earthworm)
	earthworm.update_earthworm()
	pygame.display.flip()

def check_food(settings,screen,earthworm):
	if  settings.food_exits:
		pygame.draw.rect(screen, settings.food_color, (settings.food_x * 10 ,settings.food_y * 10,10,10))
	else:
		create_food(screen,settings,earthworm)
		settings.food_exits = True

def check_events(earthworm,settings):
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			sys.exit()
		elif event.type == pygame.KEYDOWN and settings.game_active:
			if event.key == pygame.K_q:
				sys.exit()
			elif event.key == pygame.K_RIGHT and earthworm.direction != 3 :
				earthworm.direction = 4
			elif event.key == pygame.K_LEFT  and earthworm.direction != 4 and earthworm.direction != 0:
				earthworm.direction = 3
			elif event.key == pygame.K_DOWN  and earthworm.direction != 1:
				earthworm.direction = 2
			elif event.key == pygame.K_UP    and earthworm.direction != 2:
				earthworm.direction = 1

def create_food(screen,settings,earthworm):
	x = random.randint(1,59)
	y = random.randint(1,59)
	if (x,y) in earthworm.body:
		create_food(screen,settings,earthworm)
	else:
		settings.food_x = x
		settings.food_y = y
		pygame.draw.rect(screen, settings.food_color, (x * 10 ,y * 10,10,10))


def earthworm_hit_itself(earthworm):
	for body_block in earthworm.body:
		if earthworm.body.count(body_block) > 1 and body_block != earthworm.body[-1]:
			return True
	return False

def earthworm_hit_wall(earthworm):
	for body_block in earthworm.body:
		x, y = body_block
		if x == 0 or x == 60 or y == 0 or y == 60 :
			return True
	return False
