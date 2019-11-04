import pygame
from pygame.sprite import Group

class Earthworm(object):
	"""docstring for Earthworm"""
	def __init__(self,screen, settings):
		super(Earthworm, self).__init__()
		self.screen = screen
		self.settings = settings
		self.body = []
		self.head_x = 0.0 
		self.head_y = 0.0
		self.initial_head()
		self.direction = 0
		# 1 means up 
		# 2 means down 
		# 3 means left 
		# 4 means right 

	def initial_head(self):
		self.body.append( (10,15) )

	def update_earthworm(self):
		if self.settings.game_active :
			for earthworm_body_block in self.body:
				x,y = earthworm_body_block
				pygame.draw.rect(self.screen, self.settings.earthworm_color, (x * 10 ,y * 10,10,10))
		else:
			for earthworm_body_block in self.body:
				x,y = earthworm_body_block
				if earthworm_body_block != self.body[-1] or x != 60:
					pygame.draw.rect(self.screen, self.settings.dead_earthworm_color, (x * 10 ,y * 10,10,10))

	def move(self):
		x,y = self.body[-1]
		if self.direction   == 4:
			self.body.append( (x + 1, y ) )
		elif self.direction == 3:
			self.body.append( (x - 1, y ) )
		elif self.direction == 2:
			self.body.append( (x , y + 1) )
		elif self.direction == 1:
			self.body.append( (x , y - 1 ) )

		if self.direction != 0:
			if   (self.settings.food_x ,  self.settings.food_y) == self.body[-1] :
				self.settings.food_exits = False
			else:
				del self.body[0]
		
		