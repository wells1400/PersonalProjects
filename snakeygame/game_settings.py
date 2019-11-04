class Game_Settings(object):
	"""docstring for Game_Settings"""
	def __init__(self):
		super(Game_Settings, self).__init__()
		self.screen_width  = 700
		self.screen_height = 600
		self.bg_color   = (0,0,0)
		self.wall_color = (0,255,0)
		self.earthworm_color = (255,255,255)
		self.dead_earthworm_color = (255,0,0)
		self.food_color = (255,0,0)

		self.food_exits = False
		self.food_x = 0
		self.food_y = 0

		self.game_active = True

		self.earthworm_speed_factor = 0.1
		