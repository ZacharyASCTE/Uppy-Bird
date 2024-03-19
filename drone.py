import numpy as np
import math
import random

active_inputs=["distanceLeftWall","velocity","distanceTop","distHole", "distanceRightWall"]
node_list_amount = [len(active_inputs),4,3]

class Drone:
    """The Bird class. Contains information about the bird and also it's brain,
        breeding behaviour and decision making"""

    def __init__(self, learning_rate_multiplier, male = None, female = None):
        """The constructor. Either a bird, which is initialized by breeding,
           a mutated bird, or a standalone bird.
        INPUT:  height: The screen height.
                        The bird will be initialized in the middle of it
                male:     Defaulted to None.
                        If set:
                          The brain (weights for NN) are taken over and mutated
                female: Defaulted to None
                         If set with male:
                        Averages the brains (weights for NN)
                          of male and female and mutate

        OUTPUT: None"""

        def NodeListGenerator(*args):
            _ = []
            for i in range(len(args[:-1])):
                _.append(np.random.normal(0, scale=0.1, size=(args[i], args[i+1])))
            return _
        
        self.node_list = []
        self.bestReported = False
        self.learning_rate_multiplier = learning_rate_multiplier
        self.radius = 0
        self.maze_line_width = 0
        self.space = 0
        self.spacing = 0
        self.loops = 0
        self.ai = False
        self.fps = 0
        self.old_fps = 0
        self.recentlyDead = False
        self.frame_start = 0
        self.time_multiplier = 0
        self.vision = 150
        self.hole_vision_multi = 0.5 # Multiplier for vision of hole
        
        self.distanceBot = 0
        self.distanceTop = 0
        self.distanceLeft = 0
        self.distanceRight = 0
        self.velocity = 0
        self.player_level = 0
        self.best_player_level = 0
        self.y = 0
        self.xPosition = 0
        self.fitness = 50
        self.jumps = 0
        self.alive = True
        self.frames = 0
        self.layers_deleted = 0
        self.old_level = 0
        self.velocity_max_up = 0
        self.velocity_max_down = 0
        self.distanceRightWall = float('inf')
        self.distanceLeftWall = float('inf')
        self.distHole = float('inf')

        self.radius = 0
        self.maze_line_width = 0
        self.space = 0
        self.spacing = 0
        self.loops = 0
        self.ai = 0
        self.y = 0
        self.window_width = 0
        self.window_height = 0
        self.xPosition = 0 # - Left + Right
        self.time_multiplier = 0
        self.layer_time_limit = 0
        self.velocity_max_up = 0
        self.velocity_max_down = 0
        self.deathModifier = 0
        self.discoveryBonus =500
        self.bonusAchieved = False

        if (male == None): #New Bird, no parents
            #easy network
            self.node_list = NodeListGenerator(*node_list_amount)
        elif (female == None): #Only one Parent (self mutate)
            self.node_list = male.node_list
            self.mutate()
        else: # Two parents - Breed.
            self.node_list = NodeListGenerator(*node_list_amount)
            self.breed(male, female)
    def refreshDict(self):
        self.possible_inputs = {"distanceRightWall": [self.distanceRightWall,self.window_width,0,],"distanceLeftWall": [self.distanceLeftWall,self.window_width,0,],"velocity": [self.velocity,self.velocity_max_up,self.velocity_max_down,],"distanceBot": [self.distanceBot,self.spacing,0,],"distHole": [self.distHole,self.window_width,-self.window_width,],"distanceTop": [self.distanceBot,self.spacing,0,]}


    def constants(self,radius,maze_line_width,space,spacing,loops,window_width,window_height,ai,time_multiplier,layer_time_limit,velocity_max_up,velocity_max_down):
        self.radius = radius
        self.maze_line_width = maze_line_width
        self.space = space
        self.spacing = spacing
        self.loops = loops
        self.ai = ai
        self.y = spacing/4
        self.window_width = window_width
        self.window_height = window_height
        self.xPosition = window_width/2 # - Left + Right
        self.time_multiplier = time_multiplier
        self.layer_time_limit = layer_time_limit
        self.velocity_max_up = velocity_max_up
        self.velocity_max_down = velocity_max_down

    def variables(self, fps, frames, layers_deleted):
        self.fps = fps
        self.frames = frames
        self.layers_deleted = layers_deleted

    def processBrain(self,tracking_list):
        """Updates what the bird sees.

        INPUT:  pipeUpperY - The y coordinate of the upper pipe
                pipeLowerY - The y coordinate of the lower pipe
                pipeDistance - The x distance to the pipe pair
        OUTPUT: None"""
        self.player_level = self.y//self.spacing

        if (self.player_level>self.best_player_level):
            self.best_player_level = self.player_level
            self.frame_start = self.frames

        self.distanceTop = self.spacing-(self.y-self.radius)%self.spacing
        self.distanceBot = self.spacing-(self.y+2*self.radius)%self.spacing-self.maze_line_width+self.radius

        if(self.y<self.radius+self.layers_deleted*self.spacing):
            self.fitness -= 1000
            self.alive = False
            self.recentlyDead = True

        else:
            self.distanceLeft = self.xPosition-tracking_list[(self.y-self.radius)//self.spacing]-self.radius
            self.distanceRight = -(self.xPosition-tracking_list[(self.y-self.radius)//self.spacing]-self.space)-self.radius
            self.distHole = abs(self.distanceLeft-self.distanceRight)/2 if abs(self.distanceLeft+self.space) <= self.vision * self.hole_vision_multi or (self.distHole != float('inf') and self.player_level == self.old_level) else float('inf') 
            #self.distanceLeftWall = self.xPosition if self.xPosition <= 50 or (self.distanceLeft != float('inf') and self.player_level == self.old_level) else float('inf')
            self.distanceLeftWall = self.xPosition if self.xPosition <= self.vision or (self.distanceLeftWall != float('inf') and self.player_level == self.old_level) else float('inf')
            self.distanceRightWall = self.window_width-self.xPosition if self.window_width-self.xPosition <= 50 or (self.distanceRightWall != float('inf') and self.player_level == self.old_level) else float('inf')
            self.distanceMasterWall = self.xPosition if self.xPosition <= self.vision or self.xPosition >= self.window_width-self.vision or (self.distanceLeftWall != float('inf') and self.player_level == self.old_level) else float('inf')
            if self.old_level == self.player_level and self.distHole!=float('inf') and not(self.bonusAchieved):
                self.bonusAchieved = True
                self.fitness += self.discoveryBonus * (sum([_!=float('inf') for _ in [self.distanceRightWall,self.distanceLeftWall,self.distHole]]))
            elif self.old_level != self.player_level: self.bonusAchieved = False

        self.old_level = self.player_level

    def handleCollision(self):
        """Checks if the bird hits the upper bounds, lower bounds or a pipe

        INPUT:  HEIGHT - The global height of the screen
                BLOCKSIZE - The global bird size
                pipe - The pipe to handle the collision with
        OUTPUT: None"""
        #Check if player collided with upper or lower pipe

        if (self.ai):
            if (self.frames>(self.layer_time_limit*self.time_multiplier+self.frame_start)):
                self.alive = False
                self.recentlyDead = True
                self.deathModifier = 200

        if (self.radius > self.xPosition) or self.radius>(self.window_width-self.xPosition):
            self.alive = False
            self.recentlyDead = True
            self.deathModifier = 600

        if ((self.distanceTop<(self.maze_line_width+2*self.radius)) and (self.distanceLeft<-self.radius or self.distanceRight<-self.radius) or ((self.radius>math.sqrt((self.distanceLeft+self.radius)**2+(self.distanceTop-self.maze_line_width-self.radius)**2) or self.radius>math.sqrt((self.distanceLeft+self.radius)**2+(self.distanceTop-self.radius)**2)) or (self.distanceTop<(self.maze_line_width+self.radius) and self.distanceTop>=self.radius and self.distanceLeft<0)) or ((self.radius>math.sqrt((self.distanceRight+self.radius)**2+(self.distanceTop-self.maze_line_width-self.radius)**2) or self.radius>math.sqrt((self.distanceRight+self.radius)**2+(self.distanceTop-self.radius)**2)) or (self.distanceTop<(self.maze_line_width+self.radius) and self.distanceTop>=self.radius and self.distanceRight<0))):
            self.alive = False
            self.recentlyDead = True
            self.deathModifier = 500
            

    def thinkIfMove(self):
        """Forward pass through neural network,
            giving the decision if the bird should jump.
        The neural network consists out of the y position of the bird,
            y distance to the bottom pipe, the y distance to the top pipe,
            the x distance to the pipe pair and the own velocity.

        INPUT:  None
        OUTPUT: boolean, which determines,
                    if the bird should jump (True) or not (False)"""
        BIAS = -0.5
        self.refreshDict()
        _ = list(map(self.normalization, [self.possible_inputs[_] for _ in active_inputs]))

        
        hidden_layer_in = np.dot(_,self.node_list[0])
        hidden_layer_out = self.sigmoid(hidden_layer_in)
        for _ in range(len(node_list_amount[1:-2])):
            hidden_layer_in = np.dot(hidden_layer_out,self.node_list[_+1])
            hidden_layer_out = self.sigmoid(hidden_layer_in)
        output_layer_in = np.dot(hidden_layer_out, self.node_list[-1])
        prediction1,prediction2,prediction3 = self.sigmoid(output_layer_in).tolist()
        if (prediction1+BIAS > 0):
            a = True
        else:
            a = False
        b = prediction2+BIAS
        c = prediction3+BIAS

        return a,b,c
    
    def debugLists(*args):
        for i in range(len(args)): print(str(i) + ": " + str(args[i]))
    
    def normalization(self, args):    #args = inputvalue,maximum,minimum
        return -1.0 if args[0] == float('inf') else (args[0]-args[1])/(args[1]-args[2])
        
    def sigmoid(self, x):
        """The sigmoid activation function for the neural net

        INPUT: x - The value to calculate
        OUTPUT: The calculated result"""

        return 1 / (1 + np.exp(-x))

    def setWeights(self, list_of_best_weights):
        """Overwrites the current weights of the birds brain (neural network).

        INPUT:  inputWeights: The weights for the neural network (input layer)
                hiddenWeights: The weights for the neural network (hidden layer)
        OUTPUT:    None"""
        self.node_list = list_of_best_weights

    def breed(self, male, female):
        """Generate a new brain (neural network) from two parent birds
             by averaging their brains and mutating them afterwards

        INPUT:  male - The male bird object (of class bird)
                female - The female bird object (of class bird)
        OUTPUT:    None"""
        for i in range(len(self.node_list)):
            for j in range(len(self.node_list[i])):
                self.node_list[i][j] = (male.node_list[i][j] + female.node_list[i][j])/2

        self.mutate()

    def mutate(self):
        """mutate (randomly apply the learning rate) the birds brain
            (neural network) randomly changing the individual weights

        INPUT:  None
        OUTPUT:    None"""
        for i in range(len(self.node_list)):
            for j in range(len(self.node_list[i])):
                for k in range(len(self.node_list[i][j])):
                    self.node_list[i][j][k] = self.getMutatedGene(self.node_list[i][j][k])

    def getMutatedGene(self, weight):
        """mutate the input by -0.125 to 0.125 or not at all

        INPUT: weight - The weight to mutate
        OUTPUT: mutatedWeight - The mutated weight"""

        multiplier = 0
        learning_rate = random.randint(0, 25) * self.learning_rate_multiplier
        randBool = bool(random.getrandbits(1)) #adapt upwards or downwards?
        randBool2 = bool(random.getrandbits(1)) #or not at all?
        if (randBool and randBool2):
            multiplier = 1
        elif (not randBool and randBool2):
            multiplier = -1

        mutatedWeight = weight + learning_rate*multiplier

        return mutatedWeight
