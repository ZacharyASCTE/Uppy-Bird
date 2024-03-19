import pygame
import random
import copy
import time
import drone
import numpy as np
import math

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True

DROCK = 200 #No of DROCKs to spawn

AI = True # Set to false, if you want to play yourself
GRAVITY_TOGGLE = True
birdView = True # Set to false, if you don't want to see what the birds see

SLOW_FPS = 1/10
REGULAR_FPS = 30
FAST_FPS = 10

fps = REGULAR_FPS

FILE_NAME = "Best_Drone_Birds.txt"

SLOW_LEARN = 1/2
REGULAR_LEARN = 0.005
FAST_LEARN = 2

learning_rate = REGULAR_LEARN

jump_switch = False
time_switch = False
kill_all_switch = False
graphics_switch = False
fps_switch = False
learn_switch = False
restart_switch = False
global_fitness_switch = False
current_player_switch = False
best_save_switch = False
current_save_switch = False

kill_all = False
graphics = True
load_bird = False
free_cam = 0

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800

time_limit_enabled = True
TIME_POINTS = 200
TIME_LIMIT = 10
TIME_MULTIPLIER = 35 #DO NOT CHANGE
LAYER_TIME_LIMIT = 7
MAX_FRAMES = TIME_MULTIPLIER*TIME_LIMIT

JUMP_MULTIPLIER = -20
ALIGNMENT_MULTIPLIER = 1
FLOOR_POINTS_MULTIPLIER = 800

GRAVITY = .375
SPACING = 200
SPACE = 100
MAZE_LINE_WIDTH = 20
PLAYER_TERMINAL_VELOCITY = -10
RADIUS = 15
JUMP_VELOCITY = 7
LOOPS = WINDOW_HEIGHT//SPACING

#carl_image = pygame.image.load('Carl.png').convert_alpha()
#carl_image = pygame.transform.scale(carl_image,(2*RADIUS,2*RADIUS))

globalFitness = -2000
best_globalFitness = -2000
best_player_level = 0
best_generation = 0
best_node_list = 0
current_player_level = 0
current_node_list = 0
best_drones_file = 0
list_of_best_drones = 0
dictionary_of_best_drones = {}
visual_dictionary_of_best_drones = {}
formatted_array = []


frames = 0
minimum_player_level = -1
layers_deleted = 0
highest_player_height = 0
highest_player_width = 0
screen_height = WINDOW_HEIGHT/2
screen_width = 0
player_position = WINDOW_WIDTH/2
player_velocity = 0

player_upper_pole_height_difference = 0
player_lower_pole_height_difference = 0
player_left_pole_distance_difference = 0
player_right_pole_distance_difference = 0

#GlobalVariable Setup

def superReplace(s,r,w=[],d=None): # string, replace, with, default
  if d or d == False or d=="":
    w = [w[_] if len(w)-1>=_ else str(d) for _ in range(len(r))]
  for _ in range(len(r)):
    s = s.replace(str(r[_]),str(w[_]))
  return s

def ListShapeGenerator(*args):
            _ = []
            for i in range(len(args[:-1])):
                _.append(0)
            return _

bestListofWeights = ListShapeGenerator(drone.node_list_amount)

player = None
multiPlayer = []
score = 0
maxscore = 0
highgen = 0
running = True
generation = 1
birdsToBreed = []
number_of_birds_to_keep = 5
singlePlayer = None
respawn = False
highscore = 0
weight = 1.2                      
magic_breeding_numbers = [(10**(math.log10(weight)*(2+_)))-(10**((_+1)*math.log10(weight)))-weight+1 for _ in range(number_of_birds_to_keep)]
magic_number = magic_breeding_numbers[-1]

maze_level = 0

tracking_list_current_number = 0
tracking_list = {}

window = pygame.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT))
window.fill((255,255,255))

def init():
    """This method is called whenever the game is started.
        This may be one of these cases
        1) First start (no matter if user or AI plays)
        2) If user plays: He died and clicked to restart
        3) If AI plays: The whole generation went extinct. This is a restart
        The method initializes Pipes, the singleplayer bird, AI Birds

    INPUT: None
    OUTPUT: None"""

    global player, running, score, multiPlayer, singlePlayer, respawn, globalFitness, screen_height, kill_all, frames,tracking_list, tracking_list_current_number, maze_level, minimum_player_level, layers_deleted, load_bird, generation

    #Reset some global variables
    score = 0
    minimum_player_level = -1
    layers_deleted = 0
    kill_all = False
    globalFitness = -2000
    running = True
    frames = 0
    maze_level = 0
    tracking_list_current_number = 0
    tracking_list = {}
    for i in range(LOOPS+2):
        tracking_list[i] = random.randint(0,WINDOW_WIDTH-SPACE+1)
        tracking_list_current_number += 1

    if (not AI): #User plays: Initialize exactly one bird.
        multiPlayer = []
        singlePlayer = drone.Drone(learning_rate)
        multiPlayer.append(singlePlayer)
    else:
        if load_bird:
            generation = 1
        print("New Gen " + str(generation))
        singlePlayer = drone.Drone(learning_rate)
        if (len(birdsToBreed) == 0):
            #This is the first init. New ones, are covered in the else below.
            print("Kaboom?")
            for _ in range(DROCK):
                #First time initialization of birds.
                multiPlayer.append(drone.Drone(learning_rate))
        else:
            #Atleast one death happened
            multiPlayer = []

            #keep the best bird of generation without mutation
            _ = drone.Drone(learning_rate)
            _.setWeights(birdsToBreed[0].node_list)
            if load_bird == True:
                _.setWeights(formatted_array)
            multiPlayer.append(_)

            #also keep the best of all time alive without mutation
            _ = drone.Drone(learning_rate)
            _.setWeights(bestListofWeights)
            if load_bird == True:
                _.setWeights(formatted_array)
                load_bird = False
            multiPlayer.append(_)

            for _ in range(int(DROCK/3)):
                #Breed and mutate the two generations best birds sometimes
                multiPlayer.append(drone.Drone(learning_rate,*magic_breeding_function()))
            for _ in range(int(DROCK/3)):
                #Breed and mutate the generations best bird a couple of times
                multiPlayer.append(drone.Drone(learning_rate,birdsToBreed[random.randint(0,1)]))

            for _ in range(int(DROCK/3)-2):
                if (respawn): #Bad genes - replace some.
                    multiPlayer.append(drone.Drone(learning_rate))
                else:
                    #Breed and mutate the generations second best bird asometimes
                    multiPlayer.append(drone.Drone(learning_rate,magic_breeding_function(True)))

            if (respawn):
                respawn = False
    for player in multiPlayer:
        player.constants(RADIUS,MAZE_LINE_WIDTH,SPACE,SPACING,LOOPS,WINDOW_WIDTH,WINDOW_HEIGHT,AI,TIME_MULTIPLIER, LAYER_TIME_LIMIT)

def magic_breeding_function(one=None):
    index_a = None
    index_b = None
    random_number = random.uniform(0,magic_number)
    for i in range(len(magic_breeding_numbers)): # best = -1 second best = -2...
        if random_number <= magic_breeding_numbers[i]:
            index_a = i
            break
    if one:
        return birdsToBreed[-(index_a+1)]
    while not(index_b!=None and index_a!=index_b):
        random_number = random.uniform(0,magic_number)
        for i in range(len(magic_breeding_numbers)): # best = -1 second best = -2...
            if random_number <= magic_breeding_numbers[i]:
                index_b = i
                break
    return birdsToBreed[-(index_a+1)], birdsToBreed[-(index_b+1)]

def reset_game():
    tracking = {}
    for i in range(LOOPS+2):
        tracking[str(i)] = random.randint(0,WINDOW_WIDTH-SPACE+1)
    return tracking
    

def generate_line(height,open_start,open_end,line_width,maze_width):
    pygame.draw.rect(window, (0,0,0), [maze_width,height,open_start,line_width])       #LR, UD, W, H
    pygame.draw.rect(window, (0,0,0), [open_end+maze_width,height,WINDOW_WIDTH-open_end,line_width])

def generate_maze(maze_height, maze_width, minimum_player_level):
    global maze_level
    global layers_deleted
    global tracking_list_current_number
    global tracking_list

    window.fill((255,255,255))

    for player in multiPlayer:
        if player.alive:
            pygame.draw.circle(window,(0,0,0),(player.xPosition+maze_width,WINDOW_HEIGHT/2 + screen_height - player.y),RADIUS)
            #carl_rectangle = carl_image.get_rect(center=(player.xPosition+maze_width,WINDOW_HEIGHT/2 + screen_height - player.y))
            #window.blit(carl_image,carl_rectangle)
            
    for x in range(minimum_player_level-2-layers_deleted):
        del tracking_list[x+layers_deleted]
    layers_deleted = max(minimum_player_level-2,0)
    
    bottom_level = int((maze_height-WINDOW_HEIGHT/2)//SPACING-2)

    for x in range(int(WINDOW_HEIGHT//SPACING+4)):
        if (x+bottom_level) in tracking_list:
            top_of_rectangle = (bottom_level+8-x-1)*SPACING + maze_height%SPACING - (maze_height//SPACING)*SPACING + WINDOW_HEIGHT/2
            start = tracking_list[x+bottom_level]
            end = start + SPACE
            generate_line(top_of_rectangle,start,end,MAZE_LINE_WIDTH,maze_width)
        elif (x+bottom_level+1) in tracking_list:
            pygame.draw.rect(window, (0,0,0), [maze_width,((len(tracking_list))*SPACING + maze_height - maze_level*SPACING - WINDOW_HEIGHT),WINDOW_WIDTH,MAZE_LINE_WIDTH])
    pygame.draw.rect(window, (0,0,0), [maze_width-MAZE_LINE_WIDTH,0,MAZE_LINE_WIDTH,(bottom_level+4-layers_deleted)*SPACING + maze_height%SPACING + WINDOW_HEIGHT/2 + MAZE_LINE_WIDTH])
    pygame.draw.rect(window, (0,0,0), [WINDOW_WIDTH+maze_width,0,MAZE_LINE_WIDTH,(bottom_level+4-layers_deleted)*SPACING + maze_height%SPACING + WINDOW_HEIGHT/2 + MAZE_LINE_WIDTH])
    

init()

while True:
    currentfitness = 0.0

    for event in pygame.event.get():
        if (event.type == 5) and (not running) and (not singlePlayer.alive):
            init()

    # if not pygame.key.get_pressed()[pygame.K_r] and not reset_switch:
    #        reset_switch = True
    # 
    # if pygame.key.get_pressed()[pygame.K_r] and reset_switch:
    #        reset_switch = False
    #        birdsToBreed = []
    #        generation = 0
    #        init()

    if not pygame.key.get_pressed()[pygame.K_1] and not pygame.key.get_pressed()[pygame.K_2] and not pygame.key.get_pressed()[pygame.K_3] and not fps_switch:
            fps_switch = True

    if pygame.key.get_pressed()[pygame.K_1] and fps_switch:
            fps_switch = False
            fps = fps*SLOW_FPS
            print(fps)

    if pygame.key.get_pressed()[pygame.K_2] and fps_switch:
            fps_switch = False
            fps = REGULAR_FPS
            print(fps)
    
    if pygame.key.get_pressed()[pygame.K_3] and fps_switch:
            fps_switch = False
            fps = fps*FAST_FPS
            print(fps)

    if not pygame.key.get_pressed()[pygame.K_6] and not pygame.key.get_pressed()[pygame.K_7] and not pygame.key.get_pressed()[pygame.K_8] and not learn_switch:
            learn_switch = True

    if pygame.key.get_pressed()[pygame.K_6] and learn_switch:
            learn_switch = False
            learning_rate = learning_rate*SLOW_LEARN
            print(learning_rate)

    if pygame.key.get_pressed()[pygame.K_7] and learn_switch:
            learn_switch = False
            learning_rate = REGULAR_LEARN
            print(learning_rate)
    
    if pygame.key.get_pressed()[pygame.K_8] and learn_switch:
            learn_switch = False
            learning_rate = learning_rate*FAST_LEARN
            print(learning_rate)
    
    if pygame.key.get_pressed()[pygame.K_f] and free_cam:
        free_cam = 0
    
    if pygame.key.get_pressed()[pygame.K_g] and not free_cam == 1:
        free_cam = 1
    
    if pygame.key.get_pressed()[pygame.K_UP]:
        free_cam = 2
        screen_height += 5
        time.sleep(.02)

    if pygame.key.get_pressed()[pygame.K_DOWN]:
        free_cam = 2
        screen_height -= 5
        time.sleep(.02)

    if pygame.key.get_pressed()[pygame.K_LEFT]:
        free_cam = 2
        screen_width +=5
        time.sleep(.02)

    if pygame.key.get_pressed()[pygame.K_RIGHT]:
        free_cam = 2
        screen_width -=5
        time.sleep(.02)

    if(not AI):
        if not pygame.key.get_pressed()[pygame.K_p] and not restart_switch:
            restart_switch = True

        if pygame.key.get_pressed()[pygame.K_p] and restart_switch:
            restart_switch = False
            tracking_list = reset_game()
            player_position = WINDOW_WIDTH/2
            init()
            pygame.display.flip()
        
        if pygame.key.get_pressed()[pygame.K_d]:
            player.xPosition += 5
            time.sleep(.02)
        
        if pygame.key.get_pressed()[pygame.K_a]:
            player.xPosition -= 5
            time.sleep(.02)
        
        if not pygame.key.get_pressed()[pygame.K_SPACE] and not jump_switch:
            jump_switch = True

        if pygame.key.get_pressed()[pygame.K_SPACE] and jump_switch:
            player.velocity = 5
            jump_switch = False
        
        if not GRAVITY_TOGGLE:
            if pygame.key.get_pressed()[pygame.K_w]:
                player.y += 5
                time.sleep(.02)

            if pygame.key.get_pressed()[pygame.K_s]:
                player.y -= 5
                time.sleep(.02)

    if (player_position > WINDOW_WIDTH):
        player_position = WINDOW_WIDTH

    if (player_position < 0):
        player_position = 0

    if (running):
        noAlive = 0

        if not pygame.key.get_pressed()[pygame.K_z] and not kill_all_switch:
            kill_all_switch = True
    
        if pygame.key.get_pressed()[pygame.K_z] and kill_all_switch:
            kill_all_switch = False
            kill_all = True

        if not pygame.key.get_pressed()[pygame.K_h] and not graphics_switch:
            graphics_switch = True
    
        if pygame.key.get_pressed()[pygame.K_h] and graphics_switch:
            graphics_switch = False
            graphics = not graphics

        if not pygame.key.get_pressed()[pygame.K_q] and not current_player_switch:
            current_player_switch = True

        if pygame.key.get_pressed()[pygame.K_q] and current_player_switch:
            current_player_switch = False
            print("This generation is " + str(generation))
            print("They have completed " + str(current_player_level) + " levels so far")
            print("Their genes are " + str(current_node_list))
        
        if not pygame.key.get_pressed()[pygame.K_b] and not global_fitness_switch:
            global_fitness_switch = True
    
        if pygame.key.get_pressed()[pygame.K_b] and global_fitness_switch:
            global_fitness_switch = False
            print("Best global fitness is " + str(best_globalFitness))
            print("This was achieved on generation " + str(best_generation))
            print("They completed " + str(best_player_level) + " levels")
            print("Their genes are " + str(best_node_list))

        if not pygame.key.get_pressed()[pygame.K_j] and not current_save_switch:
            current_save_switch = True

        if pygame.key.get_pressed()[pygame.K_j] and current_save_switch:
            current_save_switch = False
            f = open(FILE_NAME, "r")
            best_drones_file = f.read()
            f.close()
            list_of_best_drones = best_drones_file.split(";")
            print(list_of_best_drones[-5])
            list_of_best_drones[-5] = list_of_best_drones[-5].split("\n")
            print(list_of_best_drones[-5])
            next_file_number = int(list_of_best_drones[-5][1]) + 1
            f = open(FILE_NAME, "a")
            f.write("\n{};{};Current;{};{}".format(next_file_number,superReplace(str(drone.node_list_amount),["\n"," "],d=""),int(current_player_level),superReplace(str(current_node_list),["\n"," "],d="")))
            f.close()

        if not pygame.key.get_pressed()[pygame.K_k] and not best_save_switch:
            best_save_switch = True

        if pygame.key.get_pressed()[pygame.K_k] and best_save_switch:
            best_save_switch = False
            f = open(FILE_NAME, "r")
            best_drones_file = f.read()
            f.close()
            list_of_best_drones = best_drones_file.split(";")
            list_of_best_drones[-5] = list_of_best_drones[-5].split("\n")
            next_file_number = int(list_of_best_drones[-5][1]) + 1
            f = open(FILE_NAME, "a")
            f.write("\n{};{};Best;{};{}".format(next_file_number,superReplace(str(drone.node_list_amount),["\n"," "],d=""),int(best_player_level),superReplace(str(best_node_list),["\n"," "],d="")))
            f.close()
        
        if pygame.key.get_pressed()[pygame.K_n]:
            f = open(FILE_NAME, "r")
            best_drones_file = f.read()
            f.close()
            list_of_best_drones = best_drones_file.split("\n")
            for x in range(len(list_of_best_drones)):
                list_of_best_drones[x] = list_of_best_drones[x].split(";")
                dictionary_of_best_drones[list_of_best_drones[x].pop(0)] = list_of_best_drones[x]
                visual_dictionary_of_best_drones[x] = [["Neural net size is {}","Their {} levels completed is"," {} levels","Their genes are {}"][_].format(dictionary_of_best_drones[str(x+1)][_]) for _ in range(len(dictionary_of_best_drones[str(x+1)]))]
                visual_dictionary_of_best_drones[x][1] = visual_dictionary_of_best_drones[x][1]+ visual_dictionary_of_best_drones[x].pop(2)
                print(type(visual_dictionary_of_best_drones))
                pick_dictionary_key = True
            while pick_dictionary_key:
                dictionary_key = input(superReplace(str(visual_dictionary_of_best_drones),["', '","{","['","']",", ","}"],["\n","","","\n","\n",""])+ '\n\nWhich drone do you want?\n')
                if dictionary_key in dictionary_of_best_drones:
                    if dictionary_of_best_drones[dictionary_key][0] == str(drone.node_list_amount).replace(" ",""):
                        unformatted_drone_array = dictionary_of_best_drones[dictionary_key][3]
                        pick_dictionary_key = False
                    else:
                        print("Array is of wrong size")
                else:
                    print("Key not in dictionary")
            formatted_array = [np.array([[float(i) for i in superReplace(j,["[","]"],d="").split(",")] for j in k.split("],[")]) for k in superReplace(unformatted_drone_array,["array(",")]","[[["],["","","[["]).split("),")]
            load_bird = True
            kill_all = True

        frames += 1
        highest_player_height = 0
        minimum_player_level_temp = -1
        highest_player_width = 0
        current_player_level = 0
        current_node_list = 0
        for player in multiPlayer:
            player.variables(fps, frames, layers_deleted)

            if kill_all:
                player.alive = False
                player.recentlyDead = True

            if not pygame.key.get_pressed()[pygame.K_t] and not time_switch:
                time_switch = True
    
            if pygame.key.get_pressed()[pygame.K_t] and time_switch:
                time_switch = False
                time_limit_enabled = not time_limit_enabled
                if time_limit_enabled:
                    print("Time limit enabled")
                else:
                    print("Time limit disabled")

            if (AI and frames>=MAX_FRAMES and time_limit_enabled):
                player.alive = False
                player.recentlyDead = True

            if (player.alive):
                if(GRAVITY_TOGGLE):
                    player.velocity -= GRAVITY
                #Did the bird hit anything?
                player.processBrain(tracking_list)
                player.handleCollision()
                if (player.alive):
                    player.y += player.velocity
                    noAlive += 1                    

                    #Jump or not?

                    if (AI):
                        movement = player.thinkIfMove()
                        if (movement[0]):
                            player.jumps += 1
                            player.velocity = JUMP_VELOCITY
                        if (movement[1] > movement[2]):
                            if (movement[1] > 0):
                                player.xPosition -= 5
                        else:
                            if (movement[2] > 0):
                                player.xPosition += 5
                
                if (player.y//SPACING<minimum_player_level_temp or minimum_player_level_temp == -1):
                    minimum_player_level_temp = int(player.y//SPACING)

                if player.y > highest_player_height:
                    highest_player_height = player.y
                    highest_player_width = WINDOW_WIDTH/2 - player.xPosition
                    current_node_list = player.node_list
                    current_player_level = int(player.player_level)

            if(player.recentlyDead):
                player.recentlyDead = False
                player.fitness += player.y
                total_floor_points = FLOOR_POINTS_MULTIPLIER*(player.y//SPACING)
                player.fitness += total_floor_points
                alignment_points = ALIGNMENT_MULTIPLIER*(-abs((player.distanceLeft-player.distanceRight)/2)+max(abs((player.distanceLeft-player.distanceRight)/2),(800-abs(player.distanceLeft-player.distanceRight)/2)))
                player.fitness += alignment_points
                #jump_points = player.jumps*JUMP_MULTIPLIER
                #player.fitness += jump_points
                if time_limit_enabled:
                    player.fitness += TIME_POINTS*frames/MAX_FRAMES
                if (player.fitness>best_globalFitness):
                    #print([total_floor_points,alignment_points,jump_points])
                    best_globalFitness = player.fitness
                    best_generation = generation
                    best_player_level = player.player_level
                    best_node_list = player.node_list
            currentfitness = player.fitness
            
            if (not player.alive):
                if (player.fitness>globalFitness):
                    globalFitness = player.fitness
        
        if free_cam == 0:
            screen_height = max(highest_player_height, WINDOW_HEIGHT/2)
            screen_width = 0
        elif free_cam == 1:
            screen_height = highest_player_height
            screen_width = highest_player_width

        if (highest_player_height > SPACING*maze_level + WINDOW_HEIGHT/2):
            maze_level+=1
            tracking_list[tracking_list_current_number] = random.randint(0,WINDOW_WIDTH-SPACE+1)
            tracking_list_current_number += 1

        minimum_player_level = max(minimum_player_level,minimum_player_level_temp)

        if (noAlive == 0):
            running = False
        else:
            if graphics:
                generate_maze(screen_height,screen_width,minimum_player_level)
    else: #Player is dead (only seen, if user plays)
        if (not AI): #User played - highscore set?
            if (score > maxscore):
                maxscore = score
                highgen = generation

        if (AI): # Let' start breeding the corpses.
            print(globalFitness)
            if ( (score > 0) or (maxscore > 0) or (globalFitness > 0.2) ):
                #Only if atleast one bird made it through one pipe
                birdsToBreed = []
                for h in range(number_of_birds_to_keep): #Best two birds are taken
                    bestBird = -1
                    bestFitness = -2000
                    for i in range(len(multiPlayer)): #Find the best bird
                        player = multiPlayer[i]
                        if (player.fitness > bestFitness):
                            bestFitness = player.fitness
                            bestBird = i
                            if (bestFitness >= highscore):
                                highscore = bestFitness
                    if ( (h == 1) and (bestFitness >= highscore) ):
                        #new highscore! Let's keep the bird and update our scores
                        allTimeBestBird = multiPlayer[bestBird]
                        bestListofWeights = copy.deepcopy(multiPlayer[bestBird].node_list)
                        #print("highscore beaten {}\n{} - Generation {}"
                        #        .format(player.inputWeights,
                        #                player.hidden1Weights,
                        #                player.hidden2Weights,
                        #                generation))
                        highscore = bestFitness
                        highgen = generation
                        maxscore = score

                    #store the (two) best birds in the breeding list
                    birdsToBreed.append(copy.deepcopy(multiPlayer[bestBird]))
                    multiPlayer.pop(i)
                #print("Best genes of this generation: {}\n{}"
                #        .format(birdsToBreed[0].inputWeights,
                #                birdsToBreed[0].hidden1Weights,
                #                birdsToBreed[0].hidden2Weights))

            #If no progress was made in the last 50 generations - new genes.
            if (generation-highgen > 50000):
                respawn = True

            #print(score)
            #print(maxscore)
            #print(globalFitness)
            generation += 1
            init()

    # flip() the display to put your work on screen
    pygame.display.flip()
    clock.tick(fps)  # limits FPS to 60

pygame.quit()




# Height centered on player
# Gain points when you go up
# Lose points for time
# Player controls maze
