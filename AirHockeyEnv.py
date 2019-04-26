from recv_data import recv_puck, recv_robo, recv_dummy_data, est_TCP_cnxn
from send_data import est_serial_cnxn, send
from constants import Y_MAX, Y_MIN
import numpy as np
import time
import gym
from gym.utils import seeding

class AirHockeyEnv(gym.Env):
    metadata = {
        # Used for simulation environment
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }
    def __init__ (self):
        # Send to arduino actual Y position, not increments
        self._robot_pos = 0
        self._prev_robot_pos = 0
        self._next_robot_pos = None
        self._temp_robot_pos = None
        self._prev_x = 0
        self._prev_y = 0
        self._puck_x = 0
        self._puck_y = 0
        self._prev_speed_x = 0
        self._speed_x = 0
        self._speed_y = 0
        self.hit_flag = False
        self._new_episode_flag = False  # Flag to stop multiple episode increments'
        self._goal_scored = False
        self._intercepted = False
        self._hit = False   

        self.TCP_cnxn = est_TCP_cnxn()
        self.SER_cnxn = est_serial_cnxn()

        # Definining variables for rendering environment
        self.top_wall = None
        self.bot_wall = None
        self.goal = None
        self.puck = None
        self.stick = None
        self.viewer = None
        self.puck_trans = None
        self.stick_trans = None

        self.seed()
        #self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # Resets the environment on startup and at the beginning of episode
        new_state = []

        # Receive Puck data from Image Processing PI
        rc = recv_puck(self.TCP_cnxn)
        
        # Move physical robot to middle of playing field
        self._next_robot_pos = 150
        self._temp_robot_pos = self._next_robot_pos
        send(self.SER_cnxn, str(self._next_robot_pos))

        # Updates robot position directly from arduino keeps previous value until new value is sent
        
        self._robot_pos = recv_robo(self.SER_cnxn)
        if self._robot_pos == None:
            self._robot_pos = self._prev_robot_pos

        # Update puck variables 
        self._puck_x = rc[0]
        self._puck_y = rc[1]
        self._speed_x = 0  #self.get_speed('x') start at 0 for first state
        self._speed_y = 0  #self.get_speed('y')

        # Resetting flags used for informing robot
        # These flags should change to True only once per episode
        self._goal_scored = False
        # self._intercepted = False
        # self._hit = False

        new_state.append(self._robot_pos)
        new_state.append(rc[0])
        new_state.append(rc[1])
        new_state.append(self._speed_x)
        new_state.append(self._speed_y)
        
        return np.array(new_state)

    def verify_robot_pos(self):
        # A function to verify robot position is in bounds after updating
        if self._temp_robot_pos < Y_MIN:
            self._temp_robot_pos = Y_MIN
        elif self._temp_robot_pos > Y_MAX:
            self._temp_robot_pos = Y_MAX

    def send_data_new(self, action):
        # Trying out method of updating robot pos in background
        # Here 1st action would be sent to arduino and temp_pos&next_pos = robot_pos + action
        # Next iteration if robot_pos != next_pos: temp_pos = temp_pos + action
        # Once robot_pos == next_pos, next_pos = temp_pos and send next_pos

        # print("action is: {}".format(action))
        if 0 < action < 6:      # actions 1 - 5 = Move Up
            move = int(action*3)
        elif 5 < action < 11:   # actions 6 - 10 = Move Down
            move = -int((action-5)*3)
        elif action == 11:
            sendBuff = "999"
            send(self.SER_cnxn, str(sendBuff))
        elif action == 0:       # action = Do nothing
            pass

        if 0 < action < 11:
            if self._next_robot_pos is None:
                # Error handling
                self._next_robot_pos = self._robot_pos + move
                self._temp_robot_pos = self._next_robot_pos
            elif self._next_robot_pos is not None and self._robot_pos == self._next_robot_pos:
                # sends next position for robot to move to
                self._temp_robot_pos += move
                self._next_robot_pos = self._temp_robot_pos
            elif self._next_robot_pos is not None and self._robot_pos != self._next_robot_pos:
                self._temp_robot_pos += move # This updates robot pos in background while moving
            self.verify_robot_pos()
            
        # Constantly tell arduino to execute current command
        if self._puck_x < -150:
            send(self.SER_cnxn, "999")
        else:
            send(self.SER_cnxn, str(self._next_robot_pos))
        

        #print("Action is: {}, current is: {}, next is: {}, temp is: {}".format(action, self._robot_pos, self._next_robot_pos, self._temp_robot_pos))

    def step(self, action):
        # This will be the function used for the real world data.
        done = 0
        reward = 0
        hit = False
        updated = False
        intercept = False
        hit_puck = "999"    # String to send arduino to spin hockey stick
        rc = []
        new_state = []
        self._prev_x = int(self._puck_x)
        self._prev_y = int(self._puck_y)
        self._prev_speed_x = int(self._speed_x)
        self._prev_robot_pos = int(self._robot_pos)
        
        self.send_data_new(action)
        
        ## RECEIVING DATA ##
        count = 1
        while not updated:
            # Loop waits for new data to be received before moving on
            # print("count = {}".format(count))
            rc = recv_puck(self.TCP_cnxn)
            if rc[0] != self._prev_x or rc[1] != self._prev_y:
                updated = True
            else:
                count += 1
        # Receive robot position directly from arduino
        self._robot_pos = recv_robo(self.SER_cnxn)
        if self._robot_pos == None:
            self._robot_pos = self._prev_robot_pos
        #print("Robot position in AHE.py is: {}".format(self._robot_pos))
        ## END RECEIVING DATA ##

        # Update puck values if goal not scored
        if rc[0] != -999 and rc[1] != -999:
            self._puck_x = int(rc[0])
            self._puck_y = int(rc[1])
            self._speed_x = self.get_speed('x')
            self._speed_y = self.get_speed('y')
        else:
            # If -999's returned, no data received for one second
            print("Timeout")
            self._goal_scored = True

        # calculating x distance and y distance
        x_dist = self._puck_x + 300             
        y_dist = self._puck_y - self._robot_pos
        print("x_speed: {} x_dist: {} y_dist: {} Robot_pos: {}".format(self._speed_x, x_dist, y_dist, self._robot_pos))

        #print("y_dist = {}, x_dist = {}".format(y_dist, x_dist))
        # Could be used as an equation where the y_dist is more important the lower x_dist is
        # ry_dist and rx_dist are variables used to calculate the reward for real world robot
        ry_dist = abs(int(y_dist/3))
        if ry_dist > 100:
            ry_dist = 100
            
        rx_dist = int(x_dist/6)
        if rx_dist > 100:
            rx_dist = 100
        elif rx_dist < 0:
            rx_dist = 0

        #print("ry_dist is: {}, rx_dist is: {}".format(ry_dist, rx_dist)) 
        reward = (100 - ry_dist) * (0.01 * (100 - rx_dist))
        #print("reward is: {}".format(reward))
        
        if ((100 - y_dist) * (0.01 * (100 - x_dist))) > 50:  
            reward += 10
        # print("x_dist: {}, y_dist: {}, x_speed: {}".format(x_dist, y_dist, self._speed_x))
        if x_dist < 20 and x_dist >= -20 and self._speed_x < 0:
            # Reward for intercepting puck
            if (-32 < y_dist <= 5) and self._intercepted == False:
                print("PUCK INTERCEPTED!!!!!")
                reward += 100 
                self._intercepted = True
                done = True
                self._new_episode_flag = True
            if (-25 < y_dist <= 0) and action == 11 and self._hit == False:
                # Check for successful hit
                print("PUCK HIT!!!!!!!!!")
                reward += 1000
                self._hit = True
                hit = True
                done = True
                self._new_episode_flag = True
        elif x_dist > 20 and action == 11:
            # Subtract from reward if hit action is chosen while puck is not close
            reward -= 500
            pass
        
        
        # print("x_dist: {} x_speed: {} y_dist: {} y_speed: {}".format(x_dist, self._speed_x, y_dist, self._speed_y))
        # Complete Episode when puck crosses robot axis
        if x_dist < 30 and self._new_episode_flag == False:
            self._new_episode_flag = True
            done = True
        elif x_dist < 0 and self._goal_scored == True:
            self._new_episode_flag = True
            done = True
        elif x_dist > 60:
            # Resetting variables for next episode only after puck crosses threshold
            self._new_episode_flag = False
            self._intercepted = False
            self._hit = False

        new_state.append(self._robot_pos)
        new_state.append(self._puck_x)
        new_state.append(self._puck_y)
        new_state.append(self._speed_x)
        new_state.append(self._speed_y)

        return np.array(new_state), reward, self._intercepted, hit, done

    def in_play(self):
        # Return true if puck is in play
        return self._puck_x != -999 and self._puck_y != -999

    def get_speed(self, axis):
        # returns the speed based on a previous position (v1) and a current position (v2)
        if axis == 'x':
            v1 = self._prev_x
            v2 = self._puck_x
        elif axis == 'y':
            v1 = self._prev_y
            v2 = self._puck_y
        if self.in_play():
            return int((v2-v1)/4)
        elif not self.in_play():
            return 0

    def render(self, mode='human'):
        # Used for simulation environment
        screen_width = 800
        screen_height = 400
        world_width = 200
        world_height = 100
        scale = 4
        clearance = 10

        pr = 15  # puck radius

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            self.top_wall = rendering.Line(start=(0, 395), end=(200*scale, 395))
            self.bot_wall = rendering.Line(start=(0, 0), end=(200*scale, 0))

            self.goal = rendering.Line(start=(1.0, 40.0*scale), end=(1.0, 60.0*scale))
            self.stick = rendering.Line(start=(8, (self._robot_pos*scale) - 20), end=(8, (self._robot_pos*scale) + 20))
            self.puck = rendering.make_circle(radius=pr, res=30, filled=True)
            self.puck.add_attr(rendering.Transform(translation=(500, 200)))
            self.puck.set_color(0, 0, 255)

            self.puck_trans = rendering.Transform()
            self.stick_trans = rendering.Transform()

            self.puck.add_attr(self.puck_trans)
            self.stick.add_attr(self.stick_trans)
            self.viewer.add_geom(self.goal)
            self.viewer.add_geom(self.stick)
            self.viewer.add_geom(self.puck)

            self.viewer.add_geom(self.top_wall)
            self.viewer.add_geom(self.bot_wall)

        x_pos = (self._puck_x*scale) - 100
        self.puck.set_color(0, 0, 255)
        self.puck_trans.set_translation(x_pos, self._puck_y*scale - 200)
        self.stick_trans.set_translation(8, self._robot_pos*scale - 200)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def quit_render(self):
        self.viewer.close()

    '''
    def step_dummy(self, action):

        done = False
        hit = False
        int = False
        reward = 0
        x_dist = 0
        y_dist = 0
        new_state = []
        self._puck_x += self._speed_x
        self._puck_y += self._speed_y

        if 0 < action < 6:      # actions 1 - 5 = Move Up
            intensity = action
            self._robot_pos += intensity
        elif 5 < action < 11:   # actions 6 - 10 = Move Down
            intensity = action - 5
            self._robot_pos -= intensity
        elif action == 11:
            self.hit_up = True
        elif action == 12:
            self.hit_down = True
        elif action == 0:       # action = Do nothing
            pass

        # stop robot at walls
        if self._robot_pos < Y_MIN:
            self._robot_pos = Y_MIN
        elif self._robot_pos > Y_MAX:
            self._robot_pos = Y_MAX

        # Stop puck
        if self._puck_x < X_MIN:
            self._puck_x = X_MIN

        # Turn puck around if it hits a wall
        if self._puck_y < Y_MIN:
            self._speed_y = -self._speed_y
            self._puck_y = Y_MIN + (Y_MIN - self._puck_y)   # Keep consistency with distance traveled
        elif self._puck_y > Y_MAX:
            self._speed_y = -self._speed_y
            self._puck_y = Y_MAX - (self._puck_y - Y_MAX)   # Keep consistency with distance traveled

        # calculate reward
        x_dist = self._puck_x + 100  # checks distance between puck and dummy y axis (x = -97.5 (10/4))
        if x_dist > 100:
            x_dist = 100  # caps x_dist at 100
        y_dist = abs(self._puck_y - self._robot_pos)

        # The reward equation could be used if we implemented Joel's algorithm as an input
        if ((100-y_dist)*(0.01*(100-x_dist))) > 50:  # Calc reward based on dist, if puck is closer, y_dist more imp
            reward += 10

        if x_dist == 0:
            # Reward for intercepting puck
            if y_dist < 5:
                int = True
                reward += 100 - (y_dist*5)
            done = True
            if y_dist < 5:
                # Check for successful hit
                if self._puck_y >= self._robot_pos and action == 11:
                    # Successful hit up action
                    reward += 1000
                    hit = True
                    if self.puck is not None:
                        self.puck.set_color(255, 0, 0)
                        time.sleep(1)
                elif self._puck_y <= self._robot_pos and action == 12:
                    # Successful hit down action
                    reward += 1000
                    hit = True
                    if self.puck is not None:
                        self.puck.set_color(255, 0, 0)
                        time.sleep(1)
        else:
            if action == 11 or action == 12:
                reward -= 100  # Teach robot not to hit unless it is time to

        new_state.append(self._robot_pos)
        new_state.append(self._puck_x)
        new_state.append(self._puck_y)
        new_state.append(self._speed_x)
        new_state.append(self._speed_y)

        return np.array(new_state), reward, int, hit, done
    '''
