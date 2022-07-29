__author__ = "Ahmad Nagib"
__version__ = "1.0.0"

import numpy as np
import math
import gym
from gym import spaces

class SlicingEnvironment(gym.Env):
    """A RAN slicing environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}
    def __init__(self, traffic_df, max_num_packets, max_size_per_tti, num_action_lvls, num_slices,
                 max_num_steps, sl_win_size, time_quantum, total_data_episodes, num_users_poisson,
                 max_traffic_percentage, max_trans_per_tti, w_volte, w_urllc, w_video, c1_volte,
                 c1_urllc, c1_video, c2_volte, c2_urllc, c2_video, num_traffic_var, reward_function_type):
        super(SlicingEnvironment, self).__init__()
        self.traffic_df = traffic_df
        self.max_num_steps = max_num_steps
        self.max_size_per_tti = max_size_per_tti
        self.sl_win_size = sl_win_size
        self.finish_time_volte = 0
        self.finish_time_urllc = 0
        self.finish_time_video = 0
        self.time_quantum = time_quantum
        self.remaining_sizes_volte = 0
        self.remaining_times_volte = 0 
        self.avg_waiting_time_volte = 0 
        self.avg_turnaround_time_volte = 0 
        self.total_avg_waiting_time_volte = 0
        self.total_avg_waiting_time_video = 0
        self.total_avg_waiting_time_urllc = 0
        self.throughput_volte = 0 
        self.total_p_no_volte = 0
        self.ep_num = 0
        self.ep_num_actual = 0
        self.total_p_no_volte = 0
        self.total_p_no_urllc = 0
        self.total_p_no_video = 0
        self.video_demand = 0
        self.volte_demand = 0
        self.urllc_demand = 0
        self.total_data_episodes = total_data_episodes
        self.num_users_poisson = num_users_poisson
        self.max_trans_per_tti = max_trans_per_tti
        self.user_priority = 1
        self.total_avg_waiting_times = {'volte' : [], 'urllc' : [], 'video' : []} 
        self.finished_throughputs = {'volte' : [], 'urllc' : [], 'video' : []}       
        self.total_throughputs = {'volte' : [], 'urllc' : [], 'video' : []}
        self.remaining_sizes = {'volte' : [], 'urllc' : [], 'video' : []} 
        self.remaining_sizes_sum = {'volte' : [], 'urllc' : [], 'video' : []} 
        self.total_p_numbers = {'volte' : [], 'urllc' : [], 'video' : []} 
        self.done_p_numbers = {'volte' : [], 'urllc' : [], 'video' : []} 
        self.remaining_times = {'volte' : [], 'urllc' : [], 'video' : []} 
        self.remaining_times_sum = {'volte' : [], 'urllc' : [], 'video' : []} 
        self.step_rewards = []
        self.states = []
        self.actions = []
        self.w_volte = w_volte
        self.w_urllc = w_urllc
        self.w_video = w_video
        self.c1_volte = c1_volte
        self.c1_urllc = c1_urllc
        self.c1_video = c1_video
        self.c2_volte = c2_volte
        self.c2_urllc = c2_urllc
        self.c2_video = c2_video
        self.current_step = 0
        self.num_traffic_var = num_traffic_var
        self.current_traffic_var_step = 0
        self.reward_function_type = reward_function_type
        
    
        # Actions 
        self.action_space = spaces.Discrete(num_action_lvls)
    
    
        # Observations
        self.observation_space = spaces.Box(
            low=0, high=max_traffic_percentage, shape=(num_slices,1), dtype=np.float)
    
    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        

        self.ep_num += 1
        self.ep_num_actual += 1
        return self._next_observation()
    
    def _next_observation(self):
        total_demand = self.volte_demand + self.urllc_demand + self.video_demand

        print('self.volte_demand is: ', self.volte_demand)
        print('self.urllc_demand is: ', self.urllc_demand)
        print('self.video_demand is: ', self.video_demand)
        print('total_demand is: ', total_demand)
        if total_demand == 0:
            volte_relative_demand = 0
            urllc_relative_demand = 0
            video_relative_demand = 0
        else:
            volte_relative_demand = self.volte_demand / total_demand
            urllc_relative_demand = self.urllc_demand / total_demand
            video_relative_demand = self.video_demand / total_demand



        self.obs = [[round(volte_relative_demand,3)], [round(urllc_relative_demand,3)], 
                        [round(video_relative_demand,3)]]


        self.states.append(self.obs)

        return self.obs

    def step(self, action):
        # Execute one time step within the environment        
        
        
        self.finish_time_volte += self.sl_win_size
        self.finish_time_urllc += self.sl_win_size
        self.finish_time_video += self.sl_win_size
        self._take_action(action)
        self.current_step += 1
        self.current_traffic_var_step += 1      



# Delay sigmoid reward function
        avg_non_waiting_time_volte = (self.sl_win_size - self.total_avg_waiting_time_volte) / self.sl_win_size
        self.total_avg_waiting_times['volte'].append(self.total_avg_waiting_time_volte)
        self.total_throughputs['volte'].append(self.throughput_volte)
        self.finished_throughputs['volte'].append(self.finished_size_throughput_volte)
        
        avg_non_waiting_time_urllc = (self.sl_win_size - self.total_avg_waiting_time_urllc) / self.sl_win_size
        self.total_avg_waiting_times['urllc'].append(self.total_avg_waiting_time_urllc)
        self.total_throughputs['urllc'].append(self.throughput_urllc)
        self.finished_throughputs['urllc'].append(self.finished_size_throughput_urllc)        
        

        avg_non_waiting_time_video = (self.sl_win_size - self.total_avg_waiting_time_video) / self.sl_win_size
        self.total_avg_waiting_times['video'].append(self.total_avg_waiting_time_video)
        self.total_throughputs['video'].append(self.throughput_video)
        self.finished_throughputs['video'].append(self.finished_size_throughput_video)
        
        


        print('self.total_avg_waiting_time_video is: ', self.total_avg_waiting_time_video)
        print('self.total_avg_waiting_time_urllc is: ', self.total_avg_waiting_time_urllc)
        print('self.total_avg_waiting_time_volte is: ', self.total_avg_waiting_time_volte)

        c1_volte = self.c1_volte    
        c2_volte = self.c2_volte
        waiting_time_sigmoid_volte = 1 - (1 / (1 + math.exp(-c1_volte * (self.total_avg_waiting_time_volte - c2_volte))))
        
        c1_urllc = self.c1_urllc
        c2_urllc = self.c2_urllc

        waiting_time_sigmoid_urllc = 1 - (1 / (1 + math.exp(-c1_urllc * (self.total_avg_waiting_time_urllc - c2_urllc))))

        c1_video = self.c1_video
        c2_video = self.c2_video
        waiting_time_sigmoid_video = 1 - (1 / (1 + math.exp(-c1_video * (self.total_avg_waiting_time_video - c2_video))))

        
        w_volte = self.w_volte
        w_urllc = self.w_urllc
        w_video = self.w_video
    

        print('avg_non_waiting_time_volte is: ', avg_non_waiting_time_volte)
        print('avg_non_waiting_time_urllc is: ', avg_non_waiting_time_urllc)
        print('avg_non_waiting_time_video is: ', avg_non_waiting_time_video)

        if self.total_avg_waiting_time_volte == 0:
            calc_total_avg_waiting_time_volte = 1
        else:
            calc_total_avg_waiting_time_volte = (1/self.total_avg_waiting_time_volte)

        if self.total_avg_waiting_time_video == 0:
            calc_total_avg_waiting_time_video = 1
        else:
            calc_total_avg_waiting_time_video = (1/self.total_avg_waiting_time_video)

        if self.total_avg_waiting_time_urllc == 0:
            calc_total_avg_waiting_time_urllc = 1
        else:
            calc_total_avg_waiting_time_urllc = (1/self.total_avg_waiting_time_urllc)  



        if self.reward_function_type  == 'sigmoid':
            reward = w_volte * (waiting_time_sigmoid_volte) \
            + w_urllc * (waiting_time_sigmoid_urllc) \
            + w_video * (waiting_time_sigmoid_video)   

        elif self.reward_function_type == 'simple':
            reward = w_volte * (calc_total_avg_waiting_time_volte) \
            + w_urllc * (calc_total_avg_waiting_time_urllc) \
            + w_video * (calc_total_avg_waiting_time_video) 

        elif self.reward_function_type == 'shaping':
            reward = w_volte * (waiting_time_sigmoid_volte) \
            + w_urllc * (waiting_time_sigmoid_urllc) \
            + w_video * (waiting_time_sigmoid_video)  

            if (self.total_avg_waiting_time_urllc - c2_urllc) <= 0:
                reward += 3


        print('waiting_time_sigmoid_volte is: ', waiting_time_sigmoid_volte)
        print('waiting_time_sigmoid_urllc is: ', waiting_time_sigmoid_urllc)
        print('waiting_time_sigmoid_video is: ', waiting_time_sigmoid_video)


        self.step_rewards.append(reward)

        if ((self.max_num_steps - 1) <= (self.current_step)):
            done = 2

        else:
            done = False

        if (self.current_traffic_var_step >= self.num_traffic_var):
            self.current_traffic_var_step = 0
            self.finish_time_volte = 0
            self.finish_time_urllc = 0
            self.finish_time_video = 0
        obs = self._next_observation()

        print('done is: ', done)
        print('self.current_step is: ', self.current_step)


        return obs, reward, done, {}
    
    def _take_action(self, action):

        self.actions.append(action)
        if action == 0:
            self.max_size_per_tti_volte = self.max_size_per_tti * 0.3333 
            self.max_size_per_tti_urllc = self.max_size_per_tti * 0.3333 
            self.max_size_per_tti_video = self.max_size_per_tti * 0.3333 
        if action == 1:
            self.max_size_per_tti_volte = math.floor(self.max_size_per_tti * 0.5)
            self.max_size_per_tti_urllc = math.floor(self.max_size_per_tti * 0.3)
            self.max_size_per_tti_video = math.floor(self.max_size_per_tti * 0.2)  

        if action == 2:
            self.max_size_per_tti_volte = math.floor(self.max_size_per_tti * 0.7)
            self.max_size_per_tti_urllc = math.floor(self.max_size_per_tti * 0.2)
            self.max_size_per_tti_video = math.floor(self.max_size_per_tti * 0.1)  

        if action == 3:
            self.max_size_per_tti_volte = math.floor(self.max_size_per_tti * 0.8)
            self.max_size_per_tti_urllc = math.floor(self.max_size_per_tti * 0.1)
            self.max_size_per_tti_video = math.floor(self.max_size_per_tti * 0.1)

        if action == 4:
            self.max_size_per_tti_volte = math.floor(self.max_size_per_tti * 0.1)
            self.max_size_per_tti_urllc = math.floor(self.max_size_per_tti * 0.8)
            self.max_size_per_tti_video = math.floor(self.max_size_per_tti * 0.1)

        if action == 5:
            self.max_size_per_tti_volte = math.floor(self.max_size_per_tti * 0.1)
            self.max_size_per_tti_urllc = math.floor(self.max_size_per_tti * 0.7)
            self.max_size_per_tti_video = math.floor(self.max_size_per_tti * 0.2)

        if action == 6:
            self.max_size_per_tti_volte = math.floor(self.max_size_per_tti * 0.2)
            self.max_size_per_tti_urllc = math.floor(self.max_size_per_tti * 0.7)
            self.max_size_per_tti_video = math.floor(self.max_size_per_tti * 0.1)
            
        if action == 7:
            self.max_size_per_tti_volte = math.floor(self.max_size_per_tti * 0.1)
            self.max_size_per_tti_urllc = math.floor(self.max_size_per_tti * 0.6)
            self.max_size_per_tti_video = math.floor(self.max_size_per_tti * 0.3)

        if action == 8:
            self.max_size_per_tti_volte = math.floor(self.max_size_per_tti * 0.3)
            self.max_size_per_tti_urllc = math.floor(self.max_size_per_tti * 0.6)
            self.max_size_per_tti_video = math.floor(self.max_size_per_tti * 0.1)

        if action == 9:
            self.max_size_per_tti_volte = math.floor(self.max_size_per_tti * 0.3)
            self.max_size_per_tti_urllc = math.floor(self.max_size_per_tti * 0.5)
            self.max_size_per_tti_video = math.floor(self.max_size_per_tti * 0.2)

        if action == 10:
            self.max_size_per_tti_volte = math.floor(self.max_size_per_tti * 0.2)
            self.max_size_per_tti_urllc = math.floor(self.max_size_per_tti * 0.5)
            self.max_size_per_tti_video = math.floor(self.max_size_per_tti * 0.3)
            

        if action == 11:
            self.max_size_per_tti_volte = math.floor(self.max_size_per_tti * 0.3)
            self.max_size_per_tti_urllc = math.floor(self.max_size_per_tti * 0.4)
            self.max_size_per_tti_video = math.floor(self.max_size_per_tti * 0.3)
            

        if action == 12:
            self.max_size_per_tti_volte = math.floor(self.max_size_per_tti * 0.1)
            self.max_size_per_tti_urllc = math.floor(self.max_size_per_tti * 0.5)
            self.max_size_per_tti_video = math.floor(self.max_size_per_tti * 0.4)  
            

        if action == 13:
            self.max_size_per_tti_volte = math.floor(self.max_size_per_tti * 0.4)
            self.max_size_per_tti_urllc = math.floor(self.max_size_per_tti * 0.4)
            self.max_size_per_tti_video = math.floor(self.max_size_per_tti * 0.2)


        if action == 14:
            self.max_size_per_tti_volte = math.floor(self.max_size_per_tti * 0.6)
            self.max_size_per_tti_urllc = math.floor(self.max_size_per_tti * 0.3)
            self.max_size_per_tti_video = math.floor(self.max_size_per_tti * 0.1)


        current_video_traffic = self.traffic_df[(self.traffic_df['type']=='video') &
                   (self.traffic_df['arrival_times'] >= self.current_traffic_var_step*self.sl_win_size) &
                   (self.traffic_df['arrival_times'] < self.sl_win_size*(self.current_traffic_var_step+1)) &
                   (self.traffic_df['user'] <= self.num_users_poisson[0][self.current_traffic_var_step])]
        
        current_volte_traffic = self.traffic_df[(self.traffic_df['type']=='volte') &
                   (self.traffic_df['arrival_times'] >= self.current_traffic_var_step*self.sl_win_size) &
                   (self.traffic_df['arrival_times'] < self.sl_win_size*(self.current_traffic_var_step+1)) &
                   (self.traffic_df['user'] <= self.num_users_poisson[1][self.current_traffic_var_step])]

        
        current_urllc_traffic = self.traffic_df[(self.traffic_df['type']=='urllc') &
                   (self.traffic_df['arrival_times'] >= self.current_traffic_var_step*self.sl_win_size)
                   & (self.traffic_df['arrival_times'] < self.sl_win_size*(self.current_traffic_var_step+1))
                   & (self.traffic_df['user'] <= self.num_users_poisson[2][self.current_traffic_var_step])]
        
        
        current_urllc_traffic = current_urllc_traffic.reset_index(drop=True)
        current_video_traffic = current_video_traffic.reset_index(drop=True)
        current_volte_traffic = current_volte_traffic.reset_index(drop=True)
        

        self.remaining_sizes_volte, self.remaining_times_volte, self.avg_waiting_time_volte, \
        self.avg_turnaround_time_volte, throughput_volte, self.total_p_no_volte, self.num_done_p_volte, \
        self.total_avg_waiting_time_volte, total_size_volte, \
        finished_size_volte = self._rr_scheduler(current_volte_traffic, \
                                                 self.max_size_per_tti_volte, \
                                                 self.finish_time_volte, \
                                                 self.user_priority)


        self.remaining_sizes_urllc, self.remaining_times_urllc, self.avg_waiting_time_urllc, \
        self.avg_turnaround_time_urllc, throughput_urllc, self.total_p_no_urllc, self.num_done_p_urllc, \
        self.total_avg_waiting_time_urllc, total_size_urllc, \
        finished_size_urllc = self._rr_scheduler(current_urllc_traffic, \
                                                  self.max_size_per_tti_urllc, \
                                                  self.finish_time_urllc, \
                                                  self.user_priority)


        self.remaining_sizes_video, self.remaining_times_video, self.avg_waiting_time_video, \
        self.avg_turnaround_time_video, throughput_video, self.total_p_no_video, self.num_done_p_video, \
        self.total_avg_waiting_time_video, total_size_video, \
        finished_size_video = self._rr_scheduler(current_video_traffic, \
                                                  self.max_size_per_tti_video, \
                                                  self.finish_time_video, \
                                                  self.user_priority)
    
        self.remaining_sizes_sum['volte'].append(sum(self.remaining_sizes_volte))
        self.remaining_sizes_sum['urllc'].append(sum(self.remaining_sizes_urllc))
        self.remaining_sizes_sum['video'].append(sum(self.remaining_sizes_video))
    
        self.remaining_sizes['volte'].append(self.remaining_sizes_volte)
        self.remaining_sizes['urllc'].append(self.remaining_sizes_urllc)
        self.remaining_sizes['video'].append(self.remaining_sizes_video)
        
        self.remaining_times_sum['volte'].append(sum(self.remaining_times_volte))
        self.remaining_times_sum['urllc'].append(sum(self.remaining_times_urllc))
        self.remaining_times_sum['video'].append(sum(self.remaining_times_video))
        
        self.remaining_times['volte'].append(self.remaining_times_volte)
        self.remaining_times['urllc'].append(self.remaining_times_urllc)
        self.remaining_times['video'].append(self.remaining_times_video)
        
        self.total_p_numbers['volte'].append(self.total_p_no_volte)
        self.total_p_numbers['urllc'].append(self.total_p_no_urllc)
        self.total_p_numbers['video'].append(self.total_p_no_video)
        
        self.done_p_numbers['volte'].append(self.num_done_p_volte)
        self.done_p_numbers['urllc'].append(self.num_done_p_urllc)
        self.done_p_numbers['video'].append(self.num_done_p_video)

        if ((throughput_volte + sum(self.remaining_sizes_volte)) == 0):
            self.throughput_volte = 1
            self.finished_size_throughput_volte = 1
        else:
            self.throughput_volte = throughput_volte / total_size_volte
            self.finished_size_throughput_volte = finished_size_volte / total_size_volte

        self.volte_demand = total_size_volte
        
        
        if ((throughput_urllc + sum(self.remaining_sizes_urllc)) == 0): 
            self.throughput_urllc = 1
            self.finished_size_throughput_urllc = 1
        else:
            self.throughput_urllc = throughput_urllc / total_size_urllc
            self.finished_size_throughput_urllc = finished_size_urllc / total_size_urllc
        
        self.urllc_demand =  total_size_urllc

        if ((throughput_video + sum(self.remaining_sizes_video)) == 0):
            self.throughput_video = 1
            self.finished_size_throughput_video = 1
        else:
            self.throughput_video = throughput_video / total_size_video
            self.finished_size_throughput_video = finished_size_video / total_size_video
        
        self.video_demand = total_size_video



    # Python program for implementation of RR Scheduling
    def _rr_scheduler(self, current_traffic, max_size_per_ms, finish_time, user_priority):

        total_time = 0 
        total_size = 0
        total_time_counted = 0
        total_size_counted = 0
        finished_jobs_size = 0
        # proc is process list
        proc = []
        wait_time = 0
        turnaround_time = 0
        remaining_sizes = []
        remaining_times = []
        time_step = finish_time - self.sl_win_size


        for index, row in current_traffic.iterrows(): 
            # Getting the input for process
            # process arrival time and burst time
            arrival, burst = row["arrival_times"], row["packet_sizes"]
            remaining_size = burst
            remaining_time = math.ceil(remaining_size/max_size_per_ms)
            # processes are appended to the proc list in following format
            proc.append([arrival, burst, remaining_size, remaining_time, 0, 0, row["user"]])
            # total_time gets incremented with burst time of each process
            total_size += burst

        total_time = math.ceil(total_size/max_size_per_ms)
        total_p_no = len(proc)
        left_over = {}
        # Keep traversing in round robin manner until the total_time == 0
        num_done_processes = 0
        ready_queue = []
        proc_temp = proc.copy()
        process_num = 0
        remaining_tti_bits = max_size_per_ms*self.time_quantum

        remaining_trans = self.max_trans_per_tti
        start_of_time_step = 1
        current_user = 0

        while time_step < finish_time:
            while (len(proc_temp) > 0):
                if proc_temp[0][0] == time_step and start_of_time_step == 1:

                    ready_queue.append(process_num)
                    left_over[process_num] = 0
                    proc[process_num][4] = 1
                    proc_temp.pop(0)
                    process_num +=1
                else:
                    break


            if len(ready_queue) > 0:
                if left_over[ready_queue[0]] == 1:
                    left_over[ready_queue[0]] = 0
                    ready_queue.append(ready_queue.pop(0))    
                if start_of_time_step == 0 and user_priority == 1:
                    for process_number in ready_queue:
                        if proc[process_number][6] == current_user:
                            j = process_number
                            ready_queue.remove(process_number)
                            ready_queue.insert(0,process_number)

                            break
                        else:
                            j = ready_queue[0]
                else:
                    j = ready_queue[0]
                if (proc[j][2] <= remaining_tti_bits and proc[j][2] > 0):
                    current_user = proc[j][6]
                    ready_queue.pop(0)

                    total_size_counted += proc[j][2]
                    finished_jobs_size += proc[j][2]
                    total_time_counted += math.ceil(proc[j][2]/max_size_per_ms)

                    num_done_processes += 1

                    current_p_wait_time = time_step - proc[j][0] - math.ceil(proc[j][1]/max_size_per_ms) + 1

                    wait_time += current_p_wait_time

                    current_p_turnaround_time = time_step - proc[j][0] + 1

                    turnaround_time += current_p_turnaround_time

                    remaining_tti_bits -= proc[j][2]
                    remaining_trans -= 1
                    start_of_time_step = 0

                    # flag is set to 1 once wait time is calculated
                    proc[j][4] = 0
                    proc[j][5] = 1

                    # the process has completely ended here thus setting it's remaining time to 0.
                    proc[j][2] = 0 
                    proc[j][3] = 0


                elif (proc[j][2] > remaining_tti_bits ):
                    current_user = proc[j][6]
                    left_over[j] = 1

                    # if process has not finished, decrementing it's remaining time by time_quantum
                    proc[j][2] -= remaining_tti_bits
                    proc[j][3] -= self.time_quantum              

                    total_time_counted += self.time_quantum
                    total_size_counted += remaining_tti_bits
                    finished_jobs_size += remaining_tti_bits
                    remaining_tti_bits = 0
                    remaining_trans = 0
                    start_of_time_step = 0


                else:
                    print("hello else!")


            if remaining_tti_bits == 0 or remaining_trans == 0 or len(ready_queue) == 0:
                start_of_time_step = 1
                time_step += 1
                remaining_tti_bits = max_size_per_ms*self.time_quantum
                remaining_trans = self.max_trans_per_tti


        if num_done_processes == 0 and total_p_no > 0:
            avg_waiting_time = self.sl_win_size
            avg_turnaround_time = self.sl_win_size
        elif num_done_processes == 0 and total_p_no == 0:
            avg_waiting_time = 0
            avg_turnaround_time = 0
        else:
            avg_waiting_time = (wait_time * 1) / num_done_processes

            avg_turnaround_time = (turnaround_time * 1) / num_done_processes
        total_wait_time = wait_time

        for k in range(len(proc)):

            remaining_sizes.append(proc[k][2])
            remaining_times.append(proc[k][3])

            if proc[k][5] == 0:
                finished_jobs_size -= (proc[k][1] - proc[k][2])

                unfinished_wait = time_step - proc[k][0] - math.ceil(proc[k][1]/max_size_per_ms) + proc[k][3] 
                total_wait_time += unfinished_wait

                
        if (len(proc)>0):
            total_avg_waiting_time = (total_wait_time * 1) / len(proc)

        else:
            total_avg_waiting_time = 0

        return(remaining_sizes, remaining_times, avg_waiting_time, avg_turnaround_time,
               total_size_counted, total_p_no, num_done_processes, total_avg_waiting_time,
               total_size, finished_jobs_size)
