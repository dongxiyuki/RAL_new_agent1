# 模型训练部分

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import random
import os
import json

def train(model, data, agent, al, epoch, env, budget, MEMORY_CAPACITY):
    # single agent
    center_point = []
    for i_episode in range(epoch):
        print('epoch:', i_episode + 1)
        data.reset()
        data.first_random(200)
        model.train()
        model.give_label()
        acc_change = model.acc_change()
        print(acc_change)
        # if acc_change < -0.3:
        #     print('break!')
        #     os.system('pause')
        s = al.update()
        for j in range(budget):
            s = Variable(torch.from_numpy(s)).type(torch.FloatTensor)
            a = agent.choose_action(s, j)
    
            s_next, r = env.feedback(a) 
            agent.store_transition(s, a, r, s_next) 
            s = s_next 
            
            if agent.memory_counter > MEMORY_CAPACITY:
                agent.Learn()
            
            # print('rest_cluster:', data.unlabeled_num_of_each_cluster)
            # center_point.append(data.center_points.tolist())
            print('rest_data:', model.get_rest_unlabeled_data_effect())

        # model.net.reset()
        torch.save(agent.target_net, '\model\model_cluster_20_v12.pkl')
    
    # with open("center.json","w") as f:
    #     json.dump(center_point, f)
