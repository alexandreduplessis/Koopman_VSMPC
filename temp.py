import torch

class Last_cumulated(): # m and T are switched here, to modify
    def __init__(self, m, h, T):
        self.m = m
        self.h = h
        self.T = T
        self.last_cumulated_list = torch.tensor([0.]*(m+h+1))
    
    def append(self, x):
        copy = torch.clone(self.last_cumulated_list)
        self.last_cumulated_list[:-1] = copy[1:]
        self.last_cumulated_list[-1] = x
    
    def return_list(self):
        return self.last_cumulated_list
    
    def before(self, nb=None):
        if nb == None:
            nb = self.T
        # print("taille before :", nb)
        before_list = torch.stack([torch.cat([torch.tensor([self.last_cumulated_list[j+self.h-i]]) for i in range(self.h, -1, -1)], 0) for j in range(nb)], 0)
        return before_list
    
    def after(self, nb=None):
        if nb == None:
            nb = self.T
        after_list = torch.stack([torch.cat([torch.tensor([self.last_cumulated_list[j+self.h-i+1]]) for i in range(self.h, -1, -1)], 0) for j in range(nb)], 0)
        return after_list
    
    def __str__(self):
        return "{}".format(self.last_cumulated_list)