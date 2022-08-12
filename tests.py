import torch

class last_m_plus_one():
    def __init__(self, m):
        self.last_list = torch.tensor([0.]*(m+1))
        self.m = m
    
    def append(self, x):
        copy = torch.clone(self.last_list)
        self.last_list[:-1] = copy[1:]
        self.last_list[-1] = x
    
    def return_tensor(self):
        return self.last_list
    
    def __str__(self):
        return "{}".format(self.last_list)

# my_list = last_m_plus_one(4)
# print(my_list)

# for i in range(10):
#     my_list.append(i)
#     print(my_list)



class last_cumulated():
    def __init__(self, m, h):
        self.m = m
        self.h = h
        self.last_cumulated_list = torch.tensor([0.]*(m+h+1))
    
    def append(self, x):
        copy = torch.clone(self.last_cumulated_list)
        self.last_cumulated_list[:-1] = copy[1:]
        self.last_cumulated_list[-1] = x
    
    def before(self):
        before_list = torch.stack([torch.cat([torch.tensor([self.last_cumulated_list[j+self.h-i]]) for i in range(self.h, -1, -1)], 0) for j in range(self.m)], 0)
        return before_list
    
    def after(self):
        after_list = torch.stack([torch.cat([torch.tensor([self.last_cumulated_list[j+self.h-i+1]]) for i in range(self.h, -1, -1)], 0) for j in range(self.m)], 0)
        return after_list
    
    def __str__(self):
        return "{}".format(self.last_cumulated_list)

# my_list = last_cumulated(3, 2)

# for i in range(10):
#     my_list.append(i)
#     print(my_list)

# print("before : {}".format(my_list.before()))
# print("after : {}".format(my_list.after()))