class Env():
    def __init__(self, initpos = 0, obj = 10):
        self.pos = initpos
        self.obj = obj
    
    def measure(self):
        return self.pos
    
    def control_system(self, command):
        self.pos += command
        return self.pos