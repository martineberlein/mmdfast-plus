class Learner:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def train(self, epochs):
        self.model.train(self.data)