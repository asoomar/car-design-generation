
class BestKeeper(object):
    """Class to keep the best stuff"""
    def __init__(self, rounds):
        self.num_rounds = rounds
        self.best_loss = [1e10 for _ in range(rounds)]
        self.z_vec = [0 for _ in range(rounds)]
        self.best_z = None

    def report(self, round, loss, z):
        if loss < self.best_loss[round]:
            self.best_loss[round] = loss
            self.z_vec[round] = z
        minidx = self.best_loss.index(min(self.best_loss))
        self.best_z = self.z_vec[minidx]

    def get_best(self):
        return self.best_z