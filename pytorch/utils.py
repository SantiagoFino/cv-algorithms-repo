class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets the values of the Average meters
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the internal state with a new value and increases the count. 
        The average is then updated.
        Param:
            val (float):
            n (int):
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count