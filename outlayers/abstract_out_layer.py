import abc


class AbstractOutLayer(abc.ABC):

    @abc.abstractmethod
    def __call__(self, logit):
        pass

    @abc.abstractmethod
    def loss(self, logits, target):
        pass

    @abc.abstractmethod
    def maximum_target(self, logits_len):
        pass

    @abc.abstractmethod
    def get_prediction(self, score):
        pass

    @abc.abstractmethod
    def get_item(self, score, idx, subidx):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def __repr__(self):
        pass