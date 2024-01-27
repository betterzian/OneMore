from abc import abstractmethod

class TimeBlockSim:

    @abstractmethod
    def get_info(self,len):
        pass

    @abstractmethod
    def set_info(self,data):
        pass