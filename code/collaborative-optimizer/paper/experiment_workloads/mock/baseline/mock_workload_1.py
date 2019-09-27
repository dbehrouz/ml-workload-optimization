from workload import Workload
import time


class mock_workload_1(Workload):
    def run(self, **args):
        time.sleep(5)
        return True
