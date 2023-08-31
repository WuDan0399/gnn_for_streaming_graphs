from itertools import groupby
from operator import attrgetter
import torch
from collections import defaultdict

class Task:
    def __init__(self, op, dst, msg_idx):
        self.op = op
        self.dst = dst
        self.msg_idx = msg_idx


class TaskQueue:
    def __init__(self):
        self.task_buffer = []
        self.message_buffer = []

    def push_task(self, op, dst, msg):
        msg_idx = self.push_message(msg)
        self.task_buffer.append(Task(op,  dst, msg_idx))


    def push_message(self, message: torch.Tensor):
        idx = len(self.message_buffer)
        self.message_buffer.append(message)
        return idx


    def bulky_push(self, destination:int, old_out_neighbors:list, new_out_neighbors:list,
                   old_message:torch.Tensor, new_message:torch.Tensor):
        # update the task queue, create 'delete' and 'add' task for all outgoing edges
        ## Need to clone the messages, otherwise any change on original tensor affects the message in message_buffer
        old_message_idx = self.push_message(old_message.clone())
        new_message_idx = self.push_message(new_message.clone())
        for out_neighbor in old_out_neighbors:
            self.task_buffer.append(Task('delete', destination, out_neighbor, old_message_idx))
        for out_neighbor in new_out_neighbors:
            self.task_buffer.append(Task('add', destination, out_neighbor, new_message_idx))

    def empty(self):
        self.task_buffer.clear()
        self.message_buffer.clear()

    def reduce(self, aggregator: str) -> dict :
        # First, group tasks by 'dst' and 'op'
        task_dict = defaultdict(lambda : defaultdict(list))
        for task in self.task_buffer :
            message = self.message_buffer[task.msg_idx]
            task_dict[task.dst][task.op].append(message)

        # Then, reduce messages of each group
        reduce_fn = torch.min if aggregator == 'min' else torch.max
        for dst, ops in task_dict.items() :
            for op, messages in ops.items() :
                ops[op] = reduce_fn(torch.stack(messages), dim=0).values

        return task_dict


    def print(self):
        for task in self.task_buffer:
            print(f"<{task.op}, {task.src}, {task.dst}, {self.message_buffer[task.msg_idx].shape}>")

    # def reduce(self, aggregator:str) -> dict:
    #     # groupby returns consecutive keys, so need to sort first
    #     # reduce with given aggregator after grouped
    #     result = {}
    #     task_q_sorted = sorted(self.task_buffer, key=attrgetter('dst', 'op')) # sorting by 'dst' first, then by 'op'
    #     for dst, tasks_in_dst in groupby(task_q_sorted, key=attrgetter('dst')):
    #         # todo : return reduced result
    #         result[dst] = {op: [(task.src, task.msg) for task in tasks] for op, tasks in groupby(tasks_in_dst, key=attrgetter('op'))}
    #     return result
