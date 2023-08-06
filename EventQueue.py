import torch
from collections import defaultdict
from typing import Callable
from itertools import chain

class Event:
    def __init__(self, op, dst, msg_idx):
        self.op = op
        self.dst = dst
        self.msg_idx = msg_idx


class EventQueue:
    def __init__(self):
        self.event_buffer_monotonic = []
        self.event_buffer_accumulative = []
        self.event_buffer_user = []  # todo, implement for user-defined events
        self.message_buffer = []


    def push_monotonic_event(self, op, dst, msg):
        msg_idx = self.push_message(msg)
        self.event_buffer_monotonic.append(Event(op, dst, msg_idx))


    def push_accumulative_event(self, op, dst, msg):
        msg_idx = self.push_message(msg)
        self.event_buffer_accumulative.append(Event(op, dst, msg_idx))

    def push_user_event(self, op, dst, msg):
        msg_idx = self.push_message(msg)
        self.event_buffer_user.append(Event(op, dst, msg_idx))


    def push_message(self, message: torch.Tensor):
        idx = len(self.message_buffer)
        self.message_buffer.append(message)
        return idx


    def bulky_push(self, old_out_neighbors:list, new_out_neighbors:list,
                   old_message:torch.Tensor, new_message:torch.Tensor, operation_type:str="monotonic"):
        if operation_type == "monotonic":
            # update the task queue, create 'remove' and 'insert' task for all outgoing edges
            ## Need to clone the messages, otherwise any change on original tensor affects the message in message_buffer
            old_message_idx = self.push_message(old_message.clone())
            new_message_idx = self.push_message(new_message.clone())
            for out_neighbor in old_out_neighbors:
                self.event_buffer_monotonic.append(Event('remove', out_neighbor, old_message_idx))
            for out_neighbor in new_out_neighbors:
                self.event_buffer_monotonic.append(Event('insert', out_neighbor, new_message_idx))
        elif operation_type == "accumulative":
            shared_neighbours = set(old_out_neighbors) & set(new_out_neighbors)
            if shared_neighbours:
                delta_message_idx = self.push_message(new_message.clone() - old_message.clone())
                for shared_neighbour in shared_neighbours:
                    self.event_buffer_accumulative.append(Event('update', shared_neighbour, delta_message_idx))
            removed_neighbours = set(old_out_neighbors) - shared_neighbours
            if removed_neighbours:
                message_idx = self.push_message((-old_message).clone())
                for removed_neighbour in removed_neighbours:
                    self.event_buffer_accumulative.append(Event('update', removed_neighbour, message_idx))
            added_neighbours = set(new_out_neighbors) - shared_neighbours
            if added_neighbours:
                message_idx = self.push_message(new_message.clone())
                for added_neighbour in added_neighbours:
                    self.event_buffer_accumulative.append(Event('update', added_neighbour, message_idx))
        else:
            raise ValueError(f"Operation type {operation_type} is not supported")

    def empty(self):
        self.event_buffer_monotonic.clear()
        self.event_buffer_accumulative.clear()
        self.event_buffer_user.clear()
        self.message_buffer.clear()
    
    def default_factory(self):
        return defaultdict(list)

    def reduce(self, monotonic_aggregator:Callable,
               accumulative_aggregator:Callable,
               user_defined_reduce_function:Callable) -> dict :
        # First, group tasks by 'dst' and 'op'
        task_dict = defaultdict(self.default_factory)
        for task in chain(self.event_buffer_monotonic, self.event_buffer_accumulative, self.event_buffer_user):
            message = self.message_buffer[task.msg_idx]
            task_dict[task.dst][task.op].append(message)

        # Then, reduce messages of each group
        for dst, ops in task_dict.items() :
            for op, messages in ops.items() :
                if op == 'remove' or op == 'insert' :
                    task_dict[dst][op] = monotonic_aggregator(messages)
                elif op == 'update':
                    task_dict[dst][op] = accumulative_aggregator(messages)
                else:  # user-defined events
                    task_dict[dst][op] = user_defined_reduce_function(messages)
        return task_dict


    def print(self):
        print("Events fot monotonic operations")
        for event in self.event_buffer_monotonic:
            print(f"<{event.op}, {event.dst}, {self.message_buffer[event.msg_idx].shape}>")
        print("Events fot accumulative operations")
        for event in self.event_buffer_accumulative:
            print(f"<{event.op}, {event.dst}, {self.message_buffer[event.msg_idx].shape}>")
