import numpy as np

class State:
	def __init__(self):
		self.transitions = {}

	def set_transitions(self, transitions):
		self.transitions = transitions

	def next(self, signal):
		return self.transitions[signal]

	def sample(self):
		return np.random.choice(list(self.transitions.keys()))

class Automaton:
	def __init__(self, start, end):
		self.start = start
		self.end = end

	def check(self, string):
		cur_state = self.start
		try:
			for c in string:
				cur_state = cur_state.next(c)
		except:
			return False
		return cur_state is self.end

	def generate(self):
		string = ''
		cur_state = self.start
		while cur_state is not self.end:
			signal = cur_state.sample()
			string += signal
			cur_state = cur_state.next(signal)
		return string
