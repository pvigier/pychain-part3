import argparse
from automaton import *

def create_automaton():
	start = State()
	state0 = State()
	state1 = State()
	state2 = State()
	state3 = State()
	state4 = State()
	state5 = State()
	end = State()

	start.set_transitions({'B': state0})
	state0.set_transitions({'T': state1, 'P': state2})
	state1.set_transitions({'S': state1, 'X': state3})
	state2.set_transitions({'T': state2, 'V': state4})
	state3.set_transitions({'X': state2, 'S': state5})
	state4.set_transitions({'P': state3, 'V': state5})
	state5.set_transitions({'E': end})
	return Automaton(start, end)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate strings using Reber grammar.')
	parser.add_argument('-N', type=int, help='number of strings', required=False, default=1)
	
	args = parser.parse_args()
	N = args.N

	automaton = create_automaton()

	for _ in range(N):
		print(automaton.generate())