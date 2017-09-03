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
	state6 = State()
	state7 = State()
	state8 = State()
	state9 = State()
	state10 = State()
	state11 = State()
	state12 = State()
	state13 = State()
	end = State()

	# start
	start.set_transitions({'B': state0})
	state0.set_transitions({'T': state1, 'P': state8})
	# grammaire du haut
	state1.set_transitions({'T': state2, 'P': state3})
	state2.set_transitions({'S': state2, 'X': state4})
	state3.set_transitions({'T': state3, 'V': state5})
	state4.set_transitions({'X': state3, 'S': state6})
	state5.set_transitions({'P': state4, 'V': state6})
	state6.set_transitions({'T': state7})
	# grammaire du bas
	state8.set_transitions({'T': state9, 'P': state10})
	state9.set_transitions({'S': state9, 'X': state11})
	state10.set_transitions({'T': state10, 'V': state12})
	state11.set_transitions({'X': state10, 'S': state13})
	state12.set_transitions({'P': state11, 'V': state13})
	state13.set_transitions({'P': state7})
	# end
	state7.set_transitions({'E': end})
	return Automaton(start, end)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate strings using the symmetrical Reber grammar.')
	parser.add_argument('-N', type=int, help='number of strings', required=False, default=1)
	parser.add_argument('-P', type=float, help='probability of a loop', required=False, default=0.5)
	
	args = parser.parse_args()
	N = args.N
	p_loop = args.P

	automaton = create_automaton(p_loop)

	for _ in range(N):
		print(automaton.generate())