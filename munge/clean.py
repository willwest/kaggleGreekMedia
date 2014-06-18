with open('../data/wise2014-test.libsvm', 'r') as f:
	for line in f:
		new_line = "1" + line[1:]
		print new_line.strip()
