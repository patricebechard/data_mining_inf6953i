import sys

if __name__ == "__main__":

	file = 'instacart/orders_team8.csv'
	prior_file = 'instacart/order_products__prior.csv'
	train_file = 'instacart/order_products__train.csv'

	all_transactions = {}
	print("Train file")
	with open(train_file) as f:

		f.readline()
		n_lines = 0
		for line in f:
			n_lines += 1
			if n_lines % 100000 == 0:
				print(n_lines)
			order_id = line.strip().split(',')[0]
			all_transactions[order_id] = 1

	print("Prior file")
	with open(prior_file) as f:

		f.readline()
		n_lines = 0
		for line in f:
			n_lines += 1
			if n_lines % 100000 == 0:
				print(n_lines)
			order_id = line.strip().split(',')[0]
			all_transactions[order_id] = 1

	print("Number of unique transactions : %d" % (len(all_transactions)))


	with open(file) as f:

		f.readline()
		already_seen = {}
		n_lines = 0
		for line in f:
			
			order_id = line.strip()
			if order_id in already_seen:
				n_duplicates += 1
			else:
				already_seen[order_id] = 1

	print("Number of transactions in custom file : %d" % len(already_seen))

	in_both = 0
	for elem in already_seen:
		if elem in all_transactions:
			in_both += 1

	print("Number of transaction in both custom and other files : %d" % in_both)
