import math


def calc_confidence_interval(accuracies, k=10):

	errors = [1 - acc for acc in accuracies]

	mean  = sum(errors)/(float(len(errors)))
	
	SE = math.sqrt((1/(float(k)*float(k-1)))  * sum([math.pow(errors[i] - mean,2) for i in range(0 ,len(errors))]))
	CI = 2.23 * SE
	return (1-	mean, CI)



def paired_t_test(acc1, acc2, k = 10):

	diffs = [a1 - a2 for a1,a2 in zip(acc1,acc2)]
	return calc_confidence_interval(diffs)
