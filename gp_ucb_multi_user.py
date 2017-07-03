import sys
import math
import argparse
import numpy as np
from numpy.linalg import inv

class GP:
	user = 0
	model = 0

	current_mean = None
	pre_mean = None
	prior_mean=None
	current_var = None
	pre_var = None
	kernel = None
	observe_accu = None
	cost = None
	last_run_time = None
	run_times = None
	run_time = None
	last_run_algo = None
	second_last_run_algo = None
	accuracy = None
	sum_accuracy = 0
	#other data structures
	a = None
	y = None
	beta = None
	sigma_t_k = None
	sigma_t = None
	c_t = None

	max_cost = 0
	max_iteration = 0



	def __init__(self, mean, cov, nUser, nModel, test, cost, max_cost, _with):
		(self.user, self.model) = (nUser, nModel)

		self.current_mean = np.zeros((nUser, nModel))
		self.pre_mean = np.zeros((nUser, nModel))
		self.prior_mean = np.zeros((nUser, nModel))
		self.current_var = np.zeros((nUser, nModel))
		self.pre_var = np.zeros((nUser, nModel))
		self.kernel = np.zeros((nUser, nModel, nModel))

		for i in range (nUser):
			self.current_mean[i] = mean
			self.pre_mean[i] = mean
			self.prior_mean[i] = mean
			for j in range(nModel):
				self.current_var[i][j] = cov[j][j]
				self.pre_var[i][j] = cov[j][j]
			self.kernel[i] = cov

		self.observe_accu = test
		self.cost = cost
		self.max_cost = max_cost

		self.last_run_time = np.zeros(nUser, dtype=np.int)
		self.run_times = np.zeros(nUser, dtype=np.int)
		self.run_time = np.zeros((nUser, nUser * nModel + 5), dtype=np.int)
		self.last_run_algo = np.zeros(nUser, dtype=np.int)-1
		self.second_last_run_algo = np.zeros(nUser, dtype=np.int)-1
		self.accuracy = np.zeros(nUser)
		self.sum_accuracy = 0

		#other data structures
		self.a = np.zeros((nUser, nUser * nModel + 5), dtype=np.int)
		self.y = np.zeros(nUser * nModel + 5)
		self.beta = np.zeros(nUser * nModel + 5)
		self.sigma_t_k = np.zeros((nUser,nUser * nModel+5))
		self.sigma_t = np.zeros((nUser, nUser * nModel + 5, nUser * nModel + 5))
		self.c_t = np.zeros((nUser * nModel + 5, nUser * nModel + 5))

		self.max_iteration = nUser * nModel


		if _with == 0:
			for i in range(1, nUser * nModel + 5):
				self.beta[i] = math.log(100 * i * i * 1.0 / 0.01)
			self.beta[0] = math.log(100 * 1 * 1 /0.01)
		else:
			for i in range(1, nUser * nModel + 5):
				pi = 3.14159265359
				self.beta[t] = 2 * max_cost * math.log(pi * pi * 100 * i * i * 1.0 / (0.5 * 6))
			self.beta[0] = 2 * max_cost * math.log(3.14 * 3.14 * 100 /(0.5 * 6))

	def choose_model(self, t, user_id):
		max_ = -100000.0
		algo_ = -5

		for algo in range(self.model):

			tmp =  math.sqrt(self.beta[int(self.last_run_time[user_id] * 1.0 / self.cost[user_id][algo])]) * math.sqrt(self.current_var[user_id][algo])
			tmp_reward = self.current_mean[user_id][algo] + tmp
			
			if tmp_reward > max_: # 
				max_ = tmp_reward
				algo_ = algo
		self.a[user_id][t] = algo_ # For user user_id, at time stamp t, choose algorithm a[i][t]

	def compute_criteria(self):
		average = 0
		for i in range(self.user):
			average = average + math.sqrt(self.beta[self.last_run_time[i]] * 1.0) * math.sqrt(self.current_var[i][self.second_last_run_algo[i]])
		average = average * 1.0 / self.user

		return average


	def choose_user(self, algo, t):
		user_id = -5

		# the same warm start for all algorithms
		if t <= self.user:
			#print t, self.user
			user_id = t - 1  

		else:
			if algo == "rr":
				user_id = (t-1) % self.user
	
			elif algo == "maximal mean-variance drop" or algo == "maximal variance drop":
				average = compute_criteria()

				if algo == "maximal mean-variance drop":
					max_mean_variance = -100000
					for i in range(self.user):
						if (math.sqrt(self.beta[self.last_run_time[i]] * 1.0) * math.sqrt(self.current_var[i][self.second_last_run_algo[i]])) > average : 
							a1 = self.current_mean[i][self.last_run_algo[i]] + math.sqrt(self.beta[last_run_time[i]] * 1.0 / self.cost[i][self.a[i][t]]) * math.sqrt(self.current_var[i][self.last_run_algo[i]])
							a2 = self.pre_mean[i][self.second_last_run_algo[i]] + math.sqrt(self.beta[last_run_time[i]-1] * 1.0 / self.cost[i][self.a[i][t]]) * math.sqrt(self.pre_var[i][self.second_last_run_algo[i]])
							
							mean_variance = a2-a1
													
							if mean_variance > max_mean_variance:
								max_mean_variance = mean_variance
								user_id = i
	
				else:
					max_info_gain = -100000
					for i in range(self.user):
						if (math.sqrt(self.beta[self.last_run_time[i]]*1.0) * math.sqrt(self.current_var[i][self.second_last_run_algo[i]])) >= average:
							drop = math.sqrt(current_var[i][last_run_algo[i]]) - math.sqrt(pre_var[i][second_last_run_algo[i]])
							if drop > max_info_gain:
								max_info_gain = drop
								user_id = i	

			elif algo == "random":
				user_id = random.randint(0, self.user - 1)

	
		return user_id


	def update(self, user_id, t):

		# maintain some data structures

		self.run_times[user_id] += 1
		#print len(self.run_time)
		#print user_id, self.run_times[user_id]
		self.run_time[user_id][self.run_times[user_id]] = t
		if (self.run_times[user_id] >=2):
			self.second_last_run_algo[user_id] = self.last_run_algo[user_id]
		self.last_run_algo[user_id] = self.a[user_id][t]
		self.last_run_time[user_id] = t
		self.y[t] = self.observe_accu[user_id][self.a[user_id][t]] + 0.01
		self.accuracy[user_id] = self.observe_accu[user_id][self.a[user_id][t]]
	
		if self.a[user_id][t]!= -5:
			print (str(user_id) + " " + str(self.a[user_id][t]))
			#fout.write(str(user_id) + " " + str(self.a[user_id][t]) + "\n")
			#for v in range(self.user):
			#	self.sum_accuracy = self.sum_accuracy + self.accuracy[v]	
			#	fout2.write(str(self.cost[v][self.a[v][self.last_run_algo[v]]])+ " " + str(self.accuracy[v])+" ")
			#fout2.write("\n")
	
	
		u = user_id
		for i in range(self.run_times[u]):
			for j in range(self.run_times[u]):
				self.sigma_t[u][i][j] = self.kernel[u][self.a[u][self.run_time[u][i+1]]][self.a[u][self.run_time[u][j+1]]]
				if i == j:
					self.sigma_t[u][i][j] = self.sigma_t[u][i][j] + 0.01 * 0.01
		x=[]
		for v in range(self.run_times[u]):
			x.append([self.sigma_t[u][v][:self.run_times[u]]])
		self.c_t = inv(np.asarray(x).reshape(self.run_times[u],self.run_times[u]))
		y_tmp=[]
		m_tmp=[]
		tmp3=[]
		for i in range(self.run_times[u]):
			y_tmp.append(self.y[self.run_time[u][i+1]])
			m_tmp.append(self.prior_mean[u][self.a[u][self.run_time[u][i+1]]])
			tmp3.append(self.y[self.run_time[u][i+1]] - self.prior_mean[u][self.a[u][self.run_time[u][i+1]]])
		for i in range(self.model):
			for j in range(self.run_times[u]):
			
				self.sigma_t_k[u][j] = self.kernel[u][self.a[u][self.run_time[u][j+1]]][i]
		
			tmp1 = np.dot(self.sigma_t_k[u][:self.run_times[u]],self.c_t)			
			tmp2 = np.dot(tmp1, tmp3)
			self.pre_mean[u][i] = self.current_mean[u][i]
			self.current_mean[u][i] = tmp2 + self.prior_mean[u][i]
			#current_mean[u][i] = tmp2 
			update = np.dot(tmp1,self.sigma_t_k[u][:self.run_times[u]])
			self.pre_var[u][i] = self.current_var[u][i]
			
			self.current_var[u][i] = self.kernel[u][i][i] - update
			
				

def compute_prior(nTrain, nModel, train_dir, sigma):

	train = np.zeros((nTrain, nModel))
	mean = np.zeros(nModel)
	cov = np.zeros((nModel, nModel))
	i = 0
	for line in open(train_dir,'r'):
		line = line.strip("\n").split("\t") # this may be changed according to the format of the training dataset
		assert len(line) == nModel
		for j in range(len(line)):
			train[i][j] = float(line[j])
		i += 1

	for j in range(nModel):
		for i in range(nTrain):
			mean[j] += train[i][j]
		mean[j] = mean[j] * 1.0 / nTrain

	for i in range(nModel):
		for j in range(nModel):
			tmp = 0
			for k in range(nTrain):
				tmp += (train[k][i] - train[k][j]) * (train[k][i] - train[k][j])
			cov[i][j] = math.exp((-1)* tmp * 1.0 / (2 * sigma * sigma))
	return mean, cov

def read_in(te, co, _with, nModel, nUser):
	accu=np.zeros((nUser, nModel))
	cost=np.zeros((nUser, nModel)) + 1
	max_cost = 0
	j = 0

	for line in open(te,'r'):
		line = line.strip("\n").split("\t")
		for i in range(len(line)):
			accu[j][i] = float(line[i])
		j = j + 1

	if _with == 1:
		j = 0
		for line in open(co,'r'):
			line = line.strip(" \n").split(" ")
			for i in range(len(line)):
				cost[j][i] = float(line[i])
				if cost[j][i] > max_cost:
					max_cost = cost[j][i]
			j = j + 1

	return accu, cost, max_cost



def main():
	mean, covariance = compute_prior(FLAGS.train_dataset, FLAGS.model, FLAGS.train, FLAGS.lengthscale)
	test, cost, max_cost = read_in(FLAGS.test, FLAGS.cost, FLAGS.with_cost, FLAGS.model, FLAGS.user)
	gp = GP(mean, covariance, FLAGS.user, FLAGS.model, test, cost, max_cost, FLAGS.with_cost)

	#fout = open("sequence_"+FLAGS.choose_user+".txt",'w')
	#fout2 = open("regret_"+FLAGS.choose_user+".txt",'w')

	t = 1
	while t <= gp.max_iteration:
		for i in range(FLAGS.user):
			gp.choose_model(t, i)

		user_id = gp.choose_user(FLAGS.choose_user, t)
		print user_id

		# at each round, we only update the 'user_id''s mean and covariance
		gp.update(user_id, t) 

		t += 1

	return

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_dataset", type = int, default = 0, help = "how many training datasets")
	parser.add_argument("--user", type = int, default = 0, help = "how many users")
	parser.add_argument("--model", type = int, default = 0, help = "how many models")
	parser.add_argument("--with_cost", type = int, default = 0, help="cost sensitive:1; cost insensitive:0")
	parser.add_argument("--train", type = str, default = "train.txt", help = "the path for the training dataset")
	parser.add_argument("--test", type = str, default = "test.txt", help = "the path for the testing dataset")
	parser.add_argument("--cost", type = str, default = "cost.txt", help = "the path for the data indicating the cost")
	parser.add_argument("--lengthscale", type = float, default = 1, help = "the length scale then computing the prior covariance")
	parser.add_argument("--choose_user", type = str, default = "rr", help = "the algorithm we use to choose users/datasets to run")

	# To do: add options for choosing kernels to compute the prior covariance; currently it's RBF
	# python gp_ucb_multi_user.py --train_dataset=15 --user=5 --model=9 --with_cost=0 \
	# --train=/Users/serena/Desktop/newdata/deep_learning/train.txt \
	# --test=/Users/serena/Desktop/newdata/deep_learning/test.txt \
	# --cost=/Users/serena/Desktop/newdata/deep_learning/cost.txt \
	# --lengthscale=1 \
	# --choose_user=rr

	FLAGS, unparsed = parser.parse_known_args()
	main()
