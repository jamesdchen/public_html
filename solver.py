from networks import *
from pde import *

import time
import os
from os import environ

import matplotlib.pyplot as plt
plt.switch_backend('agg') # If plots return errors, change the backend to something else

import multiprocessing

import shutil


_MULTIPROCESSING_CORE_COUNT = multiprocessing.cpu_count()-1
SWAP_MEMORY = True


def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


print("------------------------------------------------------")
print(" WILL USE " + str(_MULTIPROCESSING_CORE_COUNT) + " THREADS ")
print("------------------------------------------------------")


class SimpleSolver:
	
	##
	# @brief Subclass that contain the parameters of the simulation: PDE, horizon,
	# discretization, network.
	class Parameters:
	
		def _constant_learning_rate(self, m):
			return self.learning_rate
	
		def __init__(
				self,
				pde,
				d,
				T,
				n_timesteps,
				X0,
				Yini
				):
			
			self.pde = pde
			
			self.d = d
			self.T = T
			
			self.dtype = tf.float32
			
			# Time discretization
			self.n_timesteps = n_timesteps
			self.delta_t = (self.T + 0.0) / self.n_timesteps
			self.tstamps = 0.0 + self.delta_t * np.arange(self.n_timesteps+1)
			
			# Default learning parameters
			self.network = None # This should be defined when using some solvers that do not implement a network
			self.batch_size = 64
			self.valid_size = 256
			self.n_miniter = 2000
			self.n_maxiter = 100000
			self.n_dispstep = 100
			self.learning_rate = 5e-4
			
			# Define learning rate function
			self.get_learning_rate = self._constant_learning_rate
			self.use_predefined_learning_rate_strategy = False

			self.Yini = Yini

			# Initial condition
			# Should be a line vector (thus reshape)
			self.X0 = np.reshape(X0, (1, self.d))
			
			self.last_loss = 0
			
			# Divide nn by dimension (useful for initialization in AllenCahn)
			self.divide_nn_by_dimension = True
			
			# Normalize X's
			self.normalize_input_X = True
			
			# Normalize Y's
			self.normalize_input_Y_gX = True
			
			self.override_mc=False
			self.compute_reference=True
			
			# Learning rate decrease rule
			self.check_step_count = 1000
			
			# Initializing Y0 around E(g(X_T))?
			self.initialize_Y0_around_gXT = True
			
			# L2 reg amount
            # This will add a regularization term: weight * sum_W (norm(W)^2) to the loss function
			self.l2_regularizer_weight = 0#1e-5
			
			
	def __init__(
		self, 
		a_parameters, 
		base_name, 
		n_threads=8, 
		seed=None, 
		name_suffix=None, 
		record_session=False, 
		save_session=False,
		output_directory = None
		):
		
		# Name for save paths
		self.base_name = base_name
		self.timestamp = str(int(np.floor(time.time() * 1000)))
		if output_directory is None:
			self.output_directory = "./output/" + self.timestamp + "_" + self.base_name
		else:
			self.output_directory = output_directory
		
		if name_suffix is not None:
			self.output_directory += "_" + name_suffix + "/"
			
		if not os.path.exists(self.output_directory):
			os.makedirs(self.output_directory)
		
		self.par = a_parameters
		
		if seed is not None:
			self.seed = seed
			tf.set_random_seed(self.seed)
		else:
			self.seed = None
		
		self.record_session = record_session
		self.save_session = save_session
		
		self.open()
	
	##
	# @brief Open a new session and create the test network.
	# 
	# @param self 
	def open(self):
		# Set number of threads
		#~ self.sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=n_threads))
		
		# Or monitor the placement of the tf variables (CPU? GPU?)
		#~ self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
		
		# Or control gpu memory growth
		config = tf.ConfigProto()
		
		# Allow memory growth or not
		if environ.get('ALLOCATE_ALL_GPU_MEMORY') is not None:	
			print("------------------------------------------------------")
			print(" WILL ALLOCATE ALL GPU MEMORY ")
			print("------------------------------------------------------")
		else:
			print("------------------------------------------------------")
			print(" WILL USE GPU MEMORY GROWTH ")
			print("------------------------------------------------------")
			config.gpu_options.allow_growth=True
		
		config.intra_op_parallelism_threads = _MULTIPROCESSING_CORE_COUNT
		config.inter_op_parallelism_threads = _MULTIPROCESSING_CORE_COUNT
		self.sess = tf.Session(config=config)
		
		# Or default session
		#~ self.sess = tf.Session()
		
		self.t_normalizing_mean = 0.0
		self.t_normalizing_std = 1.0
		
		if self.par.normalize_input_X or self.par.normalize_input_Y_gX or self.par.initialize_Y0_around_gXT:
			
			print("Computing stats on X")
			
			## Get initial normalization means and variance
			nb_initial_normalization_sample = 10000
			initial_normalization_sample_paths_X, _ = self.get_sample_paths(nb_initial_normalization_sample, self.par.X0)
			
			if self.par.normalize_input_X:
				print("Computing normalization mean and variance")

				# For easier computation, consider all dimensions are symmetrical and perform computations only on X_0
				self.initial_normalization_means = np.mean(initial_normalization_sample_paths_X[:, 0, :])
				self.initial_normalization_vars = np.std(initial_normalization_sample_paths_X[:, 0, :])
				print("self.initial_normalization_means", self.initial_normalization_means)
				print("self.initial_normalization_vars", self.initial_normalization_vars)
				
				# Furthermore, normalize t
				# Note that we only give t=0... t=T-deltat to the neural network
				self.t_normalizing_mean = np.mean(self.par.tstamps[:-1])
				self.t_normalizing_std = self.t_normalizing_mean + 1e-8
				print("self.t_normalizing_mean", self.t_normalizing_mean)
				print("self.t_normalizing_std", self.t_normalizing_std)
			
			if self.par.initialize_Y0_around_gXT or self.par.normalize_input_Y_gX:
				print("Computing initial Y0 value")
				tmp = self.par.pde.g_np(self.par.T, initial_normalization_sample_paths_X)
				print("tmp.shape", tmp.shape)
				self.Y0_init_value = np.mean(tmp[:, :, -1], axis=0, keepdims=False)[0]
				tmp2 = np.mean(tmp[:, 0, :], axis=1)
				self.Y_mean_value = np.mean(tmp2)
				self.Y_stddev_value = self.Y_mean_value
				print("self.Y0_init_value", self.Y0_init_value)
				print("self.Y_mean_value", self.Y_mean_value)
				print("self.Y_stddev_value", self.Y_stddev_value)
		
		if self.par.initialize_Y0_around_gXT:
			self.Y0_initializer = tf.constant_initializer(self.Y0_init_value)
		else:
			print("Y0 random uniform initializer")
			self.Y0_initializer = tf.random_uniform_initializer(minval=self.par.Yini[0], maxval=self.par.Yini[1], dtype=tf.float32, seed=1)
			print("self.Y0_initializer", self.Y0_initializer)
		
		if self.par.normalize_input_Y_gX:
			print("Input Y and gX (if it exists) will be scaled by Y0")
			self.Y_normalizing_mean = self.Y_mean_value
			self.Y_normalizing_std = np.abs(self.Y_stddev_value) + 1e-8
		else:
			self.Y_normalizing_mean = 0.0
			self.Y_normalizing_std = 1.0
		
		# Build the test network	
		with tf.variable_scope(self.base_name, reuse=tf.AUTO_REUSE):
		
			# Test placeholders
			self.X_test = tf.placeholder(tf.float32, (None, self.par.d, self.par.n_timesteps+1))
			self.dB_test = tf.placeholder(tf.float32, (None, self.par.d, self.par.n_timesteps))
	
			self.test_Y0, self.test_loss, self.test_Y, self.test_Z = self.build(
				X_in=self.X_test, 
				dB_in=self.dB_test, 
				training_flag=False, 
				reuse=tf.AUTO_REUSE
				)
		
		print(self.test_Y)
		print(self.test_Z)
		
		# Set saver up
		self.saver = tf.train.Saver()
		
	##
	# @brief Create a graph taking inputs and returning the path and the loss. 
	# 
	# @param self 
	# @param X_in Input data tensor.
	# @param dB_in Brownian data tensor.
	# @param training_flag Should be True for the training graph, False for the test graph.
	# @param reuse Should be True if no variable should be created, False else, or tf.AUTO_REUSE.
	#
	# @return Y0 The initial Y tensor.
	# @return loss The loss tensor.
	# @return Y_list A list of tensors depicting Y at each timestep.
	# @return Z_list A list of tensors depicting Z at each timestep.
	def build(
		self, 
		X_in, 
		dB_in, 
		training_flag, 
		reuse
		):
			
		start_time = time.time()
		
		# Sample size
		sample_size = tf.shape(X_in)[0]
		
		# Variables
		
		# Y0
		Y0 = tf.get_variable("Y0", [], tf.float32, self.Y0_initializer)
		Y = tf.tile(tf.reshape(Y0, [1, 1]), [sample_size, 1])
		
		if self.par.normalize_input_X:
			print("Input X of the neural networks will be normalized by the statically computed mean and variance")
			unnormalized_input = X_in
			X_normalized = (unnormalized_input - self.initial_normalization_means) / np.sqrt(self.initial_normalization_vars + 1e-6)
		else:
			print("Input X of the neural networks will NOT be normalized by the statically computed mean and variance")
			X_normalized = X_in
		
		
		## Create Z from X
		## for horizontal networks (construction layer by layer)
		if not self.par.network.is_vertical():
			print(">>> Network is horizontal.")
			if self.par.network.handles_starting_gradient():
				print("Network handling initial gradient.")
				
				Z_horizontal = self.par.network.get_horizontal_network(
					a_input=X_normalized[:, :, :-1], 
					a_namespace="network", 
					a_training=training_flag, 
					a_timesteps=self.par.tstamps[:-1], 
					reuse=reuse
					)
				if self.par.divide_nn_by_dimension:
					Z_horizontal /= self.par.d
			else:
				print("Network not handling initial gradient. Creating a Z0")
				# Z0
				# We divide Z0 by d in order to have a small initialization value. This rule is arbitrary
				Z0 = tf.get_variable("Z0", None, tf.float32, tf.random_uniform([1, self.par.d], minval=-.05, maxval=.05, dtype=tf.float32, seed=2))
				if self.par.divide_nn_by_dimension:
					Z0 = Z0 / self.par.d
				Z0_tiled = tf.tile(Z0, [sample_size, 1])
				
				Z_horizontal = self.par.network.get_horizontal_network(
					a_input=X_normalized[:, :, 1:-1], 
					a_namespace="network", 
					a_training=training_flag,
					a_timesteps=(self.par.tstamps[1:-1] - self.t_normalizing_mean) / self.t_normalizing_std, 
					reuse=reuse
					)
				print("Z_horizontal", Z_horizontal)
				Z_horizontal = tf.concat([tf.expand_dims(Z0_tiled, axis=2), Z_horizontal], axis=2)
				print("Z_horizontal", Z_horizontal)
				if self.par.divide_nn_by_dimension:
					Z_horizontal /= self.par.d
			print("X_normalized[:, :, :-1].shape", X_normalized[:, :, :-1].shape)
			print("Z_horizontal", Z_horizontal)
		
		else:
			print(">>> Network is vertical.")
			if self.par.network.handles_starting_gradient():
				print("Network handling initial gradient.")

				Z = self.par.network.get_vertical_network(
						X_normalized[:, :, 0], 
						str(0), 
						training_flag,
						a_timestep=(self.par.tstamps[1:-1] - self.t_normalizing_mean) / self.t_normalizing_std,
						reuse=reuse)
				if self.par.divide_nn_by_dimension:
					Z /= self.par.d
				Z_list = [Z]
			else:
				print("Network not handling initial gradient. Creating Z0")
				Z0 = tf.get_variable("Z0", None, tf.float32, tf.random_uniform([1, self.par.d], minval=-.05, maxval=.05, dtype=tf.float32, seed=2))
				if self.par.divide_nn_by_dimension:
					Z0 = Z0 / self.par.d
				Z = tf.tile(Z0, [sample_size, 1])
				Z_list = [Z]

		Y_list = [Y]
		
		# Integration scheme
		for i in range(0, self.par.n_timesteps):
			
			t = self.par.tstamps[i]
			delta_t = self.par.tstamps[i+1] - self.par.tstamps[i]
			delta_t = self.par.tstamps[i+1] - self.par.tstamps[i]
			
			X = X_in[:, :, i]
			dB = dB_in[:, :, i]
			
			# If horizontal, get the right Z
			if not self.par.network.is_vertical():
				Z = Z_horizontal[:, :, i]
			
			# Compute Y
			try:
				Y = Y - self.par.pde.f_tf_grad(t, X, Y, Z) * delta_t + tf.reduce_sum(tf.multiply(self.par.pde.vol_tf(t, X, Z), dB), axis=1, keep_dims=True)
			except:
				Y = Y - self.par.pde.f_tf(t, X, Y, self.par.pde.vol_tf(t, X, Z)) * delta_t + tf.reduce_sum(tf.multiply(self.par.pde.vol_tf(t, X, Z), dB), axis=1, keep_dims=True)

			Y_list.append(Y)
			
			if self.par.network.is_vertical() and i < self.par.n_timesteps-1:
				# Compute Z from neural network: request for a network from the network instance 
				# (it will create a new neural network or return an existing network)
                # We divide by d to initialize to small values. This rule is arbitrary
				Z = self.par.network.get_vertical_network(
						a_input=X_normalized[:, :, i+1], 
						a_namespace=str(i+1), 
						a_training=training_flag,
						a_timestep=(self.par.tstamps[i+1] - self.t_normalizing_mean) / self.t_normalizing_std, 
						reuse=reuse
						) 
				if self.par.divide_nn_by_dimension:
					Z /= self.par.d
				Z_list.append(Z)
		
		Y_final = tf.stack(Y_list, axis=2)
		
		# Reform Z if vertical
		if self.par.network.is_vertical():
			Z_final = tf.stack(Z_list, axis=2)
		else:
			Z_final = Z_horizontal
		loss = tf.losses.mean_squared_error(self.par.pde.g_tf(self.par.T, X_in[:, :, -1]), Y)

		
		if self.record_session:
			self.merged_summaries = tf.summary.merge_all()
			self.train_writer = tf.summary.FileWriter('./log/train/' + self.base_name, self.sess.graph)
			self.test_writer = tf.summary.FileWriter('./log/test/' + self.base_name, self.sess.graph)
		
		end_time = time.time()
		print("Build running time: ", end_time-start_time, "s")
		
		return Y0, loss, Y_final, Z_final
		
	##
	# @brief Create a validation sample. Alternate two steps:
	# - training on a batch of paths
	# - validating on the validation sample
	# 
	# @param self 
	# @param run_name Prefix of output files.
	# @param write_output Write output file or not.
	# @param min_decrease_rate Stop training when
	#
	# @return NumPy flexible array with (Iter, Y0, Loss and Runtime)
	def train(
		self, 
		run_name=None, 
		write_output=True, 
		min_decrease_rate=None, 
		nb_max_iter=None
		):
		
		if nb_max_iter is None:
			nb_max_iter = self.par.n_maxiter
		
		self.iter_history = []
		self.loss_history = []
		self.init_history = []
		self.init_grad_history = []
		self.runtime_history = []
		self.learning_rate_history = []
		self.training_loss_history = []
		
		# Validation sample
		start_time = time.time()
		X_valid, dB_valid = self.get_sample_paths(self.par.valid_size, self.par.X0)
		feed_dict_valid = {
			self.X_test: X_valid,
			self.dB_test: dB_valid
		}
		
		# Get trainable vars and grads
		start_time = time.time()
			
		# Sanity check: should work with reuse since no additional variable is created
		with tf.variable_scope(self.base_name, reuse=True):
			
			# Training queue: create a queue that will return training elements Xtraining, dBtraining when they are created
			self.training_queue = tf.FIFOQueue(
									capacity=40, 
									dtypes=[tf.float32, tf.float32], 
									shapes=[(self.par.batch_size, self.par.d, self.par.n_timesteps+1), (self.par.batch_size, self.par.d, self.par.n_timesteps)]
									)
			print("self.training_queue", self.training_queue)
			self.X_training, self.dB_training = self.training_queue.dequeue()

			# Build training graph
			#~ with tf.device('/gpu:0'):

			self.training_Y0, self.training_loss, _, _ = self.build(
															X_in=self.X_training, 
															dB_in=self.dB_training, 
															training_flag=True, 
															reuse=True
															)
			
			# Define learning rate (can vary between iterations)
			self.learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
			
			trainable_vars = tf.trainable_variables()
			print(len(trainable_vars), " variables")
			for i in range(min(30, len(trainable_vars))):
				print(trainable_vars[i])
			
			for i in range(len(trainable_vars)):
				if i == 0:
					self.trainable_size = tf.size(trainable_vars[i])
				else:
					self.trainable_size += tf.size(trainable_vars[i])
			

			if self.par.l2_regularizer_weight > 0:
				# Add regularizer to training loss
				regularized_variables = [v for v in tf.trainable_variables() if 'bias' not in v.name and 'beta' not in v.name and 'Y0' not in v.name and 'Z0' not in v.name]
				
				print("regularized_variables")
				for v in regularized_variables:
					print(v)
				
				self.training_loss += self.par.l2_regularizer_weight * tf.add_n([tf.nn.l2_loss(v) for v in regularized_variables])
			
			
		# This should be in AUTO mode since the gradient variables are overriden from anterior training runs
		with tf.variable_scope(self.base_name, reuse=tf.AUTO_REUSE):
			
			# Define optimizer
            # One could choose one of the following optimizers...
			#~ optimizer = tf.contrib.opt.NadamOptimizer(self.learning_rate)
			optimizer = tf.train.AdamOptimizer(self.learning_rate, name='Adam')
			#~ optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
			
			## Gradient clipping
			gradients, variables = zip(*optimizer.compute_gradients(self.training_loss, var_list=trainable_vars))
			
            # One could clip the g radient using the following line
			#~ gradients, _ = tf.clip_by_global_norm(gradients, _GRADIENT_CLIPPING_FACTOR * norm_memory)
			sgd_op = optimizer.apply_gradients(zip(gradients, variables))
			
			# For BN
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			
			self.training_ops = tf.group(*([sgd_op] + update_ops))
			
			adam_vars = [var for var in tf.global_variables() if 'Adam' in var.name]
			#~ print("adam_vars", adam_vars)
			
			end_time = time.time()
			print("Optimizer init runtime: ", end_time-start_time, "s")
		
		end_time = time.time()
		
		# Begin generating training examples
		enqueue_op = self.training_queue.enqueue(
							tf.py_func(
								self.get_sample_paths, 
								[self.par.batch_size, self.par.X0], 
								[tf.float32, tf.float32]
								)
							)
		
		self.coord = tf.train.Coordinator()
		qr = tf.train.QueueRunner(self.training_queue, [enqueue_op] * 8)
		
		enqueue_threads = qr.create_threads(self.sess, coord=self.coord, start=True)
		
		tf.train.start_queue_runners(self.sess)
		
		# Initialize variables
		with tf.variable_scope(self.base_name, reuse=True):
			self.sess.run(tf.global_variables_initializer())
		
		nb_free_param = self.sess.run(self.trainable_size)
		print("Number of free parameters:", nb_free_param)
		
		# Iteration timer
		runtime = 0.
		start_time = time.time()
		
		loss_hist_stop_criterion = []
		
		## Save trainable vars
		var_list = tf.trainable_variables() + tf.get_collection("NOT_TRAINING_VAR_TO_SAVE")
        
		print("Saving " + str(len(var_list)) + " variables")
		
		training_saver = tf.train.Saver(var_list=var_list)
		savedir = "./saver/" + type(self).__name__ + "_" + type(self.par.network).__name__ + "_" + str(time.time()) + "/"
		if not os.path.exists(savedir):
			os.makedirs(savedir)
		
		# The learning rate is given by a function defined in the parameters
		current_learning_rate = self.par.get_learning_rate(0)
		
		for i in range(nb_max_iter + 1):
			
			# Use predefined learning rate strategy
			if self.par.use_predefined_learning_rate_strategy:
				current_learning_rate = self.par.get_learning_rate(i)
			
			# Generate training data and iterate
			feed_dict_train = {
				self.learning_rate : current_learning_rate
			}

			# Every 100 steps, check test loss
			if i % self.par.n_dispstep == 0:
				
				print("----------------------------------------------------------------")
				
				end_time = time.time()
				rtime = end_time-start_time
				runtime += rtime # do not count the time spent working on the test sample
				
				if i > 0:
					print("Training took %5.3f s" % rtime)
				
				# Evaluate the loss
				start_time = time.time()
				test_loss, Y0, Z0dim0 = self.sess.run([self.test_loss, self.test_Y0, self.test_Z[0][0][0]], feed_dict=feed_dict_valid)
				end_time = time.time()
				
				rtime = end_time-start_time
				print("Evaluation on the test set took %5.3f s" % rtime)
				
				# Compute mean training loss
				if i == 0:
					training_loss = 0
					
				# Store results
				self.iter_history.append(i)
				self.loss_history.append(test_loss)
				self.init_history.append(Y0)
				self.init_grad_history.append(Z0dim0)
				self.runtime_history.append(runtime)
				self.learning_rate_history.append(current_learning_rate)
				self.training_loss_history.append(training_loss)
				to_print = "iter={:>8d}  test_loss={:>8.5E}  Y0={:>8.5E}  time={:>8.5f}s  learning_rate={:>8.5f}  Z0dim0={:>8.5f}	training_loss={:>8.5E}".format(i, test_loss, Y0, runtime, current_learning_rate, Z0dim0, training_loss)
				
				loss_hist_stop_criterion.append(test_loss)
				
                # If the test loss was inferior to the last best test loss, save the parameters used
				if i == 0 or test_loss < self.best_loss:
					start_time = time.time()
					save_path = training_saver.save(self.sess, savedir + "model.ckpt")
					end_time = time.time()
					rtime = end_time-start_time
					print("Model saved in %5.3f s" % rtime)
					
					to_print += " model saved"
					self.best_loss = test_loss
					
					
				self.last_loss = test_loss
				
				print(to_print)

				# Every 2000 steps, check test loss
				if not self.par.use_predefined_learning_rate_strategy and min_decrease_rate is not None and i % self.par.check_step_count == 0:
					mean_loss_from_last_check = np.mean(loss_hist_stop_criterion)
					if i != 0:
						# Check if the loss has decreased more than min_decrease_rate in proportion
						decrease_rate = (self.last_loss_check - mean_loss_from_last_check) / self.last_loss_check 
						print("Loss decrease_rate was: ", decrease_rate)
						if decrease_rate < min_decrease_rate:
							if i >= self.par.n_miniter:
								break
							current_learning_rate = np.maximum(1e-6, current_learning_rate / 2.)
                            # One could reinitialize the variables of the optimizer using this:
							#~ self.sess.run(tf.variables_initializer(adam_vars))
					self.last_loss_check = mean_loss_from_last_check
					loss_hist_stop_criterion = []
				
				start_time = time.time()
			
			self.sess.run([self.training_ops], feed_dict=feed_dict_train)
			if i % (self.par.n_dispstep + 1) == 0:
				training_loss = self.sess.run(self.training_loss, feed_dict=feed_dict_train)
		
		# Restore the best model
		training_saver.restore(self.sess, savedir + "model.ckpt")
		print("Restored best variables")
		
		# Remove saving folder
		shutil.rmtree(savedir)
		
		# Request thread stop
		self.coord.request_stop()
		
		# Get best loss index
		best_loss_index = np.argmin(self.loss_history)
		self.best_loss = self.loss_history[best_loss_index]
		self.best_Y0 = self.init_history[best_loss_index]
		self.best_Z0 = self.init_grad_history[best_loss_index]
		self.best_iter = self.iter_history[best_loss_index]
		print("Best loss:", self.best_loss, ", best Y0:", self.best_Y0, ", best Z0:", self.best_Z0, ", iter:", self.best_iter)
		
		base_name = self.output_directory 
		if base_name[-1] != "/":
			base_name += "/"
		np.savetxt(
			base_name + "_best_stats.csv",
			[self.best_loss, self.best_Y0, self.best_Z0, self.best_iter, nb_free_param],
			header="best_loss, best_Y0, best_Z0, best_iter, nb_free_parameters"
		)
		
		# Get path & dB
		Y_path_tmp, Z_path_tmp = self.sess.run([self.test_Y, self.test_Z], feed_dict=feed_dict_valid)

		Y_path = np.zeros((1, self.par.n_timesteps+1), dtype=np.float32)
		Z_path = np.zeros((self.par.d, self.par.n_timesteps), dtype=np.float32)
		for t in range(self.par.n_timesteps):
			Y_path[:, t] = np.mean(Y_path_tmp[:, :, t], 0)
			Z_path[:, t] = np.mean(Z_path_tmp[:, :, t], 0) 
		Y_path[:, self.par.n_timesteps] = np.mean(Y_path_tmp[:, :, self.par.n_timesteps], 0)

		print("mean_Y=", Y_path)		

		
		# Save results to file
		output_array = np.zeros(len(self.iter_history), dtype=[('Iter', int), ('Loss', float), ('Y0', float), ('Z0', float), ('Runtime', float), ('Learning_rate', float), ('Training_loss', float)])
		output_array["Iter"] = self.iter_history
		output_array["Loss"] = self.loss_history
		output_array["Y0"] = self.init_history
		output_array["Z0"] = self.init_grad_history
		output_array["Runtime"] = self.runtime_history
		output_array["Learning_rate"] = self.learning_rate_history
		output_array["Training_loss"] = self.training_loss_history
		
		if write_output:
			# Get current time as the base name
			base_name = self.output_directory + "/"
			if run_name is not None:
				base_name += run_name
			
			np.savetxt(
					base_name + "_output_array.csv", 
					output_array, 
					fmt="%8d, %8.5E, %8.5E, %8.5E, %8.5f, %8.5f, %8.5E",
					header="Iter, Loss, Y0, Z0, Runtime, Learning_rate, Training_loss"
					)
			# Inspect realizations, plot and save
			self.solution = self.inspect_realizations(suffix_name=run_name)
			
			# Plot loss
			self.plot_and_save_univariate_variable_details(output_array["Iter"], np.expand_dims(output_array["Loss"], axis=0), "Loss", "orange", run_name, xlabel="Iteration", logy=True)
			
			# Plot comp time
			self.plot_and_save_univariate_variable_details(output_array["Iter"], np.expand_dims(output_array["Runtime"], axis=0), "Runtime", "pink", run_name, xlabel="Iteration")
			
			# Plot Y0
			self.plot_and_save_univariate_variable_details(output_array["Iter"], np.expand_dims(output_array["Y0"], axis=0), "Y0", "pink", run_name, xlabel="Iteration")
			
		if self.save_session:
			# Save model
			self.save_model_to_path(base_name + "model.ckpt", global_step=i)
		
		self.output_array = output_array
		
		return output_array
	
	##
	# @brief Saves model (variables) to path
	#
	# @param self
	# @param model_output_path The output path
	# @param global_step Iteration step count
	def save_model_to_path(
		self, 
		model_output_path, 
		global_step=None
		):
		save_path = self.saver.save(self.sess, model_output_path, global_step=global_step)	
	
	##
	# @brief Restores model (variables) from path
	#
	# @param self
	# @param restore_model_path The input path
	def restore_model_from_file(
		self, 
		restore_model_path
		):
		print("Restoring model at path: ", restore_model_path)
		start_time = time.time()
		self.saver.restore(self.sess, restore_model_path)
		end_time = time.time()
		print("Restore completed in ", end_time-start_time, "s")
	
	##
	# @brief Generate sample path X, dB from pde
	# 
	# @param self
	# @param n_path_count Number of paths to be generated
	# @param X0 Initial condition
	#
	# @return X Generated path (NumPy array, shape (nb_realizations, dimension, timesteps))
	# @return dB Generated Brownian (NumPy array, shape (nb_realizations, dimension, timesteps))
	def get_sample_paths(
		self, 
		n_path_count, 
		X0,
		seed=None
		):
		return self.par.pde.generate_paths(X0, self.par.delta_t, n_path_count, self.par.n_timesteps, seed=seed)
	
	##
	# @brief Compute Y, Z from the given path using the parameters learnt
	#
	# @param self
	# @param X Input path
	# @param dB Input Brownian
	#
	# @return Y Generated Y, shape (nb_realizations, dimension, timesteps)
	# @return Z Generated Z, shape (nb_realizations, dimension, timesteps)
	def run_simulation_from_path(
		self, 
		X, 
		dB
		):
		
		# Number of realizations
		nb_realizations = X.shape[0]
		
		# Run simulations
		#~ tmp_Y, tmp_Z = self.sess.run([self.test_Y, self.test_Z], 
		Y_path, Z_path, loss = self.sess.run([self.test_Y, self.test_Z, self.test_loss], 
			feed_dict={
				self.X_test: X,
				self.dB_test: dB
			})
		
		return Y_path, Z_path, loss
	
	##
	# @brief Save and plot realization details for a given model
	#
	# @param self
	# @param suffix_name Will be used as a suffix in the output directory.
	#
	# @return output_means A NumPy array containing t and the means of X, Y, Z.
	def inspect_realizations(
		self, 
		suffix_name=None,
		seed=1
		):
		
		nb_realizations = 1500
		
		# Compute new realizations
		X, dB = self.get_sample_paths(nb_realizations, self.par.X0, seed=seed)
		
		Y_path, Z_path, test_loss = self.run_simulation_from_path(X, dB)
		
		# Final test loss
		np.savetxt(
			self.output_directory + "/final_test_loss.csv", 
			np.array([test_loss]),
			header="test_loss" 
			)
		
		
		Xplot = X[:, :, :]
		
		# Plot X
		X_data = self.plot_and_save_univariate_variable_details(self.par.tstamps, Xplot[:, 0, :], "X (dim 0)", "blue", suffix_name)
		
		# Plot Y
		Y_data = self.plot_and_save_univariate_variable_details(self.par.tstamps, Y_path[:, 0, :], "Y", "red", suffix_name)
		
		# Plot Z
		Z_data = self.plot_and_save_univariate_variable_details(self.par.tstamps, Z_path[:, 0, :], "Z", "green", suffix_name)
		
		# Plot Y - g(X)
		diffYG = Y_path - self.par.pde.g_np(self.par.tstamps, Xplot)
		diffYG_data = self.plot_and_save_univariate_variable_details(self.par.tstamps, diffYG[:, 0, :], "Y - g(X)", "purple", suffix_name)
		
		# Plot histogram of Y_T - g(X_T)
		fig = plt.figure()
		plt.hist(diffYG[:, 0, -1], bins=100, histtype="stepfilled")
		plt.xlabel("Y_T - g(X_T)")
		plt.title("Distribution of Y_T - g(X_T) (" + str(nb_realizations) + "samples)")
		# Save figure
		fig.savefig(self.output_directory + "/" + "hist_YmGX" + "_plot.pdf")
		if run_from_ipython():
			plt.show()
		
		# Plot trajectories
		fig = plt.figure()
		for i in range(min(diffYG.shape[0], 10)):
			plt.plot(self.par.tstamps, diffYG[i, 0, :])
		plt.xlabel("t")
		plt.ylabel("Y_t - g(X_t)")
		plt.title("Sample trajectories")
		
		# If exact solution is known, get the L1 and L2 norm + sample solutions
		if self.par.compute_reference and (issubclass(type(self.par.pde), SemilinearPDE_ReferenceSolution) or issubclass(type(self.par.pde), SemilinearPDE_MCSolution)):
			
			print("Plotting sample...")
			start_time = time.time()
			
			if issubclass(type(self.par.pde), SemilinearPDE_ReferenceSolution):
				# Get exact solution
				Ysol = np.zeros((nb_realizations, 1, self.par.n_timesteps+1))
				Zsol = np.zeros((nb_realizations, self.par.d, self.par.n_timesteps))
				
				n_threads=_MULTIPROCESSING_CORE_COUNT
				print("n_threads", n_threads)
				
				def computesol(i):
					t = self.par.tstamps[i]
					return self.par.pde.reference_solution(t, Xplot[:, :, i])
				
				for i in range(self.par.n_timesteps+1):
					Ysol[:, :, i] = computesol(i) #res[i]
				
				def computegrad(i):
					t = self.par.tstamps[i]
					return self.par.pde.reference_gradient(t, Xplot[:, :, i])
				
				for i in range(self.par.n_timesteps):
					Zsol[:, :, i] = computegrad(i) #res[i]
			
			if issubclass(type(self.par.pde), SemilinearPDE_MCSolution):
				
				if not self.par.override_mc:
					Ysol, Zsol = self.par.pde.get_solution_from_file(self.par.d, self.par.n_timesteps, Xplot)
				else:
					Ysol, Zsol = self.par.pde.get_solution_from_file(self.par.d, self.par.n_timesteps, Xplot, override_mc_size=self.par.override_mc_size, override_n_threads=self.par.override_n_threads)
			
			# Get the errors
			Y0err = Y_path[0, 0, 0]-Ysol[0, 0, 0]
			
			YL1err =  np.abs(Ysol-Y_path)
			int_YL1err = self.par.delta_t * (np.sum(YL1err[:, :, 1:-1], axis=2) + (YL1err[:, :, 1] + YL1err[:, :, -1]) / 2)
			quantile_5_int_YL1err = np.percentile(int_YL1err, 5, axis=0).reshape(-1)
			quantile_50_int_YL1err = np.percentile(int_YL1err, 50, axis=0).reshape(-1)
			quantile_95_int_YL1err = np.percentile(int_YL1err, 95, axis=0).reshape(-1)
			mean_int_YL1err = np.mean(int_YL1err)
			
			ZL2err = np.power(np.linalg.norm(Zsol-Z_path, axis=1, ord=2, keepdims=True), 2)
			int_ZL2err = self.par.delta_t * (np.sum(ZL2err[:, :, 1:-1], axis=2) + (ZL2err[:, :, 1] + ZL2err[:, :, -1]) / 2)
			quantile_5_int_ZL2err = np.percentile(int_ZL2err, 5, axis=0).reshape(-1)
			quantile_50_int_ZL2err = np.percentile(int_ZL2err, 50, axis=0).reshape(-1)
			quantile_95_int_ZL2err = np.percentile(int_ZL2err, 95, axis=0).reshape(-1)
			mean_int_ZL2err = np.mean(int_ZL2err)
			
			ZL2normsol = np.power(np.linalg.norm(Zsol, axis=1, ord=2, keepdims=True), 2)
			
			print("Y0err", Y0err)
			print("mean_int_YL1err", mean_int_YL1err)
			print("mean_int_ZL2err", mean_int_ZL2err)
			
			# Compute the Y0 if f==0
			print("Y0_f", Ysol[0, 0, 0])
			Y0_fnull = np.mean(self.par.pde.g_np(self.par.T, Xplot[:, :, -1]), axis=0)[0]
			print("Y0_fnull", Y0_fnull)
			
			# Compute the error on Z0_0
			Z00err = Z_path[0, 0, 0]-Zsol[0, 0, 0]
			
			# Compute the L2 error on Z0
			Z0sol_l2_norm = np.linalg.norm(Zsol[0, :, 0], ord=2)
			Z0_l2_err_norm = np.linalg.norm(Z_path[0, :, 0] - Zsol[0, :, 0], ord=2)
			
			np.savetxt(
				self.output_directory + "/losses.csv", 
				np.array([
						Y0err, 
						mean_int_YL1err, 
						quantile_5_int_YL1err, 
						quantile_50_int_YL1err, 
						quantile_95_int_YL1err, 
						mean_int_ZL2err, 
						quantile_5_int_ZL2err, 
						quantile_50_int_ZL2err, 
						quantile_95_int_ZL2err,
						Z00err,
						Ysol[0, 0, 0],
						Zsol[0, 0, 0],
						Z0_l2_err_norm,
						Z0sol_l2_norm
						]),
				header="Y0err, mean_int_YL1err, quantile_5_int_YL1err, quantile_50_int_YL1err, quantile_95_int_YL1err, mean_int_ZL2err, quantile_5_int_ZL2err, quantile_50_int_ZL2err, quantile_95_int_ZL2err, Z00err, Y0ref, Z00ref, Z0_l2_err_norm, Z0sol_l2_norm" 
				)
			
			print("Saving Z0")
			tmp = np.stack((Z_path[0, :, 0], Zsol[0, :, 0]), axis=0)
			print(tmp)
			np.savetxt(
				self.output_directory + "/Z0_ref_sol.csv",
				tmp
				)
			
			# Plot L1 error
			self.plot_and_save_univariate_variable_details(self.par.tstamps, YL1err[:, 0, :], "L1_error_Y", "red", "L1_error_Y", "Time", False, plot_std=True)
			
			# Plot relative error
			self.plot_and_save_univariate_variable_details(self.par.tstamps, YL1err[:, 0, :] / (np.abs(Ysol[:, 0, :] + 1e-8)), "Relative_error_Y", "red", "Relative_error_Y", "Time", False, plot_std=True)
			
			# Plot L2 error
			self.plot_and_save_univariate_variable_details(self.par.tstamps, ZL2err[:, 0, :], "L2_error_Z", "purple", "L2_error_Z", "Time", False, plot_std=True)
			
			# Plot relative error
			self.plot_and_save_univariate_variable_details(self.par.tstamps, ZL2err[:, 0, :] / (ZL2normsol[:, 0, :] + 1e-8), "Relative_error_Z", "purple", "Relative_error_Z", "Time", False, plot_std=True)

			
			# Plot some trajectories
			for i in range(5):
				fig, ax = plt.subplots()
				yline_dbsde, = ax.plot(self.par.tstamps, Y_path[i, 0, :], color="blue")
				yline_sol, = ax.plot(self.par.tstamps, Ysol[i, 0, :], color="purple")
				
				ax2 = ax.twinx()
				zline_dbsde, = ax2.plot(self.par.tstamps[:-1], Z_path[i, 0, :], color="red")
				zline_sol, = ax2.plot(self.par.tstamps[:-1], Zsol[i, 0, :], color="orange")
				
				ax.legend([yline_dbsde, yline_sol, zline_dbsde, zline_sol], ["Y DBSDE", "Y Exact solution", "Z (dim 0) DBSDE", "Z (dim 0) Exact solution"])
				ax.set_title("Plotting Y and Z (" + str(i) + "th sample)")
				ax.set_xlabel("Time")
				ax.set_ylabel("Y")
				ax2.set_ylabel("Z (dim 0)")
				fig.savefig(self.output_directory + "/" + "sample_traj_" + str(i) + "_plot.pdf")
				
				traj = np.zeros((self.par.n_timesteps+1, 3))
				traj[:, 0] = self.par.tstamps
				traj[:, 1] = Y_path[i, 0, :]
				traj[:, 2] = Ysol[i, 0, :]
				np.savetxt(
					self.output_directory + "/Y_sample_traj_" + str(i) + ".csv", 
					np.array(traj),
					header="Time, Ypath, Ysol" 
					)
				
				if run_from_ipython():
					plt.show()
				
				traj = np.zeros((self.par.n_timesteps, 3))
				traj[:, 0] = self.par.tstamps[:-1]
				traj[:, 1] = Z_path[i, 0, :]
				traj[:, 2] = Zsol[i, 0, :]
				np.savetxt(
					self.output_directory + "/Z_sample_traj_" + str(i) + ".csv", 
					np.array(traj),
					header="Time, Zpathdim0, Zsoldim0" 
					)
				
				traj = np.zeros((self.par.n_timesteps+1, 2))
				traj[:, 0] = self.par.tstamps
				traj[:, 1] = X[i, 0, :]
				np.savetxt(
					self.output_directory + "/X_sample_traj_" + str(i) + ".csv", 
					np.array(traj),
					header="Time, Xdim0" 
					)
				
				end_time = time.time()
				print("Done computing sample, elapsed time " + str(end_time-start_time) + "s")
		
		else:
			# plot some trajectories without reference
			# Plot some trajectories
			for i in range(5):
				fig, ax = plt.subplots()
				yline_dbsde, = ax.plot(self.par.tstamps, Y_path[i, 0, :], color="blue")
				
				ax2 = ax.twinx()
				zline_dbsde, = ax2.plot(self.par.tstamps[:-1], Z_path[i, 0, :], color="red")
				
				ax.legend([yline_dbsde, zline_dbsde], ["Y DBSDE", "Z (dim 0) DBSDE"])
				ax.set_title("Plotting Y and Z (" + str(i) + "th sample)")
				ax.set_xlabel("Time")
				ax.set_ylabel("Y")
				ax2.set_ylabel("Z (dim 0)")
				fig.savefig(self.output_directory + "/" + "sample_traj_" + str(i) + "_plot.pdf")
				
				traj = np.zeros((self.par.n_timesteps+1, 2))
				traj[:, 0] = self.par.tstamps
				traj[:, 1] = Y_path[i, 0, :]
				np.savetxt(
					self.output_directory + "/Y_sample_traj_" + str(i) + ".csv", 
					np.array(traj),
					header="Time, Ypath" 
					)
				
				if run_from_ipython():
					plt.show()
				
				traj = np.zeros((self.par.n_timesteps, 2))
				traj[:, 0] = self.par.tstamps[:-1]
				traj[:, 1] = Z_path[i, 0, :]
				np.savetxt(
					self.output_directory + "/Z_sample_traj_" + str(i) + ".csv", 
					np.array(traj),
					header="Time, Zpathdim0" 
					)
				
				traj = np.zeros((self.par.n_timesteps+1, 2))
				traj[:, 0] = self.par.tstamps
				traj[:, 1] = X[i, 0, :]
				np.savetxt(
					self.output_directory + "/X_sample_traj_" + str(i) + ".csv", 
					np.array(traj),
					header="Time, Xdim0" 
					)
			
		# Get means
		output_means = np.zeros((self.par.tstamps.size, 4), np.float32)
		output_means[:, 0] = self.par.tstamps
		output_means[:, 1] = X_data[:, 1]
		output_means[:, 2] = Y_data[:, 1]
		output_means[:self.par.tstamps.size-1, 3] = Z_data[:, 1]
		
		return output_means
		
	##
	# @brief Save and plot details for given variable
	#
	# @param x Time. Should be np.array((time_dim,)).
	# @param variable Univariate variable of interest. Should be np.array((realization_dim, time_dim)).
	# @param name Name of the variable (for plot) - string.
	# @param color Color of the plot (e.g. "blue",...).
	# @param xlabel Label of the x axis.
	# @param logy Should y be log-scaled?
	#
	# @return Array containing (time, means, 5-quantile, 50-quantile, 95-quantile) for each timestep
	def plot_and_save_univariate_variable_details(
		self, 
		x, 
		variable, 
		name, 
		color, 
		suffix_name=None, 
		xlabel="Time", 
		logy=False,
		plot_std=False
		):
		
		# Compute means
		variable_means = np.mean(variable, axis=0).reshape(-1)
		
		# Compute quantiles
		variable_5_quantile = np.percentile(variable, 5, axis=0).reshape(-1)
		variable_95_quantile = np.percentile(variable, 95, axis=0).reshape(-1)
		variable_50_quantile = np.percentile(variable, 50, axis=0).reshape(-1)
		
		# Compute std
		variable_standard_deviation = np.std(variable, axis=0).reshape(-1)
		
		# Get right dimensions for x
		x = x[:variable_5_quantile.size]
		
		## Plot
		fig, ax = plt.subplots()
		
		# Plot means
		means_line, = ax.plot(x, variable_means, color=color)
		
		if plot_std:
			# Plot stds
			std_up, = ax.plot(x, variable_means + 2 * variable_standard_deviation, color=color, ls="dashed")
			#~ std_down, = ax.plot(x, variable_means - 2 * variable_standard_deviation, color=color, ls="dashed")
		
		# Fill 5 -> 95 quantiles
		quantile_area = ax.fill_between(x, variable_5_quantile, variable_95_quantile, where=variable_95_quantile >= variable_5_quantile, facecolor=color, alpha=0.3, interpolate=True)
		
		ax.set_xlabel(xlabel)
		ax.set_ylabel(name)
		if logy:
			ax.set_yscale("log", nonposy='clip')
		
		if plot_std:
			ax.legend([means_line, std_up, quantile_area], ["Mean", "Mean + 2 * std", "5 to 95% quantiles"])
		else:
			ax.legend([means_line, quantile_area], ["Mean", "5 to 95% quantiles"])
		
		# Save data
		if plot_std:
			output_data = np.zeros((*variable_5_quantile.shape, 6), np.float32)
			output_data[:, 0] = x
			output_data[:, 1] = variable_means
			output_data[:, 2] = variable_5_quantile
			output_data[:, 3] = variable_50_quantile
			output_data[:, 4] = variable_95_quantile
			output_data[:, 5] = variable_standard_deviation
			
			output_name = self.output_directory + "/" + name
			if suffix_name is not None:
				output_name += "_" + suffix_name
			
			np.savetxt(
				output_name + "_plot_data.csv", 
				output_data, 
				fmt="%8.5f, %8.5f, %8.5f, %8.5f, %8.5f, %8.5f",
				header="x, Means, 5-quantile, 50-quantile, 95-quantile, std"
				)
		else:
			# Save data
			output_data = np.zeros((*variable_5_quantile.shape, 5), np.float32)
			output_data[:, 0] = x
			output_data[:, 1] = variable_means
			output_data[:, 2] = variable_5_quantile
			output_data[:, 3] = variable_50_quantile
			output_data[:, 4] = variable_95_quantile
			
			output_name = self.output_directory + "/" + name
			if suffix_name is not None:
				output_name += "_" + suffix_name
			
			np.savetxt(
				output_name + "_plot_data.csv", 
				output_data, 
				fmt="%8.5f, %8.5f, %8.5f, %8.5f, %8.5f",
				header="x, Means, 5-quantile, 50-quantile, 95-quantile"
				)
		
		# Save figure
		fig.savefig(output_name + "_plot.pdf")
		
		if run_from_ipython():
			plt.show()
		
		plt.close('all')
		
		return output_data
	
	##
	# @brief Closes the session and resets the graph.
	#
	def close(self):
		tf.reset_default_graph()
		self.sess.close()



############################################################################################################	
############################################################################################################	
############################################################################################################


class SimpleSolver_Recurrent_Y_gX(SimpleSolver):
	
	##
	# @brief Create a graph taking inputs and returning the path and the loss. 
	# 
	# @param self 
	# @param X_in Input data tensor.
	# @param dB_in Brownian data tensor.
	# @param training_flag Should be True for the training graph, False for the test graph.
	# @param reuse Should be True if no variable should be created, False else, or tf.AUTO_REUSE.
	#
	# @return Y0 The initial Y tensor.
	# @return loss The loss tensor.
	# @return Y_list A list of tensors depicting Y at each timestep.
	# @return Z_list A list of tensors depicting Z at each timestep.
	def build(
		self, 
		X_in, 
		dB_in, 
		training_flag, 
		reuse
		):
			
		start_time = time.time()
		
		# Sample size
		sample_size = tf.shape(X_in)[0]
		
		# Variables
		
		# Y0
		Y0 = tf.get_variable("Y0", [], tf.float32, self.Y0_initializer)
		Y = tf.tile(tf.reshape(Y0, [1, 1]), [sample_size, 1])
		
		if self.par.normalize_input_X:
			print("Input X of the neural networks will be normalized by the statically computed mean and variance")
			unnormalized_input = X_in
			X_normalized = (unnormalized_input - self.initial_normalization_means) / np.sqrt(self.initial_normalization_vars + 1e-6)
		else:
			print("Input X of the neural networks will NOT be normalized by the statically computed mean and variance")
			X_normalized = X_in
		
		
		## Create Z from X
		## for horizontal networks (construction layer by layer)
		if not self.par.network.is_vertical():
			print(">>> Network is horizontal.")
			
			raise Exception("This solver " + self.__class__.__name__ + " does not support horizontal network + " + self.par.network.__class__.__name__)
		
		else:
			print(">>> Network is vertical.")
			if self.par.network.handles_starting_gradient():
				print("Network handling initial gradient.")

				Z = self.par.network.get_vertical_network(
						tf.concat([X_normalized[:, :, 0], (Y - self.Y_normalizing_mean) / self.Y_normalizing_std, (self.par.pde.g_tf(0, X_in[:, :, 0]) - self.Y_normalizing_mean) / self.Y_normalizing_std], axis=1), 
						str(0), 
						training_flag,
						a_timestep=(0 - self.t_normalizing_mean) / self.t_normalizing_std,
						reuse=reuse)
				if self.par.divide_nn_by_dimension:
					Z /= self.par.d
				Z_list = [Z]
			else:
				print("Network not handling initial gradient. Creating Z0")
				Z0 = tf.get_variable("Z0", None, tf.float32, tf.random_uniform([1, self.par.d], minval=-.05, maxval=.05, dtype=tf.float32, seed=2))
				if self.par.divide_nn_by_dimension:
					Z0 = Z0 / self.par.d
				Z = tf.tile(Z0, [sample_size, 1])
				Z_list = [Z]

		Y_list = [Y]
		
		# Integration scheme
		for i in range(0, self.par.n_timesteps):
			
			t = self.par.tstamps[i]
			delta_t = self.par.tstamps[i+1] - self.par.tstamps[i]
			
			X = X_in[:, :, i]
			dB = dB_in[:, :, i]
			
			# If horizontal, get the right Z
			if not self.par.network.is_vertical():
				Z = Z_horizontal[:, :, i]
			
			# Compute Y
			try:
				Y = Y - self.par.pde.f_tf_grad(t, X, Y, Z) * delta_t + tf.reduce_sum(tf.multiply(self.par.pde.vol_tf(t, X, Z), dB), axis=1, keep_dims=True)
			except:
				Y = Y - self.par.pde.f_tf(t, X, Y, self.par.pde.vol_tf(t, X, Z)) * delta_t + tf.reduce_sum(tf.multiply(self.par.pde.vol_tf(t, X, Z), dB), axis=1, keep_dims=True)
			
			Y_list.append(Y)
			
			if self.par.network.is_vertical() and i < self.par.n_timesteps-1:
				
				# Compute Z from neural network: request for a network from the network instance 
				# (it will create a new neural network or return an existing network)
				Z = self.par.network.get_vertical_network(
						a_input=tf.concat([X_normalized[:, :, i+1], (Y - self.Y_normalizing_mean) / self.Y_normalizing_std, (self.par.pde.g_tf(self.par.tstamps[i+1], X_in[:, :, i+1]) - self.Y_normalizing_mean) / self.Y_normalizing_std], axis=1), 
						a_namespace=str(i+1), 
						a_training=training_flag,
						a_timestep=(self.par.tstamps[i+1] - self.t_normalizing_mean) / self.t_normalizing_std, 
						reuse=reuse
						) 
				if self.par.divide_nn_by_dimension:
					Z /= self.par.d
				Z_list.append(Z)
		
		Y_final = tf.stack(Y_list, axis=2)
		
		# Reform Z if vertical
		if self.par.network.is_vertical():
			Z_final = tf.stack(Z_list, axis=2)
		else:
			Z_final = Z_horizontal
		
		loss = tf.losses.mean_squared_error(self.par.pde.g_tf(self.par.T, X_in[:, :, -1]), Y)
		
		if self.record_session:
			self.merged_summaries = tf.summary.merge_all()
			self.train_writer = tf.summary.FileWriter('./log/train/' + self.base_name, self.sess.graph)
			self.test_writer = tf.summary.FileWriter('./log/test/' + self.base_name, self.sess.graph)
		
		end_time = time.time()
		print("Build running time: ", end_time-start_time, "s")
		
		return Y0, loss, Y_final, Z_final


############################################################################################################	
############################################################################################################	
############################################################################################################


##
# @brief This solver only works with pdes for which an exact solution is known.
# It uses Z:= exact Z so that the only parameter to be optimized is Y_0.
# It is used for sanity check..
class ExactSolver(SimpleSolver):
	
	##
	# @brief Create a graph taking inputs and returning the path and the loss. 
	# 
	# @param self 
	# @param X_in Input data tensor.
	# @param dB_in Brownian data tensor.
	# @param training_flag Should be True for the training graph, False for the test graph.
	# @param reuse Should be True if no variable should be created, False else, or tf.AUTO_REUSE.
	#
	# @return Y0 The initial Y tensor.
	# @return loss The loss tensor.
	# @return Y_list A list of tensors depicting Y at each timestep.
	# @return Z_list A list of tensors depicting Z at each timestep.
	def build(
		self, 
		X_in, 
		dB_in, 
		training_flag, 
		reuse
		):
			
		start_time = time.time()
		
		# Sample size
		sample_size = tf.shape(X_in)[0]
		
		# Variables
		
		# Y0
		Y0 = tf.get_variable("Y0", [], tf.float32, self.Y0_initializer)
		Y = tf.tile(tf.reshape(Y0, [1, 1]), [sample_size, 1])
		
		## Create Z from X
		print(">>> Using exact solution.")
		Z = self.par.pde.reference_gradient_tf(
				t = self.par.tstamps[0],
				X = X_in[:, :, 0])
		Z_list = [Z]

		Y_list = [Y]
		
		# Integration scheme
		for i in range(0, self.par.n_timesteps):
			
			t = self.par.tstamps[i]
			delta_t = self.par.tstamps[i+1] - self.par.tstamps[i]
			delta_t = self.par.tstamps[i+1] - self.par.tstamps[i]
			
			X = X_in[:, :, i]
			dB = dB_in[:, :, i]
			
			# Compute Y
			try:
				Y = Y - self.par.pde.f_tf_grad(t, X, Y, Z) * delta_t + tf.reduce_sum(tf.multiply(self.par.pde.vol_tf(t, X, Z), dB), axis=1, keep_dims=True)
			except:
				Y = Y - self.par.pde.f_tf(t, X, Y, self.par.pde.vol_tf(t, X, Z)) * delta_t + tf.reduce_sum(tf.multiply(self.par.pde.vol_tf(t, X, Z), dB), axis=1, keep_dims=True)

			Y_list.append(Y)
			
			if i < self.par.n_timesteps-1:
				Z = self.par.pde.reference_gradient_tf(
					t = self.par.tstamps[i+1],
					X = X_in[:, :, i+1])
				Z_list.append(Z)
		
		Y_final = tf.stack(Y_list, axis=2)
		
		Z_final = tf.stack(Z_list, axis=2)
		
		loss = tf.losses.mean_squared_error(self.par.pde.g_tf(self.par.T, X_in[:, :, -1]), Y)
		
		if self.record_session:
			self.merged_summaries = tf.summary.merge_all()
			self.train_writer = tf.summary.FileWriter('./log/train/' + self.base_name, self.sess.graph)
			self.test_writer = tf.summary.FileWriter('./log/test/' + self.base_name, self.sess.graph)
		
		end_time = time.time()
		print("Build running time: ", end_time-start_time, "s")
		
		return Y0, loss, Y_final, Z_final
