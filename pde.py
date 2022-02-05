
import abc

import numpy as np
import tensorflow as tf
import scipy.integrate as integrate

import multiprocessing
from joblib import Parallel, delayed

import hashlib

# The following two arguments are used when computing MC solutions
_MULTIPROCESSING_CORE_COUNT = 2
_NUM_MC_RANDOM_SAMPLES = 10000


##
# @brief Abstract base class for a semilinear PDE.
#
# The semilinear PDE should be defined by
# u_t + mu * grad(u) + 1/2 * Tr(sigma * sigma^T * Hess(u)) + f = 0
# with final condition u(T, x) = g(x), where diff_np is mu, vol_np is
# sigma, f_tf is f, g_tf is g.
#
# IMPORTANT: in the following, the function f_tf take Z as argument,
# where Z := sigma^T Grad u. The function f_tf_grad take as argument grad,
# which is Grad u.
# If provided, the algorithm will use f_tf_grad. If the function is not defined,
# it will use f_tf: make sure your function definitions are coherent. 
class SemilinearPDE(abc.ABC):
	
	@abc.abstractmethod
	##
	# @brief Defines the diffusion function for the PDE (NumPy format) -
	# prefer overriding this function with get_next in case an explicit
	# solution is known.
	#
	# @param self
	# @param t Current time.
	# @param X Current state.
	# @param delta_t Time interval to the next state.
	#
	# @return The diffusion function: diff * delta_t.
	def diff_np(self, t, X, delta_t):
		pass
	
	@abc.abstractmethod
	##
	# @brief Defines the volatility function for the PDE (NumPy format) -
	# prefer overriding this function with get_next in case an explicit
	# solution is known.
	#
	# @param self
	# @param t Current time.
	# @param X Current state.
	# @param delta_t Brownian motion to the next state.
	#
	# @return The diffusion function: vol.dot(delta_b).
	def vol_np(self, t, X, delta_b):
		pass
	
	@abc.abstractmethod
	##
	# @brief Defines the volatility function for the PDE (TensorFlow format) -
	# prefer overriding this function with get_next in case an explicit
	# solution is known.
	#
	# @param self
	# @param t Current time.
	# @param X Current state.
	# @param delta_t Brownian motion to the next state.
	#
	# @return The diffusion function: vol.dot(delta_b).
	def vol_tf(self, t, X, delta_b):
		pass
		
	@abc.abstractmethod
	##
	# @brief Defines the nonlinear function (tf.Tensor format).
	#
	# @param self
	# @param t Current time.
	# @param X Current state.
	# @param Y Current Y.
	# @param Z Current Z.
	#
	# @return The nonlinear function's tensor.
	def f_tf(self, t, X, Y, Z):
		pass
		
	@abc.abstractmethod
	##
	# @brief Defines the final condition function (tf.Tensor format).
	#
	# @param self
	# @param t Current time.
	# @param X Current state.
	#
	# @return The final condition function's tensor.
	def g_tf(self, t, X):
		pass
	
	@abc.abstractmethod
	##
	# @brief Defines the final condition function (NumPy array format).
	#
	# @param self
	# @param t Current time.
	# @param X Current state.
	#
	# @return The final condition function's tensor.
	def g_np(self, t, X):
		pass
	
	##
	# @brief Compute X_{t+1} from arguments (in NumPy format).
	# 
	# @param self
	# @param t Time
	# @param X X_t
	# @param delta_t {t+1} - t
	# @param delta_b B_{t+1}-B_t
	#
	# @return X_{t+1}
	def get_next(self, t, X, delta_t, delta_b):
		return X + self.diff_np(t, X, delta_t) + self.vol_np(t, X, delta_b)
	
	##
	# @brief Generate paths from initial conditions with provided integration parameters.
	# 
	# @param self
	# @param X0 Initial condition.
	# @param delta_t {t+1} - t
	# @param n_path_count Number of paths to be generated.
	# @param n_step Number of integration steps.
	# @param t0 Initial time.
	#
	# @return X Sample paths.
	# @return dB Sample Brownian motion.
	def generate_paths(self, X0, delta_t, n_path_count, n_step, t0=0, seed=None):
		n_dim = X0.size
		X = np.zeros((n_path_count, n_dim, n_step+1), dtype=np.float32)
		X[:, :, 0] = X0
		
		# Generate dB
		if seed is not None:
			np.random.seed(seed)
		dB = np.random.normal(loc=0.0, scale=np.sqrt(delta_t), size=(n_path_count, n_dim, n_step)).astype(np.float32) # Variance = delta_t

		for i in range(0, n_step):
			# Propagation using Euler
			X[:, :, i+1] = self.get_next(t0 + i * delta_t, X[:, :, i], delta_t, dB[:, :, i])
		
		return X, dB
		

############################################################################################################	
############################################################################################################	
############################################################################################################	


##
# @brief Semilinear PDE with a known reference solution
# @returns Y, Z solutions. Shape should be [batch_size, d, n_timesteps].
class SemilinearPDE_ReferenceSolution(SemilinearPDE):
	
	@abc.abstractmethod
	def reference_solution(self, t, X):
		pass
	
	@abc.abstractmethod
	def reference_gradient(self, t, X):
		pass
	
	@abc.abstractmethod
	def reference_gradient_tf(self, t, X):
		pass


############################################################################################################	
############################################################################################################	
############################################################################################################	


