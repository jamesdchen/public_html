import numpy as np
import tensorflow as tf

import abc

from tensorflow import random_normal_initializer as nor_init

from tensorflow.contrib.layers import fully_connected as fc


############################################################################################################	
############################################################################################################	
############################################################################################################	


##
# @brief Abstract template for a neural network.
#
class AbstractNetwork(abc.ABC):
	
	def is_recurrent_y_gx(self):
		return False
	
	@abc.abstractmethod
	##
	# @brief Returns whether the network should be built vertically (timestep by timestep) or horizontally (layer by layer).
	#
	# @param self
	def is_vertical(self):
		pass
	
	##
	# @brief Check whether the network also creates a network to compute Z0 from (X0, ...).
	#
	# @param self
	def handles_starting_gradient(self):
		return False
	
	##
	# @brief Gets an initializer for a layer.
	def get_initializer(self, a_in_size, a_out_size, seed=None):
		return nor_init(stddev = 1.41/np.sqrt(a_in_size + a_out_size), seed=seed)

class HorizontalNetwork(AbstractNetwork):
	
	def is_vertical(self):
		return False
	
	@abc.abstractmethod
	##
	# @brief Given an input tensor, build a network and give an output tensor.
	#
	# @param self
	# @param a_input Input tensor for the network
	# @param a_namespace Namespace for the network (e.g. timestep)
	# @param a_training Training indicator tensor. Will be True when training, else false.
	# @param a_timesteps Index of the current timestep.
	# @param reuse Flag to make the network reuse or create variables.
	#
	# @return Output tensor.
	def get_horizontal_network(
			self, 
			a_input, 
			a_namespace, 
			a_training, 
			a_timesteps, 
			reuse
			):
		pass

class VerticalNetwork(AbstractNetwork):
	
	def is_vertical(self):
		return True
	
	@abc.abstractmethod
	##
	# @brief Given an input tensor, build a network and give an output tensor.
	#
	# @param self
	# @param a_input Input tensor for the network
	# @param a_namespace Namespace for the network (e.g. timestep)
	# @param a_training Training indicator tensor. Will be True when training, else false.
	# @param a_timesteps Index of the current timestep.
	# @param reuse Flag to make the network reuse or create variables.
	#
	# @return Output tensor.
	def get_vertical_network(
			self, 
			a_input, 
			a_namespace, 
			a_training, 
			a_timestep, 
			reuse
			):
		pass



############################################################################################################	
############################################################################################################	
############################################################################################################	


##
# @brief The network returning Z:=0, used for sanity check
#
class ZeroNetwork(AbstractNetwork):
	
	def __init__(self):
		pass
	
	def is_vertical(self):
		return False
	
	def handles_starting_gradient(self):
		return True
	
	def get_horizontal_network(
			self, 
			a_input, 
			a_namespace, 
			a_training, 
			a_timesteps, 
			reuse
			):
		zero_line = tf.zeros([1, tf.shape(a_input)[1], tf.shape(a_input)[2]])
		return tf.tile(zero_line, [tf.shape(a_input)[0], 1, 1])



############################################################################################################	
############################################################################################################	
############################################################################################################



##
# @brief Get a fully connected network as in Jentzen et al but share all the learnt
# parameters (eg beta, gamma, W) between the timesteps (but not the moving means). 
#
# This is equivalent to assuming the network should learn an identical response in all 
# timesteps, i.e. that Z is function only of (mu_t, sigma_t) (batch moving means and vars). 
#
class SingleWeights_Vertical(VerticalNetwork):
	
	def handles_starting_gradient(self):
		return True
	
	def __init__(self, output_size, hidden_layer_sizes=[], activation_fn=tf.nn.elu, layer_normalization_flag=False):
		self.output_size = output_size
		self.hidden_layer_sizes = hidden_layer_sizes
		self.activation_fn = activation_fn
		self.layer_normalization_flag = layer_normalization_flag

	##
	# @brief For this network, returns layers using the shared parameters.
	def get_vertical_network(
		self, 
		a_input, 
		a_namespace, 
		a_training, 
		a_timestep, 
		reuse
		):
		
		#~ print("a_input", a_input)
		
		layer = a_input
		
		for i in range(len(self.hidden_layer_sizes)):
			layer = self.vertical_layer(
				layer,
				self.hidden_layer_sizes[i],
				activation_fn=self.activation_fn,
				reuse=reuse,
				scope="hidden_layer_" + str(i),
				layer_normalization_flag=self.layer_normalization_flag
				)
		
		output = self.vertical_layer(
				layer,
				self.output_size,
				activation_fn=None,
				reuse=reuse,
				scope="output_layer",
				layer_normalization_flag=self.layer_normalization_flag
				)
		
		return output
	
	def vertical_layer(
		self,
		inputs,
		output_size,
		activation_fn,
		reuse,
		scope,
		layer_normalization_flag
		):
		if layer_normalization_flag:
			res = fc(
				inputs,
				output_size,
				activation_fn=None,
				reuse=reuse,
				scope=scope,
				biases_initializer=None
				)
			res = tf.contrib.layers.layer_norm(
				res,
				activation_fn=activation_fn,
				reuse=reuse,
				scope=scope + "_layer_norm"
				)
		else:
			res = fc(
				inputs,
				output_size,
				activation_fn=activation_fn,
				reuse=reuse,
				scope=scope
				)
		return res


############################################################################################################	
############################################################################################################	
############################################################################################################


##
# @brief Get a fully connected network as in Jentzen et al but share all the learnt
# parameters (eg beta, gamma, W) between the timesteps (but not the moving means). 
#
# This is equivalent to assuming the network should learn an identical response in all 
# timesteps, i.e. that Z is function only of (mu_t, sigma_t) (batch moving means and vars). 
#
class SingleWeights_Vertical_ConcatTime(SingleWeights_Vertical):
	
	##
	# @brief For this network, returns layers using the shared parameters.
	#
	def get_vertical_network(
		self, 
		a_input, 
		a_namespace, 
		a_training,
		a_timestep, 
		reuse
		):
		
		sample_size = tf.shape(a_input)[0]
		tstep = tf.cast(tf.tile(tf.constant([[a_timestep]]), [sample_size, 1]), dtype=tf.float32)
		net_input = tf.concat([a_input, tstep], axis=1)
		
		output = super().get_vertical_network(
				a_input=net_input,
				a_namespace=a_namespace,
				a_training=a_training,
				a_timestep=a_timestep,
				reuse=reuse
			)
		
		return output



############################################################################################################	
############################################################################################################	
############################################################################################################



class SingleWeights_Vertical_Shortcut(SingleWeights_Vertical):
	
	##
	# @brief For this network, returns layers using the shared parameters.
	#
	def get_vertical_network(
		self, 
		a_input, 
		a_namespace, 
		a_training,
		a_timestep, 
		reuse
		):
		
		sample_size = tf.shape(a_input)[0]
		tstep = tf.cast(tf.tile(tf.constant([[a_timestep]]), [sample_size, 1]), dtype=tf.float32)
		net_input = tf.concat([a_input, tstep], axis=1)
		
		if len(self.hidden_layer_sizes) > 0:
			layer = self.vertical_layer(
				net_input,
				self.hidden_layer_sizes[0],
				activation_fn=self.activation_fn,
				reuse=reuse,
				scope="hidden_layer_0",
				layer_normalization_flag=self.layer_normalization_flag
				)
		else:
			return self.vertical_layer(
				tf.concat(net_input, axis=1),
				self.output_size,
				activation_fn=None,
				reuse=reuse,
				scope="output_layer",
				layer_normalization_flag=self.layer_normalization_flag
				)
		
		for i in range(1, len(self.hidden_layer_sizes)):
			layer = self.vertical_layer(
				tf.concat([layer, net_input], axis=1),
				self.hidden_layer_sizes[i],
				activation_fn=self.activation_fn,
				reuse=reuse,
				scope="hidden_layer_" + str(i),
				layer_normalization_flag=self.layer_normalization_flag
				)
		
		output = self.vertical_layer(
				tf.concat([layer, net_input], axis=1),
				self.output_size,
				activation_fn=None,
				reuse=reuse,
				scope="output_layer",
				layer_normalization_flag=self.layer_normalization_flag
				)
				
		return output



############################################################################################################	
############################################################################################################	
############################################################################################################



class SingleWeights_Vertical_Residual(SingleWeights_Vertical):
	
	##
	# @brief For this network, returns layers using the shared parameters.
	#
	def get_vertical_network(
		self, 
		a_input, 
		a_namespace, 
		a_training,
		a_timestep, 
		reuse
		):
		
		sample_size = tf.shape(a_input)[0]
		tstep = tf.cast(tf.tile(tf.constant([[a_timestep]]), [sample_size, 1]), dtype=tf.float32)
		net_input = tf.concat([a_input, tstep], axis=1)
		
		layer = net_input
		
		for i in range(0, len(self.hidden_layer_sizes)):
			layer = self.vertical_layer(
				layer,
				self.hidden_layer_sizes[i],
				activation_fn=self.activation_fn,
				reuse=reuse,
				scope="hidden_layer_" + str(i),
				layer_normalization_flag=self.layer_normalization_flag
				)
			
			if i == 0:
				residual_layer = layer
			elif i % 2 == 0 or i == len(self.hidden_layer_sizes)-1:
				if residual_layer.get_shape().as_list()[1] != layer.get_shape().as_list()[1]:
					# Make a projection layer
					residual_layer = fc(
						residual_layer,
						self.hidden_layer_sizes[i],
						activation_fn=None,
						reuse=reuse,
						scope="projection_layer_" + str(i),
						biases_initializer=None
						)
				layer += residual_layer
				residual_layer = layer
		
		output = self.vertical_layer(
				layer,
				self.output_size,
				activation_fn=None,
				reuse=reuse,
				scope="output_layer",
				layer_normalization_flag=self.layer_normalization_flag
				)
				
		return output
