from auxi import *
from mytf import *
import os, time, sys
import numpy as np
from tensorflow.python import debug as tf_debug
			
last_time = time.time()

class tf_classify_summary(object):
	def __init__(self, use_cores = None):
		# self.configProto = tf.ConfigProto(intra_op_parallelism_threads=use_cores, inter_op_parallelism_threads=use_cores, allow_soft_placement=True, device_count = {'CPU': use_cores})
		# self.configProto = tf.ConfigProto(device_count = {'CPU': use_cores})
		# self.configProto.gpu_options.allocator_type = 'BFC'
		self.configProto = tf.ConfigProto(device_count = {'GPU': 0})

	def p_predlabels(self, sess, feed, start = 0, end = 50):
		print_tensor(self.pred, sess, feed, start = start, end = end) 
		print feed[self.labels][start:end]

	def add_placeholder(self, xshape, yshape):	
		self.x = tf.placeholder(tf.float32, xshape, name = 'x')
		self.labels = tf.placeholder(tf.float32, yshape, name = 'labels')
	
	def add_loss_op(self, pred, Config):
		reg_term = Config.lamb * sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
		tf.summary.scalar('reg_term', reg_term)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.labels,logits = pred)) + reg_term
		return loss

	def add_acc_op(self, pred):
		pred_ = tf.argmax(pred,1)
		labels_ = tf.argmax(self.labels,1)
		prediction = tf.equal(pred_, labels_)
		acc = tf.reduce_mean(tf.cast(prediction, tf.float32))
		return acc

	def add_train_op(self):
		self.optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  		with tf.control_dependencies(update_ops):
			if hasattr(self, 'assign_op'):
				with tf.control_dependencies([self.assign_op]):
					train_op = self.optimizer.minimize(self.loss)
			else:
				train_op = self.optimizer.minimize(self.loss)
		return train_op

	def train(self, Config):
		print_learn_args(Config)
		with tf.Session(config = self.configProto) as sess:
			# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
			# sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
			self.build()
			self.merged = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			sess.run(init)

			summaries_dir = get_summaries_dir()
			train_dir = summaries_dir + 'train'
			test_dir = summaries_dir + 'test'
			os.mkdir(train_dir)
			os.mkdir(test_dir)
			self.train_writer = tf.summary.FileWriter(train_dir, sess.graph)
			self.test_writer = tf.summary.FileWriter(test_dir)

			lr = Config.lr
			halve_times = 0
			max_acc = 0.0
			no_update_turn = 0
			
			test_feed = self.get_batch_feed(Config.test_num, 0, lr)

			PRINT_TURN = 20
			ASS_LR_TURN = 80 # assess whether learning rate should halve
			update_lr = True
			SAVE_TURN = 200		

			def update_time():
				global last_time
				print 'time passed by:',time.time() - last_time
				print '*' * 100
				last_time = time.time()

			print 'Start Training!'
			for self.epoch in range(Config.train_times):
				feed = self.get_batch_feed(Config.mini_bsize, 1, lr)
				sess.run([self.train_op], feed_dict = feed)
				
				if self.epoch % PRINT_TURN == 0:
					print 'epoch',self.epoch
					train_summary, train_loss, train_acc = sess.run([self.merged, self.loss, self.acc], feed)
					print 'Loss:',train_loss
					self.train_writer.add_summary(train_summary, self.epoch)
					update_time()
			
				if self.epoch > 0 and self.epoch % ASS_LR_TURN == 0:
					if hasattr(self, 'pretrain_name'):
						print 'pretrain_name:',self.pretrain_name,',is it true?'
					print 'Have halved',halve_times,'times'
					test_summary, test_loss, test_acc = sess.run([self.merged, self.loss, self.acc], test_feed)
					self.test_writer.add_summary(test_summary, self.epoch)
					print 'Test Accuracy:',test_acc
					update_time()

					if update_lr:
						if test_acc > max_acc:
							max_acc = test_acc
							no_update_turn = 0
						else:
							no_update_turn += 1

						if no_update_turn >= 4:
							max_acc = 0
							lr /= 2.0 
							halve_times += 1
							no_update_turn = 0
					
				if halve_times > 10:
					update_lr = False

			print 'Training Completed!'
			self.p_predlabels(sess, test_feed)
			print_tensor(self.acc, sess, feed, 'Train Accuracy')
			
			print_tensor(self.loss, sess, test_feed, 'Test Loss:')
			print_tensor(self.acc, sess, test_feed, 'Test Accuracy')
			print dt.now()	
			print_learn_args(Config)
