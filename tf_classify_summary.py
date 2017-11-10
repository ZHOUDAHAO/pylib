from auxi import *
from mytf import *
import os, time, sys
import numpy as np
			
last_time = time.time()

class tf_classify_summary(object):
	def __init__(self, use_cores = None):
		if use_cores is None:
			self.configProto = None
		else:
			self.configProto = tf.ConfigProto(intra_op_parallelism_threads=use_cores, inter_op_parallelism_threads=use_cores, allow_soft_placement=True, device_count = {'CPU': use_cores})

	def p_predlabels(self, sess, feed, start = 0, end = 50):
		print_tensor(self.pred, sess, feed, start = start, end = end) 
		print feed[self.labels][start:end]

	def add_placeholder(self, xshape, yshape):	
		self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
		self.x = tf.placeholder(tf.float32, xshape, name = 'x')
		self.labels = tf.placeholder(tf.float32, yshape, name = 'labels')
	
	def add_loss_op(self, pred, Config):
		regularizer = tf.contrib.layers.l2_regularizer(scale = Config.lamb)
		reg_term = tf.contrib.layers.apply_regularization(regularizer)
		tf.summary.scalar('reg_term', reg_term)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.x,logits = pred)) + reg_term
		return loss

	def add_acc_op(self, pred):
		pred_ = tf.argmax(pred,1)
		labels_ = tf.argmax(self.labels,1)
		prediction = tf.equal(pred_, labels_)
		acc = tf.reduce_mean(tf.cast(prediction, tf.float32))
		return acc

	def add_train_op(self, loss):
		self.optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  		with tf.control_dependencies(update_ops):
			if hasattr(self, 'assign_op'):
				with tf.control_dependencies([self.assign_op]):
					train_op = self.optimizer.minimize(loss)
			else:
				train_op = self.optimizer.minimize(loss)
		return train_op

	def train(self, Config, sgd = False):
		print_learn_args(Config)
		with tf.Session(config = self.configProto) as sess:
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
			
			if not sgd:
				print 'Traditional Gradient Descent'
				feed = self.get_batch_feed(0, Config.train_num, Config.train_num, 1, lr)
			else:
				print 'Stochastic Gradient Descent'
			test_feed = self.get_batch_feed(0, Config.test_num, Config.test_num, 0, lr)

			PRINT_TURN = 20
			ASS_LR_TURN = 60 # assess whether learning rate should halve
			update_lr = True

			def update_time():
				global last_time
				print 'time passed by:',time.time() - last_time
				print '*' * 100
				last_time = time.time()

			print 'Start Training!'
			for self.epoch in range(Config.train_times):
				if sgd:
					feed = self.get_batch_feed(0, Config.train_num, Config.mini_bsize, 1, lr)
				
				if self.save and self.epoch % SAVE_TURN == 0:
					self.saver.save(sess, 'model', global_step = self.epoch)
	
				if self.epoch % PRINT_TURN == 0:
					print 'epoch',self.epoch
					train_summary, train_loss, train_acc = sess.run([self.merged, self.loss, self.acc], feed)
					print 'Loss:',train_loss
					self.train_writer.add_summary(train_summary, self.epoch)
					update_time()
					
				if self.epoch % ASS_LR_TURN == 0:
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

						if no_update_turn >= 5:
							max_acc = 0
							lr /= 2.0 
							halve_times += 1
							no_update_turn = 0
				sess.run(self.train_op, feed_dict = feed)

				if halve_times > 10:
					update_lr = False

			print 'Training Completed!'
			self.p_predlabels(sess, test_feed)
			print_tensor(self.acc, sess, feed, 'Train Accuracy')
			
			print_tensor(self.loss, sess, test_feed, 'Test Loss:')
			print_tensor(self.acc, sess, test_feed, 'Test Accuracy')
			print dt.now()	
			print_learn_args(Config)
