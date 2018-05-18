import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions
import numpy as np
import random
import numpy as np
import pandas as pd

log_interval = 100

# データファイルを読み込む
df = pd.read_csv('amzn-anon-access-samples-history-2.0.csv')
users = np.unique(df.LOGIN.values)

# 時系列データを作る
logdata = {}
for u in users: 
	logdata[u] = []
	for v in df[df.LOGIN==u].ACTION.values:
		d = 0 if v == 'add_access' else 1
		logdata[u].append(d)

# RNNの定義をするクラス
class Encoder_Decoder_Model(chainer.Chain):

	def __init__(self):
		super(Encoder_Decoder_Model, self).__init__()
		with self.init_scope():
			self.hidden1 = L.Linear(4, 6)
			self.encoder = L.StatefulGRU(6, 2)
			self.hidden2 = L.Linear(4, 6)
			self.decoder = L.StatefulGRU(6, 2)
			self.output = L.Linear(2, 4)

	def reset_state(self):
		self.encoder.reset_state()
		self.decoder.reset_state()

	def encode_one(self, x):
		h1 = F.tanh(self.hidden1(x))
		y = self.encoder(h1)
		return y

	def decode_one(self, x):
		h1 = F.tanh(self.hidden2(x))
		h2 = F.tanh(self.decoder(h1))
		y = self.output(h2)
		return y

def one_hot(h):
	z = np.zeros((1,4), dtype=np.float32)
	z[0,h] = 1.0
	return z
	
# カスタムUpdaterのクラス
class RNNUpdater(training.StandardUpdater):

	def __init__(self, optimizer):
		super(RNNUpdater, self).__init__(
			None,
			optimizer,
			device=-1
		)
		self.loss_log = []
		self.n_iter = 0

	# イテレーターがNoneなのでエラーが出ないようにオーバライドする
	@property
	def epoch(self):
		return 0

	@property
	def epoch_detail(self):
		return 0.0

	@property
	def previous_epoch_detail(self):
		return 0.0

	@property
	def is_new_epoch(self):
		return False
		
	def finalize(self):
		pass

	def update_core(self):
		# Optimizerを取得
		optimizer = self.get_optimizer('main')
		# ニューラルネットワークを取得
		model = optimizer.target

		key = random.choice(list(logdata.keys()))

		# RNNのステータスをリセットする
		model.reset_state()

		# 開始文字を入力
		y = model.encode_one(one_hot(2))
		# 逆順にエンコーダーに入力
		for w in logdata[key][::-1]:
			# 一つRNNを実行
			y = model.encode_one(one_hot(w))
		# 終了文字を入力
		y = model.encode_one(one_hot(3))
		
		# ステータスを引き継ぐ
		model.decoder.set_state(model.encoder.h)
		
		loss = 0

		# 開始文字を入力
		y = model.decode_one(one_hot(3))
		loss += F.softmax_cross_entropy(y, np.array([logdata[key][0]]))
		# 文の長さ分だけ
		for i in range(1, len(logdata[key])):
			# 一つ前の出力をRNNに入力
			y = model.decode_one(y)
			loss += F.softmax_cross_entropy(y, np.array([logdata[key][i]]))

		# 重みデータを一旦リセットする
		optimizer.target.cleargrads()
		# 誤差関数から逆伝播する
		loss.backward()
		# 新しい重みデータでアップデートする
		optimizer.update()
		self.n_iter += 1
		
		self.loss_log.append(np.mean(loss.data))
		if len(self.loss_log) == log_interval:
			print('%d iter, loss = %f'%(self.n_iter,np.mean(self.loss_log)))
			self.loss_log = []

# ニューラルネットワークの作成
model = Encoder_Decoder_Model()

# 機械学習を実行する
import os.path
if not os.path.isfile('logmap.npz'):
	print('start training')
	# 誤差逆伝播法アルゴリズムを選択
	optimizer = optimizers.Adam()
	optimizer.setup(model)
	updater = RNNUpdater(optimizer)
	trainer = training.Trainer(updater, (100000, 'iteration'), out="result")
	trainer.run()
	chainer.serializers.save_npz( 'logmap.npz', model )
else:
	chainer.serializers.load_npz( 'logmap.npz', model )

# 全てのログに対して実行
print('make result')
X = []
Y = []
XY = []
D = []
for key in logdata.keys():
	# RNNのステータスをリセットする
	model.reset_state()
	# 開始文字を入力
	y = model.encode_one(one_hot(2))
	# 逆順にエンコーダーに入力
	for w in logdata[key][::-1]:
		# 一つRNNを実行
		y = model.encode_one(one_hot(w))
	# 終了文字を入力
	y = model.encode_one(one_hot(3))
		
	# ステータスを取得
	state = model.encoder.h
	X.append(state.data[0][0])
	Y.append(state.data[0][1])
	XY.append(state.data[0])
	D.append(' '.join(list(map(str,logdata[key]))))
	if len(XY) % log_interval == 0:
		print('%d / %d'%(len(XY),len(logdata.keys())))

# 結果を散布図にして保存
import matplotlib.pyplot as plt
from sklearn import cluster
kmean = cluster.AgglomerativeClustering(n_clusters=2, linkage='average')
C = kmean.fit_predict(XY)
df = pd.DataFrame({'x': X,'y': Y, 'c':C, 'd':D})
df.to_csv('result.csv')
df.plot(kind='scatter', x='x', y='y', c=C, colormap='cool')
plt.savefig('result.png')
plt.clf()

