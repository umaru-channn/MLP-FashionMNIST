// CNNModel.cpp
// CNN の順伝播・逆伝播を実装したファイル
#include "CNNModel.h"
#include <algorithm>
#include <cmath>

// Softmax（数値安定化）
// 目的：生のスコア(logits)を確率(0.0〜1.0)に変換する
static std::vector<float> Softmax(const std::vector<float>& logits)
{
	// Softmax の安定化のため max(logits) を取得
	float maxv = *std::max_element(logits.begin(), logits.end());
	// exp() の結果を格納する配列
	std::vector<float> exps(logits.size());
	// exp の合計（正規化に使用）
	float sum = 0.0f;
	// logits[i] - maxv を exp() に通す
	for (size_t i = 0; i < logits.size(); i++)
	{
		// 数値安定化のため maxv を引いた値に対して exp() を計算し、Softmax の分子を作る
		exps[i] = std::exp(logits[i] - maxv);
		// Softmax の正規化用に exp の総和を加算していく
		sum += exps[i];
	}
	// 合計が 1 になるように割る（確率分布になる）
	for (size_t i = 0; i < exps.size(); i++)
	{
		// Softmax の分子 exp(logit) を exp の総和で割り、確率（0〜1）に正規化する
		exps[i] /= sum;
	}
	// Softmax 結果を返す
	return exps;
}

// CNNModel コンストラクタ
// 畳み込み・プーリング・全結合層の設定
CNNModel::CNNModel()
	: 
	m_conv1(28, 28, 1, 3, 8),
	m_pool1(2),
	m_conv2(14, 14, 8, 3, 16),
	m_pool2(2),
	m_fcl1(7 * 7 * 16, 128),
	m_fcl2(128, 10)
{
}

// Forward（順伝播）
// 画像 → CNN → 10 クラス確率 を求める
std::vector<float> CNNModel::Forward(const Tensor3D& inputImage)
{
	// 入力画像 Tensor3D をメンバ変数に保存（Backprop 用）
	m_inputImage = inputImage;
	// Conv1 の順伝播（28×28×1 → 28×28×8）
	m_conv1Output = m_conv1.Forward(m_inputImage);
	// Conv1 の出力に ReLU を適用（負の値を 0 にする）
	Tensor3D relu1Out = m_relu1.Forward(m_conv1Output);
	// MaxPool1 を適用（空間解像度を 28→14 にダウンスケール）
	m_pool1Output = m_pool1.Forward(relu1Out);
	// Conv2 の順伝播（14×14×8 → 14×14×16）
	m_conv2Output = m_conv2.Forward(m_pool1Output);
	// Conv2 の出力に ReLU を適用
	Tensor3D relu2Out = m_relu2.Forward(m_conv2Output);
	// MaxPool2 を適用（14→7 にさらにダウンスケール）
	m_pool2Output = m_pool2.Forward(relu2Out);
	// Flatten により 7×7×16 → 784 の 1次元ベクトルへ変換
	Tensor3D flatTensor = m_flatten.Forward(m_pool2Output);
	// Flatten が生成した 1次元ベクトル（FC1 の入力）を取得
	const std::vector<float>& flatVec = m_flatten.GetFlatOutput();
	// 全結合層 FC1（784 → 128）で特徴変換
	m_hiddenLayer1 = m_fcl1.Forward(flatVec);
	// FC1 出力に ReLU を適用（非線形性を追加）
	for (size_t i = 0; i < m_hiddenLayer1.size(); i++)
	{
		// もし値が 0 以下なら ReLU の性質により 0 にする
		if (m_hiddenLayer1[i] < 0.0f)	{
			m_hiddenLayer1[i] = 0.0f;
		}
	}
	// 全結合層 FC2（128 → 10）でクラス別スコア（logits）を計算
	auto logits = m_fcl2.Forward(m_hiddenLayer1);
	// Softmax を適用して 10 クラスの確率分布に変換
	m_outputVector = Softmax(logits);
	// 推論結果（確率ベクトル）を返す
	return m_outputVector;
}


// CrossEntropy Loss を計算
// target：one-hot ベクトル
// outputVector：Softmax 出力
float CNNModel::ComputeLoss(const std::vector<float>& target)
{
	// log(0) による -inf を防ぐためのごく小さな値（安定性のために足す）
	float eps = 1e-9f;
	// 損失値（交差エントロピー）を格納する変数
	float loss = 0.0f;
	// 10 クラス（0〜9）すべてについて損失を計算するループ
	for (int i = 0; i < 10; i++)
	{
		// 交差エントロピーの式： - Σ t[i] * log( y[i] )
		// target[i] が 1 の位置だけ実質的に計算される（one-hot だから）
		loss -= target[i] * std::log(m_outputVector[i] + eps);
	}
	// 合計した損失値を返す
	return loss;
}
// Backward（逆伝播）
// 目的：Forward の逆順に勾配を流し、重みを更新する
void CNNModel::Backward(float learningRate)
{
	// Softmax と CrossEntropy を組み合わせた場合の誤差勾配を計算する（非常にシンプルになる）
	// 数式 dL/dz = y - t （Softmax の出力 - 教師データ）をそのまま使う
	std::vector<float> dSoftmax(10);

	// 各クラス（0〜9）について勾配を計算する
	for (int i = 0; i < 10; i++) 	{
		// Softmax の出力 y[i] から 教師の one-hot 値 t[i] を引いたものが勾配になる
		dSoftmax[i] = m_outputVector[i] - targetVector[i];
	}
	// FC2 の逆伝播
	auto dFC2Input = m_fcl2.Backward(dSoftmax, learningRate);
	// FC1 層で行った ReLU（max(0, x)）の効果を逆伝播処理に反映する
	for (size_t i = 0; i < dFC2Input.size(); i++)
	{
		// ReLU の入力値が 0 以下だった部分は、逆伝播する勾配も 0 に切り落とす
		if (m_hiddenLayer1[i] <= 0.0f)	{
			// 勾配を 0 にする（ReLU の性質：負の部分には勾配を流さない）
			dFC2Input[i] = 0.0f;
		}
	}
	// FC1 の逆伝播
	auto dFC1Input = m_fcl1.Backward(dFC2Input, learningRate);
	// Flatten の逆伝播のため、1x1xN の Tensor3D に詰め直す
	Tensor3D dFlat(1, 1, (int)dFC1Input.size());
	// FC1 の勾配ベクトル（1次元配列）を Flatten の逆伝播用に 1×1×N の Tensor3D に詰め直す
	for (int i = 0; i < (int)dFC1Input.size(); i++)	{
		// dFlat の i 番目のチャンネルに勾配をコピーする（Flatten 前の形に戻すため）
		dFlat(0, 0, i) = dFC1Input[i];
	}
	// Flatten の逆伝播（全結合層 → プーリング層へ勾配を戻す）
	Tensor3D dPool2 = m_flatten.Backward(dFlat, learningRate);
	// MaxPool2 の逆伝播（プーリング → ReLU2 へ勾配を戻す）
	Tensor3D dRelu2Out = m_pool2.Backward(dPool2);
	// ReLU2 の逆伝播（ReLU → Conv2 へ勾配を戻す）
	Tensor3D dConv2Out = m_relu2.Backward(dRelu2Out, learningRate);
	// Conv2 の逆伝播（Conv2 → Pool1 へ勾配を戻す）
	Tensor3D dPool1Out = m_conv2.Backward(dConv2Out, learningRate);
	// MaxPool1 の逆伝播（プーリング → ReLU1 へ勾配を戻す）
	Tensor3D dRelu1Out = m_pool1.Backward(dPool1Out);
	// ReLU1 の逆伝播（ReLU → Conv1 へ勾配を戻す）
	Tensor3D dConv1Out = m_relu1.Backward(dRelu1Out, learningRate);
	// Conv1 の逆伝播（Conv1 のパラメータ更新）
	m_conv1.Backward(dConv1Out, learningRate);
}

// Predict（もっとも確率の高いクラスIDを返す）
int CNNModel::Predict(const Tensor3D& inputTensor)
{
	auto probs = Forward(inputTensor);
	return (int)(std::max_element(probs.begin(), probs.end()) - probs.begin());
}

// PredictProba（確率ベクトルを返す）
std::vector<float> CNNModel::PredictProba(const Tensor3D& inputTensor)
{
	return Forward(inputTensor);
}

// Top-10（確率の高い順に並べた (クラスID, 確率) のリスト）を返す関数
std::vector<std::pair<int, float>> CNNModel::GetTop10(const Tensor3D& inputTensor)
{
	// Forward を実行して Softmax の確率ベクトルを取得
	auto probs = Forward(inputTensor);
	// (クラスID, 確率) を格納するためのベクタを作成
	std::vector<std::pair<int, float>> v;
	// v の領域を 10 個分（クラス数）あらかじめ確保して高速化
	v.reserve(10);
	// 0〜9 の各クラスについて (ID, そのクラスの確率) を追加する
	for (int i = 0; i < 10; i++)	{
		v.emplace_back(i, probs[i]);
	}
	// 確率の高い順になるようにペアをソートする
	std::sort(	v.begin(), v.end(), [](auto& a, auto& b)
		{
			// second（確率）が大きいものから順に並べる
			return a.second > b.second;
		}
	);
	// 上位 10 個（＝すべて）を返す
	return v;
}

// Top-10 の (クラスID, 確率) を wstring のクラス名に変換する関数
std::vector<std::wstring> CNNModel::GetTop10Names(const std::vector<std::pair<int, float>>& top10)
{
	// Fashion-MNIST の 10 クラス名（ID:0〜9 に対応する）
	static const wchar_t* names[10] =
	{
		 L"T-shirt/top",
		 L"Trouser",
		 L"Pullover",
		 L"Dress",
		 L"Coat",
		 L"Sandal",
		 L"Shirt",
		 L"Sneaker",
		 L"Bag",
		 L"Ankle boot"
	};
	// 結果のクラス名（wstring）を格納する配列を用意する
	std::vector<std::wstring> result;
	// 10 個分のメモリをあらかじめ確保して push_back を高速化する
	result.reserve(top10.size());
	// Top-10 の各項目に対してループを回す
	for (size_t i = 0; i < top10.size(); i++)	{
		// top10[i].first はクラスID（0〜9）
		int id = top10[i].first;
		// 対応するクラス名（names[id]）を result に追加する
		result.push_back(names[id]);
	}
	// 変換したクラス名リストを返す
	return result;
}

