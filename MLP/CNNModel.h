#pragma once
// CNNModel.h
// Fashion-MNIST 用の CNN モデル定義
// 構成：Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → FC1 → ReLU → FC2 → Softmax

#include <vector>
#include <string>
#include <utility>

#include "Tensor3D.h"								// 3次元テンソル（H×W×C）
#include "ConvLayer.h"							// 畳み込み層（Conv）
#include "MaxPoolLayer.h"					// 最大値プーリング層（Pool）
#include "FullyConnectedLayer.h"	// 全結合層（FC）
#include "ReLULayer.h"							// ReLU 活性化層
#include "FlattenLayer.h"						// Flatten（3D → 1D ベクトル変換）

// CNNModel クラス
// ・Forward() : 画像を入力し確率分布（10クラス）を出力
// ・Backward(): 逆伝播し各層のパラメータ更新を実施
// ・Predict(): 予測クラス ID 取得
// ・GetTop10(): Top-10 の予測確率取得
class CNNModel
{
public:
	// コンストラクタ
	// ・Conv/Pool/FC 層の初期化を行う
	CNNModel();

	// 順伝播する
	// ・入力 Tensor3D（28×28×1）→ 確率ベクトル（10次元）を返す
	std::vector<float> Forward(const Tensor3D& x);

	// 逆伝播する
	// ・learningRate: 学習率
	// ・Softmax + CrossEntropy の勾配を流し全層を更新する
	void Backward(float learningRate);

	// 損失を計算する
	// ・CrossEntropyLoss を返す
	float ComputeLoss(const std::vector<float>& target);
	// 学習用の教師ラベル（one-hot）をセット
	void SetTarget(const std::vector<float>& target) { targetVector = target; }
	// 画像を入力して最も確率の高いクラスIDを返す
	int Predict(const Tensor3D& inputTensor);
	// 画像を入力して Softmax の確率ベクトルを返す
	std::vector<float> PredictProba(const Tensor3D& inputTensor);
	// Top-10 の (クラスID, 確率) を返す
	std::vector<std::pair<int, float>> GetTop10(const Tensor3D& inputTensor);
	// Top-10 をクラス名文字列に変換して返す
	std::vector<std::wstring> GetTop10Names(const std::vector<std::pair<int, float>>& top10);

private:
	// Forward で使用する各層の出力（Backward で必要）
	 // 入力画像（28×28×1）
	Tensor3D m_inputImage;  
	// Conv1 の出力（28×28×8）
	Tensor3D m_conv1Output; 
	// Pool1 の出力（14×14×8）
	Tensor3D m_pool1Output;  
	// Conv2 の出力（14×14×16）
	Tensor3D m_conv2Output;  
	// Pool2 の出力（7×7×16）
	Tensor3D m_pool2Output;  

	FlattenLayer m_flatten;  // 7×7×16 → 784次元ベクトルに変換する層
	std::vector<float> m_hiddenLayer1; // FC1 の出力（ReLU後、128次元）
	std::vector<float> m_outputVector; // Softmax 出力（10次元）
	std::vector<float> targetVector;   // 教師データ(one-hot 10次元)

	// CNN を構成する層インスタンス
	// 第1畳み込み層（3×3、出力 8チャンネル）
	ConvLayer m_conv1;
	// 第1プーリング層（2×2）
	MaxPoolLayer m_pool1;
	// 第2畳み込み層（3×3、出力 16チャンネル）
	ConvLayer m_conv2;       
	// 第2プーリング層（2×2）
	MaxPoolLayer m_pool2;   
	 // FC1（784 → 128）
	FullyConnectedLayer m_fcl1;
	// FC2（128 → 10）
	FullyConnectedLayer m_fcl2; 
	 // Conv1 直後の ReLU
	ReLULayer m_relu1;
	// Conv2 直後の ReLU
	ReLULayer m_relu2;      
};
