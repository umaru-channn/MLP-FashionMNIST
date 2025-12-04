// FullyConnectedLayer.h
#pragma once
#include <vector>

// 完全結合層クラス
// ・入力ベクトル → 出力ベクトル の線形変換 (y = W x + b)
// ・活性化関数は外側(CNNModel)で行う
class FullyConnectedLayer
{
public:
	// コンストラクタ
	// inputSize  : 入力次元数
	// outputSize : 出力次元数
	FullyConnectedLayer(int inputSize, int outputSize);

	// 順伝播する
	// inputVector : 入力ベクトル (長さ inputSize)
	// 戻り値 : 出力ベクトル (長さ outputSize)
	std::vector<float> Forward(const std::vector<float>& inputVector);

	// 逆伝播する
	// dOut : 出力側勾配 (長さ outputSize)
	// learningRate : 学習率
	// 戻り値 : 入力側勾配 (長さ inputSize)
	std::vector<float> Backward(const std::vector<float>& dOut, float learningRate);

private:
	// 重み配列のインデックスを計算する
	// outNeuron : 出力ニューロン index
	// inNeuron  : 入力ニューロン index
	inline int WeightIndex(int outNeuron, int inNeuron) const
	{
		// [outNeuron][inNeuron] と考えたときの1次元配列上の位置
		return outNeuron * m_inSize + inNeuron;
	}

private:
	// 入力次元数
	int m_inSize;
	// 出力次元数
	int m_outSize;
	// 重み配列 (サイズ: m_outSize * m_inSize)
	std::vector<float> m_weights;
	// バイアス配列 (サイズ: m_outSize)
	std::vector<float> m_bias;
	// 直近の Forward で使用した入力ベクトル (逆伝播時に使用)
	std::vector<float> m_lastInputVector;
};
