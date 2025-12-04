// FullyConnectedLayer.cpp
#include "FullyConnectedLayer.h"
#include <random>
#include <cmath>

// 正規分布に従う乱数を生成する (He 初期化用)
static float GenerateNormalRandom(float mean, float stddev)
{
	// メルセンヌツイスタ乱数生成器
	static std::mt19937 rng(std::random_device{}());
	// 平均と標準偏差を指定した正規分布
	std::normal_distribution<float> dist(mean, stddev);
	// 分布に従う乱数を返す
	return dist(rng);
}

// コンストラクタ(入力次元と出力次元を指定)
FullyConnectedLayer::FullyConnectedLayer(int inputSize, int outputSize)
	: m_inSize(inputSize),
	m_outSize(outputSize)
{
	// He 初期化用の標準偏差を計算する
	float stddev = std::sqrt(2.0f / m_inSize);

	// 重み行列を正規分布で初期化する
	m_weights.resize(m_inSize * m_outSize);
	for (auto& w : m_weights)
	{
		w = GenerateNormalRandom(0.0f, stddev);
	}

	// バイアスを 0 で初期化する
	m_bias.assign(m_outSize, 0.0f);

	// 順伝播で使用する入力ベクトルを確保する
	m_lastInputVector.resize(m_inSize);
}

// 順伝播する
std::vector<float> FullyConnectedLayer::Forward(const std::vector<float>& inputVector)
{
	// 入力を保存する (逆伝播で使用)
	m_lastInputVector = inputVector;

	// 出力ベクトルを確保する
	std::vector<float> outputVector(m_outSize);

	// 出力ニューロンごとに計算する
	for (int outNeuron = 0; outNeuron < m_outSize; outNeuron++)
	{
		// バイアスを初期値に設定
		float sum = m_bias[outNeuron];

		// 各入力ニューロンの寄与分を加算する
		for (int inNeuron = 0; inNeuron < m_inSize; inNeuron++)
		{
			sum += m_weights[WeightIndex(outNeuron, inNeuron)] * inputVector[inNeuron];
		}

		// 出力ベクトルに格納する
		outputVector[outNeuron] = sum;
	}

	// 出力ベクトルを返す
	return outputVector;
}

// 逆伝播する
// 出力勾配 dOut を受け取り、入力勾配 dInput を計算する
// 同時に重みとバイアスを SGD で更新する
std::vector<float> FullyConnectedLayer::Backward(const std::vector<float>& dOut, float learningRate)
{
	// 入力側勾配を 0 で初期化する
	std::vector<float> dInputGradients(m_inSize, 0.0f);

	// 元の重みを退避する (入力側勾配の計算に使用)
	std::vector<float> oldWeights = m_weights;

	// 出力ニューロンごとに勾配計算する
	for (int outNeuron = 0; outNeuron < m_outSize; outNeuron++)
	{
		// 出力勾配 dL/d(y_outNeuron)
		float grad = dOut[outNeuron];

		// バイアスを更新する (b -= η * dL/db)
		m_bias[outNeuron] -= learningRate * grad;

		// 入力ニューロンごとの勾配計算する
		for (int inNeuron = 0; inNeuron < m_inSize; inNeuron++)
		{
			// 重み配列インデックスを計算する
			int idx = WeightIndex(outNeuron, inNeuron);

			// 入力側勾配を計算する (更新前の重みを使用)
			// dL/dx_in += dL/dy_out * W(out,in)
			dInputGradients[inNeuron] += grad * oldWeights[idx];

			// 重み勾配を計算する
			// dL/dW(out,in) = dL/dy_out * x_in
			float gradW = grad * m_lastInputVector[inNeuron];

			// SGD による重み更新 (W -= η * dL/dW)
			m_weights[idx] -= learningRate * gradW;
		}
	}

	// 入力側勾配を返す
	return dInputGradients;
}
