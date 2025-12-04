// ConvLayer.cpp
#include "ConvLayer.h"
#include <random>
#include <cmath>

// 正規分布に従う乱数を生成する(He 初期化用)
static float GenerateNormalRandomConv(float mean, float stddev)
{
	// メルセンヌツイスタ乱数生成器（シードはランダムデバイス）
	static std::mt19937 rng(std::random_device{}());
	// 平均と標準偏差を指定した正規分布
	std::normal_distribution<float> dist(mean, stddev);
	// 分布に従う乱数を返す
	return dist(rng);
}

// コンストラクタ(入力サイズ, チャネル数, カーネルサイズ, 出力チャネル数)
ConvLayer::ConvLayer(int inputHeight, int inputWidth, int inputChannel, int filterSize, int outChannels)

	:
	// 入力画像の高さを記録する
	m_inputHeight(inputHeight)
	// 入力画像の幅を記録する
	, m_inputWidth(inputWidth)
	// 入力チャネル数（1=グレースケール、3=RGBなど）
	, m_numInputChannels(inputChannel)
	// カーネル（フィルタ）の縦横サイズ（例：3 → 3×3）
	, m_filtersize(filterSize)
	// 出力チャネル数（フィルタ数、特徴マップ数）
	, m_numOutputChannels(outChannels)
	// 3×3 カーネルで入力と出力のサイズを同じに保つためのパディング（左右上下に1ずつ）
	, m_padding(1)
{
	// 1つのフィルタが見る入力の総数（He 初期化に使う）
	int inputConnections = m_filtersize * m_filtersize * m_numInputChannels;
	// He 初期化の標準偏差 σ = sqrt(2 / fan_in)
	float stddev = std::sqrt(2.0f / inputConnections);
	// 重み配列を「フィルタ数 × 入力チャネル数 × フィルタ高さ × フィルタ幅」分確保する
	m_weights.resize(m_numOutputChannels * m_numInputChannels * m_filtersize * m_filtersize);
	// 各重みを平均0・標準偏差stddevの正規分布で初期化する
	for (auto& weight : m_weights)
	{
		// 正規乱数で1つの重みを初期化
		weight = GenerateNormalRandomConv(0.0f, stddev);
	}
	// バイアスは各出力チャネルごとに1つずつ存在するため、0 で初期化する
	m_bias.assign(m_numOutputChannels, 0.0f);
}

// 順伝播する(入力特徴マップから出力特徴マップを計算)
Tensor3D ConvLayer::Forward(const Tensor3D& inputFeatureMap)
{
	// 逆伝播用に入力を保持する
	m_lastInput = inputFeatureMap;
	// 出力特徴マップを確保する
	// パディング=1, ストライド=1 のため、出力サイズは入力と同じ (H×W×outChannels)
	Tensor3D outputFeatureMap(m_inputHeight, m_inputWidth, m_numOutputChannels);
	// 出力位置 (h, w) ごとに計算する
	for (int h = 0; h < m_inputHeight; h++)
	{
		for (int w = 0; w < m_inputWidth; w++)
		{
			// 出力チャネルごとに計算する
			for (int k = 0; k < m_numOutputChannels; k++)
			{
				// 出力の初期値としてバイアスを設定
				float sum = m_bias[k];
				// フィルタ内の各位置 (fh, fw) を走査する
				for (int fh = 0; fh < m_filtersize; fh++)
				{
					for (int fw = 0; fw < m_filtersize; fw++)
					{
						// 入力画像上の対応位置(高さ)
						int ih = h + fh - m_padding;
						// 入力画像上の対応位置(幅)
						int iw = w + fw - m_padding;
						// パディング領域はスキップする
						if (ih < 0 || iw < 0 || ih >= m_inputHeight || iw >= m_inputWidth) { continue; 	}
						// 各入力チャネルについて和を取る
						for (int ic = 0; ic < m_numInputChannels; ic++)
						{
							// 入力値
							float inputValue = inputFeatureMap(ih, iw, ic);
							// 対応する重み
							float weight = m_weights[WeightIndex(fh, fw, ic, k)];
							// 積を加算
							sum += inputValue * weight;
						}
					}
				}
				// 出力特徴マップに格納する
				outputFeatureMap(h, w, k) = sum;
			}
		}
	}
	// 出力特徴マップを返す
	return outputFeatureMap;
}

// 逆伝播する(勾配を計算し、重みとバイアスを更新する)
Tensor3D ConvLayer::Backward(const Tensor3D& dOutputFeatureMap, float learningRate)
{
	// 入力側勾配 (H×W×inChannels) を 0 で初期化
	Tensor3D dInputFeatureMap(m_inputHeight, m_inputWidth, m_numInputChannels);
	dInputFeatureMap.Zero();

	// 重み勾配用の配列を 0 で初期化
	std::vector<float> dWeights(m_weights.size(), 0.0f);
	// バイアス勾配用の配列を 0 で初期化
	std::vector<float> dBiasGradient(m_numOutputChannels, 0.0f);

	// 出力特徴マップ（dOutputFeatureMap）の各画素 (h, w) について勾配計算を行う
	for (int h = 0; h < m_inputHeight; h++)
	{
		// 横方向の画素位置についてループする
		for (int w = 0; w < m_inputWidth; w++)
		{
			// 各出力チャンネル（フィルタ数）に対して処理する
			for (int k = 0; k < m_numOutputChannels; k++)
			{
				// 出力の勾配 dL/d(out) を取得（Conv2D の逆伝播の出発点）
				float gradient = dOutputFeatureMap(h, w, k);
				// バイアスは全結合同様、出力勾配の総和がそのまま勾配になる
				dBiasGradient[k] += gradient;
				// フィルタ（カーネル）内の縦位置をループ（fh = filter height）
				for (int fh = 0; fh < m_filtersize; fh++)
				{
					// フィルタ内の横位置をループ（fw = filter width）
					for (int fw = 0; fw < m_filtersize; fw++)
					{
						// 入力側の位置（パディング分だけずれる）
						int ih = h + fh - m_padding;
						int iw = w + fw - m_padding;
						// パディング範囲外は無視する
						if (ih < 0 || iw < 0 || ih >= m_inputHeight || iw >= m_inputWidth) { continue; }
						// 入力チャンネル（RGB など）ごとにループする
						for (int ic = 0; ic < m_numInputChannels; ic++)
						{
							// 重み配列のインデックス（fh, fw, ic, k の4つで1つの重みに対応）
							int idx = WeightIndex(fh, fw, ic, k);
							// 重みの勾配 dW を加算（dW = dL/d(out) * 入力値）
							dWeights[idx] += gradient * m_lastInput(ih, iw, ic);
							// 入力側勾配 dL/d(input) に重みを掛けた値を足し込む（誤差を入力に伝播）
							dInputFeatureMap(ih, iw, ic) += gradient * m_weights[idx];
						}
					}
				}
			}
		}
	}


	// 勾配降下法により畳み込みカーネル（重み）を更新する（w = w - η * dw）
	for (int i = 0; i < (int)m_weights.size(); i++)
	{
		// 計算された重み勾配を使って、対応する重みを少しずつ減らす
		m_weights[i] -= learningRate * dWeights[i];
	}
	// 勾配降下法により各出力チャンネルのバイアス項を更新する（b = b - η * db）
	for (int k = 0; k < m_numOutputChannels; k++)
	{
		// 計算済みのバイアス勾配を使ってバイアス値を更新する
		m_bias[k] -= learningRate * dBiasGradient[k];
	}
	// 入力側勾配を返す
	return dInputFeatureMap;
}
