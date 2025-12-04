// ConvLayer.h
#pragma once
#include <vector>
#include "Tensor3D.h"

// ConvLayer クラス
// ・パディング付きの2D畳み込みを行う
class ConvLayer
{
public:
	// コンストラクタ
	// inputHeight : 入力の高さ
	// inputWidth : 入力の幅
	// inputChannel : 入力チャネル数
	// filterSize : カーネルの一辺のサイズ (例: 3 → 3×3)
	// outChannels : 出力チャネル数
	ConvLayer(int inputHeight, int inputWidth, int inputChannel, int filterSize, int outChannels);

	// 順伝播する
	// ・inputFeatureMap : 入力特徴マップ
	// ・戻り値 : 畳み込み結果の特徴マップ
	Tensor3D Forward(const Tensor3D& inputFeatureMap);

	// 逆伝播する
	// ・dOutputFeatureMap : 出力側からの勾配
	// ・learningRate : 学習率
	// ・戻り値 : 入力側の勾配
	Tensor3D Backward(const Tensor3D& dOutputFeatureMap, float learningRate);

private:
	// 重み配列のインデックス計算を行うヘルパ関数
	// fh, fw : フィルタ内の位置
	// ic  : 入力チャネル
	// oc  : 出力チャネル
	inline int WeightIndex(int fh, int fw, int ic, int oc) const {
		// (outChannel ごとに filterSize×filterSize×inChannels のブロック)
		return (((oc * m_numInputChannels + ic) * m_filtersize + fh) * m_filtersize + fw);
	}

private:
	// 入力高さ
	int m_inputHeight;
	// 入力幅
	int m_inputWidth;
	// 入力チャネル数
	int m_numInputChannels;
	// カーネルサイズ(一辺)
	int m_filtersize;
	// 出力チャネル数
	int m_numOutputChannels;
	// パディング量
	int m_padding;

	// 畳み込みカーネルの重み配列 (1次元配列で保持)
	std::vector<float> m_weights;
	// 出力チャネルごとのバイアス配列
	std::vector<float> m_bias;
	// 直近の入力特徴マップ(逆伝播のために保存する)
	Tensor3D m_lastInput;
};
