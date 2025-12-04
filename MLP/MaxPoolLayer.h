// MaxPoolLayer.h
#pragma once
#include "Tensor3D.h"

// MaxPoolLayer クラス
// ・size×size の領域で最大値を取るMaxPoolingを行う
class MaxPoolLayer
{
public:
	// コンストラクタ
	// ・poolSize : プーリングの一辺のサイズ (例: 2 → 2×2 プーリング)
	MaxPoolLayer(int poolSize);
	// 順伝播する
	// ・inputFeatureMap : 入力特徴マップ (H×W×C)
	// ・戻り値 : プーリング後の出力特徴マップ
	Tensor3D Forward(const Tensor3D& inputFeatureMap);
	// 逆伝播する
	// ・dOutFeatureMap : 出力側から流れてきた勾配
	// ・戻り値 : 入力側の勾配
	Tensor3D Backward(const Tensor3D& dOutFeatureMap);

private:
	// プーリングサイズ (例: 2の場合 2×2の領域でmaxを取得する)
	int m_size;
	// 順伝播で使用した入力特徴マップ (逆伝播で最大値の位置を特定する)
	Tensor3D m_lastInputFeatureMap;
	// 順伝播の出力 (最大値 : 逆伝播で比較に使う)
	Tensor3D m_lastOutputFeatureMap;
};
