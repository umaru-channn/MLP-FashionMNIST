#pragma once
// ReLULayer.h
// 活性化層（ReLU: Rectified Linear Unit）
// ・Forward : y = max(0, x)
// ・Backward: x > 0 のときだけ勾配を通し、x <= 0 のとき勾配 0に
// ・最も一般的な CNN の活性化関数
#include "Tensor3D.h"
#include "IBaseLayer.h"

// ReLULayer クラス
// ・CNNのReLU 活性化層
// ・Forward : 要素ごとに max(0, x)
// ・Backward : 入力が 0 以下だった位置は勾配 0に
class ReLULayer : public IBaseLayer
{
public:
	// コンストラクタ（特に何もなし）
	ReLULayer() = default;

	// 順伝播する
	// ・入力テンソルに max(0, x) を適用
	// ・Backward で使用するため、入力を lastInput に保存
	Tensor3D Forward(const Tensor3D& input) override;

	// 逆伝播する
	// ・dOut(出力勾配)を受け取り 入力へ伝搬する
	// ・lastInput[h,w,c] > 0 の場合のみ dOut を通す
	// ・それ以外は 0
	Tensor3D Backward(const Tensor3D& dOut, float learningRate) override;

private:
	// Forward 時の入力を保存するテンソル(Backward で活性化関数の導関数に使う)
	Tensor3D lastInput;
};

