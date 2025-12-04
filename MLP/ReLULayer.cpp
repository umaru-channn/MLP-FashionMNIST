// ReLULayer.cpp
// CNN 活性化層（ReLU）の実装
// ・Forward : y = max(0, x)
// ・Backward: x > 0 のときだけ勾配を伝播、x <= 0 のとき 0
#include "ReLULayer.h"

// 順伝播する
Tensor3D ReLULayer::Forward(const Tensor3D& input)
{
	// 入力を保存する(逆伝播用)
	lastInput = input;
	// 入力テンソルの高さ・幅・チャネル数を取得する
	int H = input.GetH();
	int W = input.GetW();
	int C = input.GetC();

	// 出力テンソルを作成する
	Tensor3D out(H, W, C);
	// 各要素に ReLU を適用する
	for (int h = 0; h < H; h++) {
		for (int w = 0; w < W; w++) {
			for (int c = 0; c < C; c++) {
				// 入力値を取得する
				float v = input(h, w, c);
				// 0 より大きければそのまま、0 以下なら 0 にする
				out(h, w, c) = (v > 0.0f) ? v : 0.0f;
			}
		}
	}
	// 出力を返す
	return out;
}

// 逆伝播する
Tensor3D ReLULayer::Backward(const Tensor3D& dOut, float /*learningRate*/)
{
	// 勾配テンソルのサイズを取得する
	int H = dOut.GetH();
	int W = dOut.GetW();
	int C = dOut.GetC();
	// 入力側の勾配を格納するテンソルを作成する
	Tensor3D dInput(H, W, C);
	// 初期化(全要素を0 に)
	dInput.Zero();
	// 各要素ごとに勾配を計算する
	for (int h = 0; h < H; h++) {
		for (int w = 0; w < W; w++) {
			for (int c = 0; c < C; c++) {
				// 伝播時の入力値を参照する
				float x = lastInput(h, w, c);
				// 入力が正なら勾配をそのまま伝播する
				if (x > 0.0f) {
					dInput(h, w, c) = dOut(h, w, c);
				}
				else {
					// 入力が 0 以下なら勾配は 0
					dInput(h, w, c) = 0.0f;
				}
			}
		}
	}
	// 入力側への勾配を返す
	return dInput;
}
