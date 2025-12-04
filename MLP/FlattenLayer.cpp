// FlattenLayer.cpp
// CNN の 3D テンソルを 1次元ベクトルに変換する層
// ・Forward: Tensor3D → 1×1×N の Tensor3D
// ・Backward: 1×1×N → 元の H×W×C に復元

#include "FlattenLayer.h"

// Forward（順伝播）
// ・入力 Tensor3D（H × W × C）を 1 次元ベクトルに変換
// ・戻り値は 1×1×(H*W*C) の Tensor3D
Tensor3D FlattenLayer::Forward(const Tensor3D& input)
{
	// 入力形状を保存
	m_lastInput = input;
	inH = input.GetH();
	inW = input.GetW();
	inC = input.GetC();

	// 総要素数
	int total = inH * inW * inC;

	// 出力用ベクトルを確保
	m_flatOutput.resize(total);

	// Tensor3D → 1D ベクトルにコピー
	int idx = 0;
	for (int h = 0; h < inH; ++h) {
		for (int w = 0; w < inW; ++w) {
			for (int c = 0; c < inC; ++c)	{
				m_flatOutput[idx++] = input(h, w, c);
			}
		}
	}

	// 1×1×total の Tensor3D として返す
	Tensor3D out(1, 1, total);
	for (int i = 0; i < total; ++i) 	{
		out(0, 0, i) = m_flatOutput[i];
	}
	return out;
}

// 逆伝播する
// ・dOut: Flatten 出力(1×1×N)に対する勾配
// ・これを元の形状 H×W×C に戻す
Tensor3D FlattenLayer::Backward(const Tensor3D& dOut, float learningRate)
{
	int H = inH;
	int W = inW;
	int C = inC;

	// 1×1×(H*W*C) であることを確認する
	assert(dOut.GetH() == 1 && dOut.GetW() == 1);
	assert(dOut.GetC() == H * W * C);

	Tensor3D dInput(H, W, C);
	dInput.Zero();

	int idx = 0;
	for (int h = 0; h < H; ++h) {
		for (int w = 0; w < W; ++w) {
			for (int c = 0; c < C; ++c) {
				dInput(h, w, c) = dOut(0, 0, idx++);
			}
		}
	}
	return dInput;
}
