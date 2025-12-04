#pragma once
// FlattenLayer.h
// Flatten(平坦化)レイヤー
// ・CNN の畳み込み層やプーリング層の出力テンソル
//   (高さ×幅×チャンネル)
//   を1次元ベクトル配列に変換する。
// ・全結合層に渡すための必須ステップ
// ・逆伝播では1次元 dOutを元のテンソルに戻す

#include <vector>
#include <cassert>
#include "Tensor3D.h"
#include "IBaseLayer.h"

// FlattenLayer クラス
// Tensor3D → ベクトル(float配列) への変換層
class FlattenLayer : public IBaseLayer
{
public:
	// コンストラクタ(特に処理なし)
	FlattenLayer() = default;

	// Forward（順伝播）
	// ・Tensor3D を 1×1×(H*W*C) のテンソルに変換
	Tensor3D Forward(const Tensor3D& input) override;

	// Backward（逆伝播）
	// ・1×1×N の Tensor3D(Flatten の出力側)
	//   元の H×W×C の勾配に戻す
	Tensor3D Backward(const Tensor3D& dOut, float learningRate) override;
	// Forward結果を std::vector<float>として取得する
	const std::vector<float>& GetFlatOutput() const { return m_flatOutput; }

	// 入力形状取得用（FC層の入力次元計算に必要）
	int GetInputHeight()  const { return inH; }
	int GetInputWidth()   const { return inW; }
	int GetInputChannel() const { return inC; }

private:
	// Forward の入力を保持するテンソル(Backward で使う)
	Tensor3D m_lastInput;
	// Flatten 後の 1 次元ベクトル
	std::vector<float> m_flatOutput;

	// 元の入力形状
	int inH = 0;
	int inW = 0;
	int inC = 0;
};
