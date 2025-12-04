// IBaseLayer.h
// CNN 用レイヤー共通インターフェース（3Dデータ用）
// ・Tensor3D を入出力とする畳み込み層/プーリング層/活性化層の基底クラス
// ・Forward（順伝播）と Backward（逆伝播）を純粋仮想関数で定義
// ・学習時には Backward で勾配伝播とパラメータ更新を行う
#pragma once
// 3次元テンソル型（Height × Width × Channels）
// ・前のステップで作成した Tensor3D をインクルードする
#include "Tensor3D.h"
// IBaseLayer クラス
// ・CNN の各レイヤー（Conv, Pool, ReLU など）の共通インターフェース
class IBaseLayer
{
public:
	// 仮想デストラクタ
	// ・派生クラスを delete するときに正しく解放されるようにする
	virtual ~IBaseLayer() = default;
	// Forward（順伝播）インターフェース
	// ・入力: Tensor3D (入力特徴マップ)
	// ・出力: Tensor3D (出力特徴マップ)
	// ・ConvLayer / MaxPoolLayer / ReLULayer などで実装される
	virtual Tensor3D Forward(const Tensor3D& input) = 0;
	// Backward（逆伝播）インターフェース
	// ・入力: dOut → 出力側から流れてきた勾配（Tensor3D）
	// ・引数: learningRate → 学習率（パラメータ更新に使用）
	// ・出力: dInput → 一つ前のレイヤーに渡す勾配（Tensor3D）
	// ・ConvLayer / MaxPoolLayer / ReLULayer などで実装される
	virtual Tensor3D Backward(const Tensor3D& dOut, float learningRate) = 0;
};


