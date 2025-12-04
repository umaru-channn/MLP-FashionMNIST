// MaxPoolLayer.cpp
#include "MaxPoolLayer.h"
#include <cmath> 

// コンストラクタ
// ・poolSize : プーリング領域の一辺の長さ(例: 2 → 2×2 プーリング)
MaxPoolLayer::MaxPoolLayer(int poolSize)
	: 
	m_size(poolSize)
{
}

// 順伝播する
// ・入力特徴マップをsize×size単位で区切り その中の最大値を出力する
// ・最大値位置は Backward 時に必要なため入力と出力を保存する
Tensor3D MaxPoolLayer::Forward(const Tensor3D& inputFeatureMap)
{
	// 逆伝播(Backward)で最大値の場所を特定するため 入力特徴マップを保存する
	m_lastInputFeatureMap = inputFeatureMap;
	// 入力特徴マップの高さ(H) を取得する
	int H = inputFeatureMap.GetH();
	// 入力特徴マップの幅(W)を取得する
	int W = inputFeatureMap.GetW();
	// 入力特徴マップのチャネル数(C)を取得する
	int C = inputFeatureMap.GetC();
	// プーリング後の出力サイズ を取得する
	int outH = H / m_size;
	int outW = W / m_size;
	// 出力特徴マップ(outH × outW × C) を確保する
	Tensor3D out(outH, outW, C);
	// チャネルごとに最大値プーリングを実行する
	for (int c = 0; c < C; c++) {
		// 出力の高さ方向に走査する
		for (int oh = 0; oh < outH; oh++) {
			// 出力の幅方向に走査する
			for (int ow = 0; ow < outW; ow++) {
				// プーリング領域内の最大値を保持する
				float maxValue = -1e9f;  // 非常に小さい値で初期化
				// size×size のプーリング領域を探索して最大値を求める
				for (int kh = 0; kh < m_size; kh++) {
					for (int kw = 0; kw < m_size; kw++) {
						// 入力特徴マップ上の対応する位置(高さ)
						int ih = oh * m_size + kh;
						// 入力特徴マップ上の対応する位置(幅)
						int iw = ow * m_size + kw;
						// 対応する画素値を取得する
						float inputValue = inputFeatureMap(ih, iw, c);
						// 最大値を更新する
						if (inputValue > maxValue)	{
							maxValue = inputValue;
						}
					}
				}
				// プーリング領域から得られた最大値を出力特徴マップに格納する
				out(oh, ow, c) = maxValue;
			}
		}
	}

	// 逆伝播で最大位置を再判定するため Forward の出力も保存する
	m_lastOutputFeatureMap = out;
	// プーリング結果を返す
	return out;
}

// 逆伝播する
// ・dOutFeatureMap: 出力側の勾配 (outH×outW×C)
// ・戻り値: 入力側の勾配 (H×W×C)
Tensor3D MaxPoolLayer::Backward(const Tensor3D& dOutFeatureMap)
{
	// Forward 時の入力特徴マップのサイズを取得する
	int H = m_lastInputFeatureMap.GetH();
	int W = m_lastInputFeatureMap.GetW();
	int C = m_lastInputFeatureMap.GetC();

	// プーリング後の出力サイズを取得する(Forward と同じ計算)
	int outH = H / m_size;
	int outW = W / m_size;

	// 入力側の勾配マップを0で初期化
	// ・MaxPoolはパラメータを持たないため勾配は入力へ流す
	Tensor3D dInputFeatureMap(H, W, C);
	dInputFeatureMap.Zero();

	// 各チャネルごとに逆伝播処理を行う
	for (int channel = 0; channel < C; channel++) {
		// 出力特徴マップの高さ方向へループ
		for (int outY = 0; outY < outH; outY++) {
			// 出力特徴マップの幅方向へループ
			for (int outX = 0; outX < outW; outX++) {
				// Forward の出力に保存された最大値を取得する
				float maxValue = m_lastOutputFeatureMap(outY, outX, channel);
				// プーリング領域（size×size）を探索する
				for (int poolY = 0; poolY < m_size; poolY++) {
					for (int poolX = 0; poolX < m_size; poolX++) {
						// 入力側の対応する位置（pool の逆写像）
						int inY = outY * m_size + poolY;
						int inX = outX * m_size + poolX;
						// 入力値が Forward 時の最大値だった位置にだけ勾配を流す
						// MaxPool の逆伝播：最大値の位置にだけ誤差を伝える
						if (std::fabs(m_lastInputFeatureMap(inY, inX, channel) - maxValue) < 1e-6f)
						{
							// 最大値だった位置に dOut（次層からの勾配）を加算する
							dInputFeatureMap(inY, inX, channel) += dOutFeatureMap(outY, outX, channel);
						}
					}
				}
			}
		}
	}
	// 計算された入力側勾配を返す
	return dInputFeatureMap;
}
