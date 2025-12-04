// DisplayWindow.h
// 画像グリッド表示 + 詳細ビュー（拡大画像 + Top-10）の
// すべての GUI 操作を行うためのヘッダ
#pragma once
// ベクタを使う
#include <vector>
#include <string>
#include <utility>

// ウィンドウ初期化（Win32 API を使ったウィンドウ生成）
// width, height : ウィンドウのサイズ
// title         : 表示されるウィンドウタイトル
// 戻り値        : 成功すれば true
bool InitDisplayWindow(int width, int height, const wchar_t* title);

// グリッド画像（100枚）とラベル情報を GUI に渡して表示更新する
// images  : 100枚のグリッド画像（28×28）
// gtLabels  : 正解ラベル
// predLabels : 予測ラベル
// correctFlags : 正誤フラグ（true=正解）
// imgW,imgH : 1画像の大きさ（28,28）
// gridCols : グリッド列数（例：8）
// scale : グリッド画像の拡大表示倍率
void UpdateDisplayGridWithLabels(
	const std::vector<std::vector<uint8_t>>& images,
	const std::vector<int>& gtLabels,
	const std::vector<int>& predLabels,
	const std::vector<bool>& correctFlags,
	int imgW,
	int imgH,
	int gridCols,
	int scale
);

// 右側の詳細ビューを更新する
// image : 拡大表示する1枚の画像（28×28）
// top10 : Top-10 の (classID, probability) のペア
// top10Names : Top-10 のクラス名（"Sneaker" など）
void UpdateDetailView(
	const std::vector<uint8_t>& image,
	const std::vector<std::pair<int, float>>& top10,
	const std::vector<std::wstring>& top10Names
);
// トレーニング進捗バーの更新
// p : 0.0 ～ 1.0 の範囲
void SetTrainProgress(float p);
// Win32 メッセージ処理（画面再描画など）
void PumpWindowMessages();
