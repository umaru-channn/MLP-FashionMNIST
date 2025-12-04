// main.cpp
// CNN による Fashion-MNIST 画像認識
// ・28×28画像 → CNN(2層Conv + 2層MaxPool + 2層FC)
// ・学習中に一定ステップごとに画像を更新表示
// ・右側に拡大画像 + Top-10 横棒グラフをGUI表示

#include <iostream>
#include <random>
#include <conio.h>
#include <algorithm>
#include <numeric>
#include "FashionMNIST.h"
#include "Tensor3D.h"
#include "CNNModel.h"
#include "DisplayWindow.h"   // 100画像グリッド + 詳細表示（Top-10）

// 学習何ステップごとに画面更新するか
constexpr int VISUAL_INTERVAL = 100;

// プロトタイプ宣言(TrainOneEpoch から実行する)
// ランダムイメージを表示する
void ShowRandomImages(CNNModel& model, FashionMNIST& mnist);

// 28×28 グレースケール画像 → Tensor3D(28×28×1) に変換する
Tensor3D ImageToTensor(const std::vector<uint8_t>& imges)
{
	// テンソル(高さ 28, 幅 28, チャネル 1)
	Tensor3D tensor(28, 28, 1);
	// 行インデックスを処理する
	for (int row = 0; row < 28; row++)
	{
		// 列インデックスを処理する
		for (int column = 0; column < 28; column++)
		{
			// 正規化された画素値(0〜255 → 0〜1)を計算する
			float pixelValue = imges[row * 28 + column] / 255.0f;
			// テンソルに画素値を格納する
			tensor(row, column, 0) = pixelValue;
		}
	}
	// テンソルを返す
	return tensor;
}

// one-hot ベクトル生成する（10クラス）
std::vector<float> OneHot(int label)
{
	// 全要素 0 で長さ 10 のベクトルを作成する
	std::vector<float> oneHotVector(10, 0.0f);
	// 該当クラスのみ 1.0 にする
	oneHotVector[label] = 1.0f;
	return oneHotVector;
}

// CNN 学習を1エポック実行する
void TrainOneEpoch(CNNModel& model, FashionMNIST& mnist, 	float learningRate, int epochIndex, int totalEpochs)
{
	// 利用画像枚数は最大5000枚に設定する (デバッグ用: 全データを使うなら変更可能)
	size_t trainCount = std::min(mnist.trainImages.size(), (size_t)5000);
	// 学習に使うインデックス配列 [0,1,...,trainCount-1] を用意する
	std::vector<int> indices((int)trainCount);
	std::iota(indices.begin(), indices.end(), 0);
	// 各エポックごとにデータをシャッフルして汎化性能を上げる
	std::mt19937 rng(std::random_device{}());
	std::shuffle(indices.begin(), indices.end(), rng);

	// 総損失を初期化する
	float totalLoss = 0.0f;
	// 正解数を初期化する
	int correct = 0;
	// 各サンプルに対して順伝播＋逆伝播を行う (SGD)
	for (int sampleIndex = 0; sampleIndex < static_cast<int>(trainCount); sampleIndex++)
	{
		// シャッフルされたインデックスを取得する
		int idx = indices[sampleIndex];
		// 画像を取得する
		const auto& images = mnist.trainImages[idx];
		// ラベルを取得する
		int label = mnist.trainLabels[idx];
		// テンソルに変換する
		Tensor3D tensor = ImageToTensor(images);
		// 順伝播（確率分布が返ってくる）
		auto probability = model.Forward(tensor);
		// 損失計算用ターゲット(one-hot)
		auto target = OneHot(label);
		model.SetTarget(target);
		// 損失を計算する
		float loss = model.ComputeLoss(target);
		// 総損失を加算する
		totalLoss += loss;
		// 逆伝播する (SGD 更新)
		model.Backward(learningRate);
		// 推論結果 (確率分布) の中で最も値が大きい要素のインデックスを計算する
		int prediction = static_cast<int>(std::max_element(probability.begin(), probability.end()) - probability.begin());
		// 正解数をカウントする
		if (prediction == label) correct++;
		// VISUAL_INTERVAL ステップごとに画像更新する
		if (sampleIndex % VISUAL_INTERVAL == 0)
		{
			// エポックと サンプルインデックスを表示する
			std::wcout << L"[Epoch " << (epochIndex + 1) << L"] Update at step " << sampleIndex << L"\n";
			// ランダムイメージを表示する
			ShowRandomImages(model, mnist);
			// 再描画する
			PumpWindowMessages();
		}
		// プログレスバーを更新する
		float progress = static_cast<float>(sampleIndex) / static_cast<float>(trainCount);
		// 学習進捗を設定する（0～1 の値）
		SetTrainProgress((epochIndex + progress) / totalEpochs);
	}
	// 平均損失を計算する
	float avgLoss = totalLoss / static_cast<float>(trainCount);
	// 精度(%)を計算する
	float accuracy = correct * 100.0f / static_cast<float>(trainCount);
	// 結果を表示する
	std::wcout << L"Epoch " << (epochIndex + 1) << L" | Loss = " << avgLoss << L" | Accuracy = " << accuracy << L"%\n";
}

// CNN の推論結果を GUI に送る(100枚ランダム表示)
void ShowRandomImages(CNNModel& model, FashionMNIST& mnist)
{
	// 表示枚数(最大100枚)を決定する
	int count = std::min(100, static_cast<int>(mnist.trainImages.size()));
	// 画像データを格納する配列を準備する
	std::vector<std::vector<uint8_t>> images(count);
	// 正解ラベルを格納する配列を準備する
	std::vector<int> groundTruth(count);
	// 予測ラベルを格納する配列を準備する
	std::vector<int> prediction(count);
	// 正誤フラグを格納する配列を準備する
	std::vector<bool> correctFlags(count);
	// ランダムにインデックスを生成するための乱数エンジンを宣言する
	std::mt19937 random(std::random_device{}());
	// ランダムにインデックスを生成するための分布を宣言する
	std::uniform_int_distribution<int> dist(0, static_cast<int>(mnist.trainImages.size()) - 1);
	// 指定枚数分ランダムにサンプルを選び、推論結果を計算する
	for (int sampleIndex = 0; sampleIndex < count; sampleIndex++)
	{
		// ランダムに選んだサンプルのインデックスを設定する
		int randomIndex = dist(random);
		// 画像を取得する
		images[sampleIndex] = mnist.trainImages[randomIndex];
		// 正解ラベルを取得する
		groundTruth[sampleIndex] = mnist.trainLabels[randomIndex];
		// 画像をテンソルに変換してモデルに入力する
		Tensor3D inputTensor = ImageToTensor(images[sampleIndex]);
		// モデルの予測ラベルを取得する
		prediction[sampleIndex] = model.Predict(inputTensor);
		// 予測が正解かどうかを判定してフラグに記録する
		correctFlags[sampleIndex] = (prediction[sampleIndex] == groundTruth[sampleIndex]);
	}
	// 左側のグリッドに画像とラベルを表示する（scale=2 → 2倍拡大表示）
	UpdateDisplayGridWithLabels(images, groundTruth, prediction, correctFlags, 28, 28, 10, 2);
	// 詳細ビュー用に先頭の画像 (0番目) をテンソルに変換する
	Tensor3D inputTensor = ImageToTensor(images[0]);
	// CNN モデルから Top-10 の予測結果を取得する
	auto top10 = model.GetTop10(inputTensor);
	// Top-10 の予測結果に対応するクラス名を取得する
	auto top10names = model.GetTop10Names(top10);
	// 詳細ビューを更新する(画像と Top-10 推定結果を表示)
	UpdateDetailView(images[0], top10, top10names);
	// 再描画する
	PumpWindowMessages();
}

// メインエントリ
int main()
{
	// Fashion MNISTデータセットを読み込む
	FashionMNIST mnist;
	// FashionMNISTデータセットをロードする
	if (!mnist.Load("train-images-idx3-ubyte", "train-labels-idx1-ubyte", true))
	{ std::cerr << "Error: MNIST 読み込み失敗\n"; 	return 1; 	}

	// CNNのインスタンスを生成する
	CNNModel model;
	// GUI ウィンドウを初期化する
	InitDisplayWindow(1200, 980, L"CNN FashionMNIST Viewer");
	// 再描画する
	PumpWindowMessages();
	// まだ学習していない最初のイメージを表示する
	ShowRandomImages(model, mnist);
	// 学習回数を設定する
	const int epochs = 8;
	// 学習率を設定する (0.0003 → 0.001 に上げて学習を進みやすくする)
	float learningRate = 0.006f;
	// 各エポックで学習を行う
	for (int epoch = 0; epoch < epochs; epoch++)
	{
		// 1エポック学習する
		TrainOneEpoch(model, mnist, learningRate, epoch, epochs);
		// 各エポック終了時にも1回画面更新
		ShowRandomImages(model, mnist);
		// 再描画する
		PumpWindowMessages();
	}

	// ポーズする
	std::cout << "Training Finished. Press any key to exit...";
	// 最終結果を表示する
	ShowRandomImages(model, mnist);
	PumpWindowMessages();
	// キー入力待ち
	int key = _getch();
	(void)key;
	return 0;
}
